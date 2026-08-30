#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 trait_epistasis_to_allocation.py -- one trait, from validation to allocation,
                                     in one command
================================================================================

 Scoring, per-pair validation and allocation are joined into one pipeline that
 runs automatically for a single trait:

   [1] genome-wide scoring   each fold (5 by default) independently scans
                             6.4e8 pairs inside its training set and keeps the
                             top --top_k
   [2] per-fold validation   every candidate pair gets its own kernel
                             (A + D + aggregate + that pair) and is evaluated
                             on every held-out fold. The criteria for a usable
                             pair are fixed in the code, before the analysis:
                               - it improves prediction in EVERY fold
                               - min_d > 0, so even the worst fold gains
                             A fold with no discriminating power, where any
                             kernel raises accuracy, still counts: the code
                             does not decide on the data's behalf which folds
                             are worth listening to.
   [3] decision              usable pairs -> an additive-preserving strategy;
                             none -> a purely additive (ADD) plan
   [4] allocation            six arms compared: ADD / ADD+DOM / FULL_naive /
                             two_stage / weighted / add_floor, each reporting
                             mean_gv_add and add_retained_vs_ADD, so it is
                             visible at a glance which one preserved additive
                             merit

 Parallelism: step [2], the expensive one, is distributed over (pair, fold)
 units. --jobs 6 --threads_per_job 4 uses 24 cores. Step [1] is chunked and
 benefits from BLAS threading.

 Checkpointing and multiple outputs, so an interruption or a logic error still
 leaves usable results:
   <out>/_scores/<trait>_pool_seed*.json     per-fold scoring pool (existing
                                             pools can be dropped in here)
   <out>/per_pair_per_fold.csv               one row per (pair, fold), written
                                             as it goes
   <out>/pair_summary.csv                    all_folds and min_d per pair
   <out>/usable_pairs.csv                    the pairs judged usable
   <out>/<trait>_all_crosses.csv             add / dom / epi for every cross
   <out>/<trait>_strategy_summary.csv        the six arms (the main table)
   <out>/<trait>_mating_plan.csv             the actual matings per arm
   <out>/STATUS.json                         how far the pipeline has got
 Completed steps are skipped on a rerun, judged from STATUS and the presence of
 each csv.

     python trait_epistasis_to_allocation.py --bfile <plink_prefix> --trait bf \
         --types MULT --top_k 30 --folds 5 \
         --jobs 6 --threads_per_job 4 \
         --reuse_scores focal_validation/_scores \
         --out pipe_bf
================================================================================
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from itertools import combinations
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

import numpy as np
import pandas as pd


def say(*a):
    print(*a, flush=True)


def jdump(o, p):
    with open(p, "w") as f:
        json.dump(o, f, indent=2, default=str)


# ----- per-worker shared state for the validation step ---------------------
# Worker processes do not reload the genotypes or rebuild the kernels. At
# n=2796, m=35750 each of X / Ha / Hd / Xc is about 0.8GB, so one copy per
# worker at jobs=6 would need about 19GB for those matrices alone, and with the
#
# kernels on top the process is likely to be OOM-killed, which surfaces as
# BrokenProcessPool.
#
# Instead the large matrices are built once in the parent process and stored in
# the module-level dict _V below. With fork, the default start method on Linux,
# every child inherits that memory by copy-on-write: nothing is reloaded and
# nothing is pickled, so memory does not grow with --jobs. As long as the
# children never write to those arrays, and they do not, no copy is made.
_V = {}


def _set_threads(n_threads):
    """Worker initialisation sets the BLAS thread count and loads nothing."""
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS"):
        os.environ[v] = str(n_threads)


def _val_cell(task):
    """Evaluate one (pair, fold). task = (i, j, tag, seed). Uses _V as built in
    the parent process and inherited through fork; nothing is reloaded."""
    i, j, tag, seed = task
    Q = _V["Q"]
    if seed not in _V["base"]:
        return None
    accA, ya, tr, te = _V["base"][seed]
    KT = Q.pairs_kernel([(1.0, i, j, tag)], tag, _V["X"], _V["Ha"], _V["Hd"],
                        _V["Xc"])
    if KT is None:
        return None
    Kl = dict(_V["K"]); Kl["TOP"] = KT
    g1, _, _ = Q.fit_gebv(_V["y"], _V["Xfix"], Kl,
                          _V["base_keys"] + ["TOP"], tr, te)
    if g1 is None:
        return None
    return dict(snp_i=_V["snp_names"][i], snp_j=_V["snp_names"][j],
                chr_i=int(_V["chrom"][i]), chr_j=int(_V["chrom"][j]),
                pos_i=int(_V["pos"][i]), pos_j=int(_V["pos"][j]),
                seed=seed, tag=tag, acc_agg=accA,
                acc_with_pair=Q.pcc(ya, g1),
                d_vs_AGG=Q.pcc(ya, g1) - accA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bfile", required=True,
                    help="PLINK prefix (.bed/.bim/.fam) of the QC'd, pruned "
                         "genotypes")
    ap.add_argument("--trait", required=True)
    ap.add_argument("--pheno")
    ap.add_argument("--types", nargs="+", default=["MULT"])
    ap.add_argument("--top_k", type=int, default=30)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--threads_per_job", type=int, default=4)
    ap.add_argument("--cache_dir", default="adept_kernel_cache",
                    help="where exact kernels are cached; kernels depend only "
                         "on the genotypes, so one directory per dataset")
    ap.add_argument("--no_dominance", action="store_true",
                    help="block the dominance module. Set automatically when "
                         "mean heterozygosity is at or below 0.02 (inbred "
                         "lines, doubled haploids). Blocking is a statement "
                         "about the material, not about the trait.")
    ap.add_argument("--reuse_scores", default=None,
                    help="directory of existing scoring pools (<trait>_pool_seed*.json) to reuse")
    # allocation
    ap.add_argument("--n_candidates", type=int, default=100)
    ap.add_argument("--n_crosses", type=int, default=50)
    ap.add_argument("--max_use", type=int, default=4)
    ap.add_argument("--f_max", type=float, default=0.10)
    ap.add_argument("--add_quantile", type=float, default=0.70)
    ap.add_argument("--floor_quantile", type=float, default=0.80)
    ap.add_argument("--out", default="pipe")
    ap.add_argument("--seed", type=int, default=2024)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    ck = os.path.join(a.out, "_scores"); os.makedirs(ck, exist_ok=True)
    status_p = os.path.join(a.out, "STATUS.json")
    pheno = a.pheno or f"{a.trait}.phe"
    if not os.path.exists(pheno):
        raise SystemExit(f"phenotype file not found: {pheno}")

    import pig_epik_pipeline as P
    import pig_epistasis_pipeline as Q

    # ---------- loaded once in the parent, used by scoring and allocation ----------
    pc = P.CFG(); pc.BFILE, pc.N_PCS = a.bfile, 0
    pc.PHENO_FILE, pc.TRAIT = pheno, a.trait
    pc.COV_FILE = pc.QCOV_FILE = None
    X, y, chrom, pos, snp_names, _ = P.load_real(pc)
    n, m = X.shape
    say(f"[data] {n} x {m}  trait={a.trait}")
    Ha, Hd = P.noia(X)
    Xx = (X > 0).astype(np.float64) * 2 - 1
    xc = Xx - Xx.mean(0); sd = xc.std(0); sd[sd < 1e-12] = 1.0
    Xc = xc / sd; del Xx, xc
    qc = Q.CFG(); qc.CACHE_DIR = a.cache_dir; qc.DO_CIS = False
    K, meta = Q.build_kernels(qc, X, Ha, Hd, chrom, keep_xc=False)
    labels, _ = P.structure_clusters(K["A"], None, 10, a.seed)
    agg = "MULT_t" if "MULT_t" in K else "HAD"

    # ---------- capability detection, before anything is fitted ----------
    from allocation_core import mean_heterozygosity, HET_BLOCK_THRESHOLD
    het = mean_heterozygosity(X)
    if not a.no_dominance and het <= HET_BLOCK_THRESHOLD:
        a.no_dominance = True
        say(f"  [capability] mean heterozygosity {het:.4f} <= "
            f"{HET_BLOCK_THRESHOLD}: dominance module BLOCKED. The screening "
            f"baseline becomes A + {agg} rather than A + D + {agg}, and the "
            f"ADD+DOM arm is not produced. This records what the material can "
            f"support; it is not a finding that dominance is absent.")
    base_keys = ["A"] if a.no_dominance else ["A", "D"]
    base_keys = base_keys + [agg]
    say(f"  [capability] mean heterozygosity {het:.4f}; baseline model "
        f"{' + '.join(base_keys)}")
    cfg = Q.CFG()
    name_to_idx = {s: k for k, s in enumerate(snp_names)}

    # ========== [1] scoring, per fold, reusable ==========
    folds, pools = [], {}
    for s in range(a.folds):
        seed = a.seed + s * 101
        tr, te = Q.make_split(n, labels, cfg.SPLIT, cfg.TEST_FRAC, seed)
        folds.append((seed, tr, te))
        fn = f"{a.trait}_pool_seed{seed}_k{a.top_k}.json"
        # reuse: look in reuse_scores first, then the local _scores directory
        src = None
        for d in ([a.reuse_scores] if a.reuse_scores else []) + [ck]:
            p = os.path.join(d, fn)
            if os.path.exists(p):
                src = p; break
            # also accept the older naming without the _k suffix
            p2 = os.path.join(d, f"{a.trait}_pool_seed{seed}.json")
            if os.path.exists(p2):
                src = p2; break
        if src:
            pools[seed] = {t: [tuple(x) for x in v]
                           for t, v in json.load(open(src)).items()}
            say(f"  [1] fold {s+1}: reusing scores from {src}")
            continue
        t0 = time.time()
        tops = Q.score_genomewide(X[tr], y[tr], tuple(a.types), a.top_k,
                                  cfg.SCAN_CHUNK, cfg.MIN_CELL, verbose=(s == 0))
        pools[seed] = tops
        json.dump({t: [list(x) for x in v] for t, v in tops.items()},
                  open(os.path.join(ck, fn), "w"))
        say(f"  [1] fold {s+1}: scored in {time.time()-t0:.0f}s")

    # candidate pairs: the union of the per-fold top_k
    cand_pairs = {}
    for typ in a.types:
        seen = {}
        for seed, _, _ in folds:
            for (v, i, j, tag) in pools[seed].get(typ, []):
                key = (int(min(i, j)), int(max(i, j)), typ)
                seen[key] = seen.get(key, 0) + 1
        cand_pairs[typ] = seen
    total_pairs = sum(len(v) for v in cand_pairs.values())
    say(f"  [1] {total_pairs} candidate pair(s) in total (by type: "
        + ", ".join(f"{t}={len(v)}" for t, v in cand_pairs.items()) + "）")

    # ---- the parent computes each fold's baseline (A+D+aggregate) once and
    # stores it in _V together with the large matrices. This must happen BEFORE
    # the pool is created: at fork time every child inherits the parent's memory
    # as it stands, and reads it without writing, so memory does not grow with
    # the number of --jobs.
    _set_threads(a.threads_per_job)
    Xfix = np.ones((n, 1))
    base = {}
    for seed, tr, te in folds:
        gA, ya, _ = Q.fit_gebv(y, Xfix, K, base_keys, tr, te)
        if gA is not None:
            base[seed] = (Q.pcc(ya, gA), ya, tr, te)
    _V.update(X=X, y=y, Ha=Ha, Hd=Hd, Xc=Xc, K=K, agg=agg, Xfix=Xfix,
              base_keys=base_keys, snp_names=snp_names, chrom=chrom, pos=pos,
              base=base, Q=Q)
    say(f"  [parent] baselines built, {len(base)}/{a.folds} fold(s) usable; "
        f"forking {a.jobs} worker(s), which inherit memory rather than reloading")

    # ========== [2] per-pair, per-fold validation, in parallel ==========
    res_path = os.path.join(a.out, "per_pair_per_fold.csv")
    done = set()
    # A file that exists but is empty means the previous run was killed before
    # writing anything (an OOM kill, for instance). It must not be read as
    # checkpoint data, or pandas raises EmptyDataError; size zero is treated as
    # "not started".
    if os.path.exists(res_path) and os.path.getsize(res_path) > 0:
        try:
            prev = pd.read_csv(res_path)
        except pd.errors.EmptyDataError:
            prev = None
        if prev is not None and len(prev):
            done = {(r.snp_i, r.snp_j, int(r.seed)) for r in prev.itertuples()}
            say(f"  [2] resuming: {len(done)} (pair, fold) unit(s) already done")
        else:
            say(f"  [2] {res_path} exists but is empty, left by an interrupted run; "
                f"treating it as not started")
    elif os.path.exists(res_path):
        say(f"  [2] {res_path} is a zero-byte file left by an interrupted run; "
            f"treating it as not started")

    tasks = []
    for typ, seen in cand_pairs.items():
        for (i, j, _t) in seen:
            for seed, _, _ in folds:
                if (snp_names[i], snp_names[j], seed) in done:
                    continue
                tasks.append((i, j, typ, seed))
    say(f"  [2] {len(tasks)} (pair, fold) unit(s) to evaluate over {a.jobs} process(es)")

    if tasks:
        t0 = time.time()
        n_done = 0
        try:
            ctx = mp.get_context("fork")
        except ValueError:
            raise SystemExit(
                "This machine does not support the fork start method, which is "
                "common outside Linux. The code relies on fork to share the large "
                "matrices and kernels with the workers without copying. Under "
                "spawn every worker reloads all the data and memory grows linearly "
                "with --jobs, which exhausts it. Run on Linux, or use --jobs 1.")
        with open(res_path, "a") as f:
            write_header = not done and os.path.getsize(res_path) == 0 \
                if os.path.exists(res_path) else True
            with ProcessPoolExecutor(max_workers=a.jobs, mp_context=ctx,
                                     initializer=_set_threads,
                                     initargs=(a.threads_per_job,)) as ex:
                futs = [ex.submit(_val_cell, t) for t in tasks]
                for fut in as_completed(futs):
                    r = fut.result()
                    if r is None:
                        continue
                    row = pd.DataFrame([r])
                    row.to_csv(f, header=write_header, index=False)
                    write_header = False
                    f.flush()
                    n_done += 1
                    if n_done % 50 == 0:
                        el = time.time() - t0
                        rate = n_done / el
                        eta = (len(tasks) - n_done) / rate / 60 if rate else 0
                        say(f"  [2] {n_done}/{len(tasks)}  "
                            f"{el/60:.0f}min  ETA {eta:.0f}min")
        say(f"  [2] validation finished in {(time.time()-t0)/60:.0f} min")

    R = pd.read_csv(res_path)

    # ========== summary and the usable-pair criteria ==========
    rows = []
    for (si, sj), g in R.groupby(["snp_i", "snp_j"]):
        nf = len(g); npos = int((g.d_vs_AGG > 0).sum())
        rows.append(dict(
            trait=a.trait, type=g.tag.iloc[0], snp_i=si, snp_j=sj,
            chr_i=int(g.chr_i.iloc[0]), chr_j=int(g.chr_j.iloc[0]),
            n_folds_tested=nf, n_folds_improved=npos,
            all_folds=bool(npos == nf and nf == a.folds),
            mean_d=float(g.d_vs_AGG.mean()), min_d=float(g.d_vs_AGG.min()),
            max_d=float(g.d_vs_AGG.max())))
    S = pd.DataFrame(rows).sort_values(
        ["all_folds", "min_d", "mean_d"], ascending=[False, False, False])
    S.to_csv(os.path.join(a.out, "pair_summary.csv"), index=False)

    # criteria, fixed in the code: an improvement in every fold and min_d > 0
    usable = S[(S.all_folds) & (S.min_d > 0)].copy()
    usable.to_csv(os.path.join(a.out, "usable_pairs.csv"), index=False)
    say(f"\n  usable epistatic pairs: {len(usable)} / {len(S)} "
        f"(criteria: improves in all {a.folds} folds and min_d > 0)")
    if len(usable):
        say(usable[["snp_i", "snp_j", "min_d", "mean_d"]].head(20)
            .round(5).to_string(index=False))

    # ========== [3][4] allocation ==========
    per_fold_frac = {seed: float((R[R.seed == seed].d_vs_AGG > 0).mean())
                     for seed, _, _ in folds}
    jdump(dict(per_fold_improve_fraction=per_fold_frac,
               note="informational: a value near 1.0 means that fold has little "
                    "discriminating power, but it still counts towards the criteria"),
          os.path.join(a.out, "per_fold_discrimination.json"))

    from allocation_core import run_strategies

    if len(usable) == 0:
        say("\n  [3] no usable epistatic pair -> this trait falls back to a purely additive (ADD) plan")
        summ = run_strategies(
            a, X, y, chrom, pos, snp_names, Ha, Hd, Xc, K, meta,
            agg, sel=[], out_dir=a.out, no_dominance=a.no_dominance)
    else:
        sel = [(1.0, name_to_idx[r.snp_i], name_to_idx[r.snp_j], r.type)
               for r in usable.itertuples()]
        say(f"\n  [3] {len(sel)} usable pair(s) -> additive-preserving allocation")
        summ = run_strategies(
            a, X, y, chrom, pos, snp_names, Ha, Hd, Xc, K, meta,
            agg, sel=sel, out_dir=a.out, no_dominance=a.no_dominance)

    say("\n" + "=" * 78)
    say(summ.round(4).to_string(index=False))
    say("=" * 78)
    jdump(dict(done=True, n_usable_pairs=len(usable),
               n_candidate_pairs=total_pairs,
               mean_heterozygosity=het,
               dominance_blocked=bool(a.no_dominance),
               baseline_model=" + ".join(base_keys),
               n_splits_for_pair_validation=a.folds), status_p)
    say(f"\n-> {a.out}/  (STATUS.json, pair_summary.csv, usable_pairs.csv, "
        f"{a.trait}_strategy_summary.csv and others)")


if __name__ == "__main__":
    main()