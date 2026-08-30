#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 focal_pair_perfold.py -- test every candidate pair ONE AT A TIME, in every
                          fold; keep the ones that improve prediction in all
                          of them
================================================================================

 THE DESIGN
 ------------------------------------------------------------------------------
 Each fold is scored genome-wide and produces its own ranked list of candidate
 pairs. Every pair from the union of those lists is then fitted ALONE (a
 one-pair kernel on top of A + D + aggregate) and evaluated on every fold's
 held-out test set. A pair counts as reliable when it raises held-out accuracy
 in EVERY fold.

 Testing each pair in every fold -- not only in the fold that nominated it --
 is deliberate: a pair that ranked 31st in one fold and 3rd in the others is
 still a candidate worth a verdict, and the full 5-fold record is what makes
 "improves everywhere" a meaningful statement rather than an artefact of which
 list it happened to make.

 A FOLD THAT PASSES EVERYTHING PASSES NOTHING
 ------------------------------------------------------------------------------
 In the earlier bf run, five pair sets sharing almost no pairs all returned
 +0.210 to +0.215 on fold seed=2327. When a fold rewards any kernel equally,
 "improved in that fold" carries no information, and an all-folds criterion
 quietly becomes a four-fold criterion with one free pass. This script
 therefore reports, per fold, the FRACTION of tested pairs that improved. A
 fold near 1.00 is contributing no discrimination and should be read as such
 -- that diagnostic is free, it needs no extra fitting.

 COST
 ------------------------------------------------------------------------------
 One REML fit per (pair, fold). The first fit is timed and the total projected
 before the run commits, so the number comes from measurement rather than
 guesswork. Every result is appended to disk as it is produced and completed
 (pair, fold) cells are skipped on restart.

     python focal_pair_perfold.py --traits bf --types MULT \
         --top_k 30 --folds 5 --out perfold_bf
================================================================================
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pig_epik_pipeline as P
import pig_epistasis_pipeline as Q


def say(*a):
    print(*a, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bfile", required=True,
                    help="PLINK prefix (.bed/.bim/.fam) of the "
                         "QC'd, pruned genotypes")
    ap.add_argument("--traits", nargs="+", default=["bf"])
    ap.add_argument("--types", nargs="+", default=["MULT"])
    ap.add_argument("--top_k", type=int, default=30)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--score_dir", default=None,
                    help="reuse an earlier run's _scores directory")
    ap.add_argument("--max_pairs", type=int, default=0,
                    help="cap the number of unique pairs tested (0 = all). "
                         "Pairs are ordered by how many folds nominated them, "
                         "then by best score, so a cap keeps the most "
                         "reproducible candidates.")
    ap.add_argument("--cache_dir", default="adept_kernel_cache",
                    help="where exact kernels are cached; they depend "
                         "only on the genotypes, so use one directory "
                         "per dataset")
    ap.add_argument("--out", default="perfold")
    ap.add_argument("--seed", type=int, default=2024)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    ck = a.score_dir or os.path.join(a.out, "_scores")
    os.makedirs(ck, exist_ok=True)
    res_path = os.path.join(a.out, "per_pair_per_fold.csv")

    cfg = Q.CFG()
    pc = P.CFG()
    pc.BFILE, pc.N_PCS = a.bfile, 0
    pc.PHENO_FILE, pc.TRAIT = f"{a.traits[0]}.phe", a.traits[0]
    pc.COV_FILE = pc.QCOV_FILE = None
    X, _, chrom, pos, snp_names, _ = P.load_real(pc)
    n, m = X.shape
    say(f"[data] {n} animals x {m} SNPs")
    Ha, Hd = P.noia(X)
    Xx = (X > 0).astype(np.float64) * 2 - 1
    xc = Xx - Xx.mean(0); sd = xc.std(0); sd[sd < 1e-12] = 1.0
    Xc = xc / sd
    del Xx, xc

    qc = Q.CFG(); qc.CACHE_DIR = a.cache_dir; qc.DO_CIS = False
    K, meta = Q.build_kernels(qc, X, Ha, Hd, chrom, keep_xc=False)
    labels, _ = P.structure_clusters(K["A"], None, 10, a.seed)
    agg = "MULT_t" if "MULT_t" in K else "HAD"
    Xfix = np.ones((n, 1))

    done = set()
    if os.path.exists(res_path):
        prev = pd.read_csv(res_path)
        done = {(r.trait, r.type, r.snp_i, r.snp_j, int(r.seed))
                for r in prev.itertuples()}
        say(f"[resume] {len(done)} (pair, fold) cells already on disk")

    for trait in a.traits:
        if not os.path.exists(f"{trait}.phe"):
            say(f"[skip] {trait}.phe not found")
            continue
        say("\n" + "=" * 74)
        say(f"TRAIT {trait}")
        say("=" * 74)
        pc.PHENO_FILE, pc.TRAIT = f"{trait}.phe", trait
        Xt_, y, _, _, _, _ = P.load_real(pc)
        if Xt_.shape[0] != n:
            say(f"  [SKIP] {trait}: {Xt_.shape[0]} phenotyped vs {n}; run alone")
            del Xt_
            continue
        del Xt_

        folds, pools = [], {}
        for s in range(a.folds):
            seed = a.seed + s * 101
            tr, te = Q.make_split(n, labels, cfg.SPLIT, cfg.TEST_FRAC, seed)
            folds.append((seed, tr, te))
            f = os.path.join(ck, f"{trait}_pool_seed{seed}_k{a.top_k}.json")
            if os.path.exists(f):
                pools[seed] = {t: [tuple(x) for x in v]
                               for t, v in json.load(open(f)).items()}
                say(f"  fold {s+1}: pool from checkpoint")
                continue
            t0 = time.time()
            tops = Q.score_genomewide(X[tr], y[tr], tuple(a.types), a.top_k,
                                      cfg.SCAN_CHUNK, cfg.MIN_CELL,
                                      verbose=(s == 0))
            pools[seed] = tops
            json.dump({t: [list(x) for x in v] for t, v in tops.items()},
                      open(f, "w"))
            say(f"  fold {s+1}: scored in {time.time()-t0:.0f}s")

        # baseline per fold, computed once
        base = {}
        for seed, tr, te in folds:
            gA, ya, _ = Q.fit_gebv(y, Xfix, K, ["A", "D", agg], tr, te)
            if gA is None:
                continue
            base[seed] = (Q.pcc(ya, gA), ya)

        for typ in a.types:
            # union of candidates, with the number of folds that nominated each
            nom, best = {}, {}
            for seed, _, _ in folds:
                for (v, i, j, tag) in pools[seed].get(typ, []):
                    key = (int(min(i, j)), int(max(i, j)))
                    nom[key] = nom.get(key, 0) + 1
                    if key not in best or v > best[key][0]:
                        best[key] = (v, i, j, tag)
            cands = sorted(nom, key=lambda k: (-nom[k], -best[k][0]))
            if a.max_pairs:
                cands = cands[:a.max_pairs]
            say(f"\n  [{typ}] {len(cands)} unique candidate pairs across "
                f"{a.folds} folds "
                f"(nominated by: " + ", ".join(
                    f"{k}fold={sum(1 for c in cands if nom[c]==k)}"
                    for k in sorted({nom[c] for c in cands}, reverse=True))
                + ")")

            todo = [(c, seed) for c in cands for seed, _, _ in folds
                    if (trait, typ, snp_names[c[0]], snp_names[c[1]], seed)
                    not in done]
            if not todo:
                say("      all cells already computed")
                continue

            # measure one fit, then project -- no guessed estimates
            t0 = time.time()
            c0, s0 = todo[0]
            KT = Q.pairs_kernel([best[c0]], typ, X, Ha, Hd, Xc)
            tr0, te0 = next((tr, te) for sd_, tr, te in folds if sd_ == s0)
            Kl = dict(K); Kl["TOP"] = KT
            Q.fit_gebv(y, Xfix, Kl, ["A", "D", agg, "TOP"], tr0, te0)
            t_fit = time.time() - t0
            say(f"      one (pair, fold) fit = {t_fit:.1f}s; {len(todo)} cells "
                f"remaining -> ESTIMATED {t_fit*len(todo)/3600:.1f} h")

            for ci, (key, seed) in enumerate(todo, 1):
                if seed not in base:
                    continue
                accA, ya = base[seed]
                tr, te = next((tr, te) for sd_, tr, te in folds if sd_ == seed)
                KT = Q.pairs_kernel([best[key]], typ, X, Ha, Hd, Xc)
                if KT is None:
                    continue
                Kl = dict(K); Kl["TOP"] = KT
                g1, _, _ = Q.fit_gebv(y, Xfix, Kl, ["A", "D", agg, "TOP"],
                                      tr, te)
                if g1 is None:
                    continue
                row = dict(trait=trait, type=typ,
                           snp_i=snp_names[key[0]], snp_j=snp_names[key[1]],
                           chr_i=int(chrom[key[0]]), chr_j=int(chrom[key[1]]),
                           pos_i=int(pos[key[0]]), pos_j=int(pos[key[1]]),
                           seed=seed, n_folds_nominated=nom[key],
                           score=best[key][0], tag=best[key][3],
                           acc_agg=accA, acc_with_pair=Q.pcc(ya, g1),
                           d_vs_AGG=Q.pcc(ya, g1) - accA,
                           is_trans=bool(chrom[key[0]] != chrom[key[1]]))
                pd.DataFrame([row]).to_csv(
                    res_path, mode="a", index=False,
                    header=not os.path.exists(res_path))
                if ci % 25 == 0 or ci == len(todo):
                    say(f"      {ci}/{len(todo)} cells done "
                        f"({(time.time()-t0)/60:.0f} min elapsed)")

    if not os.path.exists(res_path):
        say("\nnothing computed")
        return
    R = pd.read_csv(res_path)

    say("\n" + "=" * 74)
    say("PER-FOLD DISCRIMINATION -- fraction of tested pairs that improved")
    say("=" * 74)
    for (trait, typ), g in R.groupby(["trait", "type"]):
        say(f"\n  {trait} / {typ}:")
        for seed, gg in g.groupby("seed"):
            frac = float((gg.d_vs_AGG > 0).mean())
            flag = ("   <- passes almost everything; contributes no "
                    "discrimination" if frac > 0.95 else "")
            say(f"      fold {seed}: {frac:.2f} of {len(gg)} pairs improved"
                f"{flag}")

    say("\n" + "=" * 74)
    say("RELIABLE PAIRS -- improved in EVERY fold tested")
    say("=" * 74)
    out = []
    for (trait, typ, si, sj), g in R.groupby(["trait", "type",
                                              "snp_i", "snp_j"]):
        nf = len(g)
        npos = int((g.d_vs_AGG > 0).sum())
        out.append(dict(trait=trait, type=typ, snp_i=si, snp_j=sj,
                        chr_i=int(g.chr_i.iloc[0]), chr_j=int(g.chr_j.iloc[0]),
                        n_folds_tested=nf, n_folds_improved=npos,
                        all_folds=bool(npos == nf and nf == a.folds),
                        mean_d=float(g.d_vs_AGG.mean()),
                        min_d=float(g.d_vs_AGG.min()),
                        max_d=float(g.d_vs_AGG.max()),
                        n_folds_nominated=int(g.n_folds_nominated.iloc[0]),
                        is_trans=bool(g.is_trans.iloc[0])))
    S = pd.DataFrame(out).sort_values(
        ["all_folds", "n_folds_improved", "mean_d"],
        ascending=[False, False, False])
    S.to_csv(os.path.join(a.out, "pair_summary.csv"), index=False)
    rel = S[S.all_folds]
    say(f"\n  {len(rel)} of {len(S)} pairs improved in all {a.folds} folds")
    if len(rel):
        say(rel.head(40).round(5).to_string(index=False))
    say("\n  Read min_d as the worst fold: a reliable pair should have min_d")
    say("  above zero by a margin, not by 1e-5. Cross-check against the")
    say("  per-fold discrimination table above -- if one fold passed ~all")
    say("  pairs, 'all folds' is effectively 'all folds but that one'.")
    say(f"\n-> {a.out}/pair_summary.csv, per_pair_per_fold.csv")


if __name__ == "__main__":
    main()
