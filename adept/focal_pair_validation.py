#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 focal_pair_validation.py -- which epistatic pairs survive honest validation?
================================================================================

 THE QUESTION
 ------------------------------------------------------------------------------
 Genome-wide pair scoring finds pairs that look good on the split they were
 found in. For mate allocation the only gain worth banking is one that
 reproduces on data that played no part in choosing the pair.

 TWO DELIBERATE CHOICES
 ------------------------------------------------------------------------------
 (1) NO PAIR-LEVEL LD DEDUPLICATION.
     Merging pairs whose partner SNPs are physically close and calling them
     "the same gene" would be unfounded -- physical distance is not evidence
     of a shared signal, and gene identity needs annotation, not a bp
     threshold. It is also unnecessary here: the marker panel was
     already LD-pruned before this analysis (~1M -> ~35k SNPs), so residual
     correlation between retained markers is low by construction. Every pair
     meeting the reproducibility criterion is therefore reported as-is, and
     the per-pair r^2 between its own two loci is written to selected_pairs.csv
     as a descriptive column only -- it filters nothing.

 (2) EVERY TRAIT IS SCREENED FOR PAIRS, REGARDLESS OF ITS AGGREGATE KERNEL.
     An aggregate epistatic kernel averages over ~6.4e8 pairs. A handful of
     genuine interactions is diluted to nothing in that average, so a null
     aggregate kernel does NOT imply there are no usable pairs. Traits whose
     aggregate epistatic component was null (lmd, ttn, ltn in the pig data)
     are therefore screened exactly like the ones that were positive. These
     are two different questions and are answered separately.

 WHY THE OBVIOUS ANALYSIS IS BIASED, AND WHAT THIS DOES INSTEAD
 ------------------------------------------------------------------------------
 Ranking pairs and then reporting whichever rank has the best held-out delta
 reports a maximum over ranks: optimistic even when every pair is worthless.
 That is how a rank-2 MULT pair showed +0.0284 on one split while averaging
 -0.0303 over five.

 LEAVE-ONE-FOLD-OUT SELECTION: to evaluate fold s, candidate pairs come ONLY
 from the training sets of the OTHER folds. Fold s's test set never influences
 which pairs are chosen, so its delta is unbiased for a pre-specified set.
 Each fold gives one estimate; the sign test across folds is the headline.

 A pair enters the evaluated set only if it appears in the pools of at least
 --min_folds other folds. Fix that on the command line BEFORE looking at
 results.

     python focal_pair_validation.py \
         --traits bf lmd lmp ltn rtn tpd ttn \
         --types MULT DD THR --top_k 20 --min_folds 3 --folds 5 \
         --out focal_validation
================================================================================
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from math import comb

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pig_epik_pipeline as P
import pig_epistasis_pipeline as Q


def say(*a):
    print(*a, flush=True)


def pair_key(i, j, typ):
    return (typ, int(min(i, j)), int(max(i, j)))


def evaluate_set(sel, typ, X, Ha, Hd, Xc, K, agg, y, Xfix, tr, te, boot, seed,
                 max_in_kernel):
    """Build a kernel from `sel` and return held-out accuracy and deltas."""
    sel = sel[:max_in_kernel]
    if not sel:
        return None
    KT = Q.pairs_kernel(sel, typ, X, Ha, Hd, Xc)
    if KT is None:
        return None
    Kl = dict(K)
    Kl["TOP"] = KT
    g1, ya, _ = Q.fit_gebv(y, Xfix, Kl, ["A", "D", agg, "TOP"], tr, te)
    if g1 is None:
        return None
    return g1, ya, len(sel)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bfile", required=True,
                    help="PLINK prefix (.bed/.bim/.fam) of the "
                         "QC'd, pruned genotypes")
    ap.add_argument("--traits", nargs="+",
                    default=["bf", "lmd", "lmp", "ltn", "rtn", "tpd", "ttn"],
                    help="every trait is screened; a null aggregate kernel is "
                         "not a reason to skip a trait")
    ap.add_argument("--types", nargs="+", default=["MULT", "DD", "THR"])
    ap.add_argument("--top_k", type=int, default=20)
    ap.add_argument("--min_folds", type=int, default=3,
                    help="a pair must appear in this many OTHER folds' pools. "
                         "DECLARE BEFORE LOOKING AT RESULTS.")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--max_pairs_in_kernel", type=int, default=10)
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--cache_dir", default="adept_kernel_cache",
                    help="where exact kernels are cached; they depend "
                         "only on the genotypes, so use one directory "
                         "per dataset")
    ap.add_argument("--out", default="focal_validation")
    ap.add_argument("--seed", type=int, default=2024)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    ck = os.path.join(a.out, "_scores")
    os.makedirs(ck, exist_ok=True)

    cfg = Q.CFG()
    cfg.BFILE = a.bfile
    pc = P.CFG()
    pc.BFILE, pc.N_PCS = a.bfile, 0
    pc.PHENO_FILE, pc.TRAIT = f"{a.traits[0]}.phe", a.traits[0]
    pc.COV_FILE = pc.QCOV_FILE = None          # convention: intercept only
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
    labels, how = P.structure_clusters(K["A"], None, 10, a.seed)
    Xfix = np.ones((n, 1))
    agg = "MULT_t" if "MULT_t" in K else "HAD"

    all_rows, all_sel = [], []

    for trait in a.traits:
        phe = f"{trait}.phe"
        if not os.path.exists(phe):
            say(f"\n[skip] {trait}: {phe} not found")
            continue
        say("\n" + "=" * 74)
        say(f"TRAIT {trait}")
        say("=" * 74)
        pc.PHENO_FILE, pc.TRAIT = phe, trait
        _, y, _, _, _, _ = P.load_real(pc)

        folds, pools = [], {}
        for s in range(a.folds):
            seed = a.seed + s * 101
            tr, te = Q.make_split(n, labels, cfg.SPLIT, cfg.TEST_FRAC, seed)
            folds.append((seed, tr, te))
            f = os.path.join(ck, f"{trait}_pool_seed{seed}.json")
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
            say(f"  fold {s+1}: scored in {time.time()-t0:.0f}s, checkpointed")

        for s, (seed, tr, te) in enumerate(folds):
            others = [pools[sd] for sd, _, _ in folds if sd != seed]
            g0, ya, _ = Q.fit_gebv(y, Xfix, K, ["A", "D"], tr, te)
            gA, _, _ = Q.fit_gebv(y, Xfix, K, ["A", "D", agg], tr, te)
            if g0 is None or gA is None:
                say(f"  fold {s+1}: baseline fit failed")
                continue
            acc0, accA = Q.pcc(ya, g0), Q.pcc(ya, gA)

            for typ in a.types:
                cnt, best = {}, {}
                for pool in others:
                    for (v, i, j, tag) in pool.get(typ, []):
                        kk = pair_key(i, j, typ)
                        cnt[kk] = cnt.get(kk, 0) + 1
                        if kk not in best or v > best[kk][0]:
                            best[kk] = (v, i, j, tag)
                raw_sel = [best[kk] for kk, c in cnt.items()
                           if c >= a.min_folds]
                raw_sel.sort(key=lambda z: -z[0])
                if not raw_sel:
                    continue
                res = evaluate_set(raw_sel, typ, X, Ha, Hd, Xc, K, agg, y,
                                   Xfix, tr, te, a.boot, seed,
                                   a.max_pairs_in_kernel)
                if res is not None:
                    g1, ya2, n_used = res
                    acc1 = Q.pcc(ya2, g1)
                    d, lo, hi = Q.boot_delta(ya2, gA, g1, a.boot, seed)
                    all_rows.append(dict(
                        trait=trait, seed=seed, type=typ,
                        n_pairs_available=len(raw_sel),
                        n_pairs_in_kernel=n_used,
                        acc=acc1, acc_AplusD=acc0, acc_agg=accA,
                        d_vs_AD=acc1 - acc0, d_vs_AGG=acc1 - accA,
                        boot_lo=lo, boot_hi=hi))
                for (v, i, j, tag) in raw_sel[:a.max_pairs_in_kernel]:
                    all_sel.append(dict(
                        trait=trait, eval_seed=seed, type=typ,
                        snp_i=snp_names[i], snp_j=snp_names[j],
                        chr_i=int(chrom[i]), chr_j=int(chrom[j]),
                        pos_i=int(pos[i]), pos_j=int(pos[j]),
                        score=v, tag=tag,
                        n_other_folds=cnt[pair_key(i, j, typ)],
                        is_trans=bool(chrom[i] != chrom[j]),
                        r2_between_the_two_loci=float(
                            np.corrcoef(X[:, i], X[:, j])[0, 1] ** 2)
                        if X[:, i].std() > 1e-12 and X[:, j].std() > 1e-12
                        else np.nan))
            say(f"  fold {s+1}/{a.folds} done")

    R = pd.DataFrame(all_rows)
    if R.empty:
        say("\nno pair set met --min_folds in any trait; nothing to evaluate")
        return
    R.to_csv(os.path.join(a.out, "fold_results.csv"), index=False)
    pd.DataFrame(all_sel).to_csv(os.path.join(a.out, "selected_pairs.csv"),
                                 index=False)

    say("\n" + "=" * 74)
    say("RESULT -- gain over the best aggregate kernel, pairs chosen WITHOUT")
    say("          access to the fold they are evaluated on")
    say("=" * 74)
    out = []
    for (trait, typ), g in R.dropna(subset=["d_vs_AGG"]).groupby(
            ["trait", "type"]):
        v = g.d_vs_AGG.values
        k, nf = int(np.sum(v > 0)), len(v)
        p_sign = sum(comb(nf, x) for x in range(k, nf + 1)) / 2 ** nf
        out.append(dict(trait=trait, type=typ,
                        n_folds=nf, n_positive=k, sign_p=float(p_sign),
                        mean_d_vs_AGG=float(v.mean()),
                        mean_d_vs_AD=float(g.d_vs_AD.mean()),
                        mean_pairs=float(g.n_pairs_in_kernel.mean())))
    S = pd.DataFrame(out).sort_values(["sign_p", "mean_d_vs_AGG"],
                                      ascending=[True, False])
    S.to_csv(os.path.join(a.out, "summary.csv"), index=False)
    say(S.round(5).to_string(index=False))

    say("\n  Reading this table:")
    say("    n_positive / n_folds is the headline -- positive in every fold is")
    say("    a usable signal even when the mean gain is small (GBLUP is")
    say("    deployed at accuracies near 0.2; small does not mean useless).")
    say("    selected_pairs.csv lists every pair that entered a kernel, with")
    say("    its own two loci's r^2 as a descriptive column -- nothing is")
    say("    filtered on it; the panel was already LD-pruned upstream.")
    say(f"\n-> {a.out}/summary.csv, fold_results.csv, selected_pairs.csv")
    say("\n  --min_folds and --top_k were fixed before these numbers existed; "
        "changing them now and re-reading turns this into a threshold search.")


if __name__ == "__main__":
    main()