#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 allocate_with_validated_pairs.py -- allocate using the VALIDATED epistatic
                                     pairs, and see which mating decisions
                                     change
================================================================================

 focal_pair_perfold.py first keeps the epistatic pairs that improve prediction
 in all five folds, including the most demanding one (fold 2428, which only 23%
 of pairs survive). For pig BF two pairs remain:
     2:102262831 x 8:49048406   (positive in 5/5 folds, min_d +0.0052)
     4:126553773 x 7:16464284   (positive in 5/5 folds, min_d +0.0010)

 This script feeds those specific, fold-validated pairs into allocation as the
 epistatic term, rather than the genome-wide aggregate kernel, and ranks the
 same candidates under three arms:
     ADD        additive breeding value only
     ADD+DOM    plus the expected dominance E[Hd] of each cross
     FULL       plus the expected effect of the validated pairs in that cross

 The key output is overlap_with_ADD: how many matings in the FULL plan differ
 from the purely additive plan. That is where "which breeding decisions did
 epistasis actually change" is answered. A gain in predictive accuracy says
 whether epistasis is usable; the overlap says who was actually mated once it
 was used.

 Difference from the aggregate-kernel mate_allocation: the marginal epistatic
 effects come from marker effects fitted with a kernel containing ONLY these
 pairs, not from the MULT_t kernel aggregated over 6.4e8 pairs. gv_epi
 therefore reflects exactly these pairs and is interpretable and reproducible.

     python allocate_with_validated_pairs.py --trait bf \
         --pairs "2:102262831,8:49048406" "4:126553773,7:16464284" \
         --type MULT --out allocate_bf_validated
================================================================================
"""
from __future__ import annotations

import argparse
import os
import sys
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve

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
    ap.add_argument("--trait", required=True)
    ap.add_argument("--pheno")
    ap.add_argument("--pairs", nargs="+", required=True,
                    help='validated pairs, each written as "chr:pos,chr:pos"')
    ap.add_argument("--type", default="MULT",
                    help="the interaction type of those pairs (the type column of the per-fold results)")
    ap.add_argument("--n_candidates", type=int, default=100)
    ap.add_argument("--n_crosses", type=int, default=50)
    ap.add_argument("--max_use", type=int, default=4)
    ap.add_argument("--f_max", type=float, default=0.10)
    ap.add_argument("--cache_dir", default="adept_kernel_cache",
                    help="where exact kernels are cached; they depend "
                         "only on the genotypes, so use one directory "
                         "per dataset")
    ap.add_argument("--out", default="allocate_validated")
    ap.add_argument("--seed", type=int, default=2024)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    # ---- load ----
    pc = P.CFG()
    pc.BFILE, pc.N_PCS = a.bfile, 0
    pc.PHENO_FILE = a.pheno or f"{a.trait}.phe"
    pc.TRAIT = a.trait
    pc.COV_FILE = pc.QCOV_FILE = None
    X, y, chrom, pos, snp_names, _ = P.load_real(pc)
    n, m = X.shape
    say(f"[data] {n} animals x {m} SNPs   trait={a.trait}")
    Ha, Hd = P.noia(X)
    Xx = (X > 0).astype(np.float64) * 2 - 1
    xc = Xx - Xx.mean(0); sd = xc.std(0); sd[sd < 1e-12] = 1.0
    Xc = xc / sd
    del Xx, xc

    name_to_idx = {s: k for k, s in enumerate(snp_names)}
    sel = []
    for spec in a.pairs:
        s1, s2 = spec.split(",")
        s1, s2 = s1.strip(), s2.strip()
        if s1 not in name_to_idx or s2 not in name_to_idx:
            raise SystemExit(f"SNP not in the marker set: {spec} "
                             f"({'missing '+s1 if s1 not in name_to_idx else ''} "
                             f"{'missing '+s2 if s2 not in name_to_idx else ''})")
        sel.append((1.0, name_to_idx[s1], name_to_idx[s2], a.type))
    say(f"[pairs] using {len(sel)} validated {a.type} pair(s):")
    for _, i, j, _ in sel:
        say(f"    {snp_names[i]} x {snp_names[j]}  "
            f"(chr{chrom[i]} x chr{chrom[j]})")

    # ---- kernels: A, D, and TOP = validated-pairs kernel ----
    qc = Q.CFG(); qc.CACHE_DIR = a.cache_dir; qc.DO_CIS = False
    K, meta = Q.build_kernels(qc, X, Ha, Hd, chrom, keep_xc=False)
    KT = Q.pairs_kernel(sel, a.type, X, Ha, Hd, Xc)
    if KT is None:
        raise SystemExit("no kernel could be built from the validated pairs (constant features?)")
    K = {k: K[k] for k in ("A", "D")}
    K["TOP"] = KT
    Xfix = np.ones((n, 1))

    # ---- fit full model, get variance components + marker effects ----
    keys = ["A", "D", "TOP"]
    mm, alpha, var = Q.offspring_components(a, X, Ha, Hd, chrom, K, meta,
                                            y, Xfix, keys)
    tot = sum(v for k, v in var.items() if k != "_resid") + var["_resid"]
    say("    variance components: " +
        "  ".join(f"{k}={var[k]/tot:.4f}" for k in keys) +
        f"  resid={var['_resid']/tot:.4f}")
    if var.get("TOP", 0) / tot < 1e-4:
        say("    [note] the variance component of the validated pairs was estimated at")
        say("           about zero over the whole population, so the epistatic term in")
        say("           FULL is near zero and FULL nearly coincides with ADD+DOM. This")
        say("           is not an error but REML's verdict on the whole sample, and it")
        say("           does not contradict an improvement on the held-out folds: one is\n"
            "           a whole-sample variance fraction, the other a marginal gain in\n"
            "           out-of-fold prediction.")

    # ---- back-solved additive & dominance marker effects ----
    mu = X.mean(0)
    a_hat = var["A"] * (Ha.T @ alpha) / (meta["sA"] * (
        2 * np.sum((mu / 2) * (1 - mu / 2)) + 1e-12))
    d_hat = (var["D"] * (Hd.T @ alpha) / (meta["sD"] * (
        np.sum(np.var(Hd, 0)) + 1e-12)) if "D" in var else np.zeros(m))

    # ---- epistatic marker-pair effects from the validated-pairs kernel ----
    # each column of Z is one pair's standardised product feature; the
    # corresponding effect is var[TOP] * Z^T alpha, normalised by the number of
    # columns and that column's variance
    Zcols, pair_idx = [], []
    for _, i, j, tg in sel:
        z = Q.pair_feature(a.type, tg, X, Ha, Hd, Xc, i, j)
        if z is not None:
            Zcols.append(z)
            pair_idx.append((i, j))
    Z = np.column_stack(Zcols)                              # n x npair(std)
    top_eff = var["TOP"] * (Z.T @ alpha) / (Z.shape[1] + 1e-12)
    say(f"    marginal effects of the validated pairs: " +
        ", ".join(f"{snp_names[i]}x{snp_names[j]}={e:+.4f}"
                  for (i, j), e in zip(pair_idx, top_eff)))

    # ---- candidates ----
    rng = np.random.default_rng(a.seed)
    cand = np.sort(rng.choice(n, min(a.n_candidates, n), replace=False))
    nc = len(cand)
    say(f"    candidates drawn at random: {nc}")

    # frequency basis for the expected dominance, as in mate_allocation
    p_aa = np.mean(X == 1, axis=0)
    var_A = np.var(Ha, axis=0)
    beta_d = np.where(var_A > 1e-12, (p_aa * (1 - mu)) / np.maximum(var_A, 1e-12), 0.0)

    pairs = list(combinations(range(nc), 2))
    npair = len(pairs)
    ii = np.array([p[0] for p in pairs]); jj = np.array([p[1] for p in pairs])
    say(f"    evaluating {npair:,} candidate crosses")

    add = np.empty(npair); dom = np.empty(npair); epi = np.zeros(npair)
    inb = np.empty(npair); het = np.empty(npair)

    # pre-fetch the SNP columns of the validated pairs to build the expected features
    ai = np.array([p[0] for p in pair_idx])
    aj = np.array([p[1] for p in pair_idx])
    # standardisation constants per pair, taken over the whole population as in pair_feature
    feat_mean = np.array([ (Ha[:, i]*Ha[:, j]).mean() for i, j in pair_idx ])
    feat_std  = np.array([ (Ha[:, i]*Ha[:, j]).std()  for i, j in pair_idx ])
    feat_std[feat_std < 1e-10] = 1.0

    CH = 2000
    for s in range(0, npair, CH):
        e = min(s + CH, npair)
        a_ = X[cand][ii[s:e]] / 2.0
        b_ = X[cand][jj[s:e]] / 2.0
        e_dos = a_ + b_
        e_ha_full = e_dos - mu[None, :]
        p_het = a_ * (1 - b_) + (1 - a_) * b_
        e_hd = p_het - p_aa[None, :] - beta_d[None, :] * e_ha_full
        add[s:e] = e_ha_full @ a_hat
        dom[s:e] = e_hd @ d_hat
        het[s:e] = p_het.mean(1)
        # epistasis: expected offspring product feature -> standardise -> times the marginal effect
        if a.type == "MULT":
            e_ha_i = e_ha_full[:, ai]                       # chunk x npair
            e_ha_j = e_ha_full[:, aj]
            feat = e_ha_i * e_ha_j
            feat = (feat - feat_mean[None, :]) / feat_std[None, :]
            epi[s:e] = feat @ top_eff
        inb[s:e] = meta["KA_raw"][cand[ii[s:e]], cand[jj[s:e]]] / 2.0

    df = pd.DataFrame(dict(
        sire=cand[ii], dam=cand[jj],
        gv_add=add, gv_dom=dom, gv_epi=epi,
        gv_add_dom=add + dom, gv_full=add + dom + epi,
        progeny_F=inb, exp_heterozygosity=het))
    df.to_csv(os.path.join(a.out, f"{a.trait}_all_crosses.csv"), index=False)

    # ---- three plans under the same constraints ----
    plans = {}
    for strat, col in [("ADD", "gv_add"), ("ADD+DOM", "gv_add_dom"),
                       ("FULL", "gv_full")]:
        order = np.argsort(-df[col].values)
        use, chosen = {}, []
        for k in order:
            if df.progeny_F.values[k] > a.f_max:
                continue
            s_, d_ = df.sire.values[k], df.dam.values[k]
            if use.get(s_, 0) >= a.max_use or use.get(d_, 0) >= a.max_use:
                continue
            use[s_] = use.get(s_, 0) + 1
            use[d_] = use.get(d_, 0) + 1
            chosen.append(k)
            if len(chosen) >= a.n_crosses:
                break
        plans[strat] = df.iloc[chosen].copy()
        plans[strat].insert(0, "strategy", strat)
    pd.concat(plans.values(), ignore_index=True).to_csv(
        os.path.join(a.out, f"{a.trait}_mating_plan.csv"), index=False)

    base = set(zip(plans["ADD"].sire, plans["ADD"].dam))
    rows = []
    for strat, p in plans.items():
        s = set(zip(p.sire, p.dam))
        rows.append(dict(trait=a.trait, strategy=strat, n_crosses=len(p),
                         mean_gv_full=float(p.gv_full.mean()),
                         mean_gv_add=float(p.gv_add.mean()),
                         mean_gv_dom=float(p.gv_dom.mean()),
                         mean_gv_epi=float(p.gv_epi.mean()),
                         mean_F=float(p.progeny_F.mean()),
                         n_parents=len(set(p.sire) | set(p.dam)),
                         overlap_with_ADD=len(s & base) / max(len(base), 1),
                         crosses_changed=len(base) - len(s & base)))
    summ = pd.DataFrame(rows)
    summ.to_csv(os.path.join(a.out, f"{a.trait}_plan_summary.csv"), index=False)
    say("\n" + "=" * 70)
    say(summ.round(4).to_string(index=False))
    say("=" * 70)
    full = summ[summ.strategy == "FULL"].iloc[0]
    say(f"\n  the validated epistatic pairs changed "
        f"{int(full.crosses_changed)}/{len(base)} mating(s) in FULL relative to the "
        f"(overlap {full.overlap_with_ADD:.2f})。")
    say("  overlap < 1 means epistasis changed who is mated to whom, which is the")
    say("  evidence for trait-wise allocation; overlap = 1 means the pairs improved\n"
        "  prediction without changing the optimal matings, which is an honest result too.")
    say(f"\n-> {a.out}/{a.trait}_plan_summary.csv, {a.trait}_all_crosses.csv, "
        f"{a.trait}_mating_plan.csv")


if __name__ == "__main__":
    main()
