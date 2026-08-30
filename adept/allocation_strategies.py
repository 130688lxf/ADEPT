#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 allocation_strategies.py -- maximise phenotypic gain without giving up
                             additive merit
================================================================================

 The problem: ranking crosses by the plain sum gv_full = gv_add + gv_dom +
 gv_epi trades certain additive merit for unstable epistasis. On the pig BF
 trait the mean_gv_add of the FULL arm fell from 1.30 to 0.72. Epistatic
 effects are background-dependent and stable only in some crosses, so that
 trade is dangerous.

 Three strategies that keep additive merit while still taking non-additive
 gain are provided as fixed, parallel comparison arms. The default is
 two_stage, which makes the breeder's actual logic explicit: additive merit is
 the entry ticket and non-additive merit is a bonus among otherwise equal
 candidates, so the ranking can be audited step by step.

   1. two_stage   (default) in two stages:
        stage 1  keep only crosses whose gv_add reaches the --add_quantile
                 quantile (0.70 by default, i.e. the top 30% on additive merit)
        stage 2  within that eligible set, rank by gv_dom + gv_epi
        => every selected cross is additively sound by construction, and among
           additively comparable crosses the best non-additive one is taken.
           For a trait with no epistasis, stage 2 degenerates to purely
           additive selection on its own.

   2. weighted    weighted by reliability:
        score = gv_add + w_dom*gv_dom + w_epi*gv_epi
        The weights reflect how reproducible each component is. The epistatic
        weight is its variance fraction (about 0.056 for pig BF), so epistasis
        can only fine-tune and can never override additive merit. This is
        shrinkage of the uncertain components, on the same principle by which
        GBLUP shrinks large effects. Override with --w_dom / --w_epi; the
        default is the variance fraction.

   3. add_floor   a hard floor:
        rank by gv_full, but no selected cross may have a gv_add below the
        --floor_quantile quantile (0.80 by default) of the set selected by the
        pure ADD arm. The simplest and most transparent of the three.

 All three are written out alongside the ADD / ADD+DOM / FULL(naive) baselines,
 and every arm reports mean_gv_add, so it is immediately visible which one kept
 additive merit and which one gained most phenotype.

     python allocation_strategies.py --trait bf \
         --pairs "2:102262831,8:49048406" "4:126553773,7:16464284" \
         --type MULT --strategies two_stage weighted add_floor \
         --out allocation_strategies_bf
================================================================================
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from itertools import combinations

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pig_epik_pipeline as P
import pig_epistasis_pipeline as Q


def say(*a):
    print(*a, flush=True)


def greedy_pick(df, score, n_crosses, max_use, f_max):
    """Greedy selection by descending score under the inbreeding and
    usage constraints. Returns the list of selected row indices."""
    order = np.argsort(-score)
    use, chosen = {}, []
    F = df.progeny_F.values
    S = df.sire.values; D = df.dam.values
    for k in order:
        if F[k] > f_max:
            continue
        if use.get(S[k], 0) >= max_use or use.get(D[k], 0) >= max_use:
            continue
        use[S[k]] = use.get(S[k], 0) + 1
        use[D[k]] = use.get(D[k], 0) + 1
        chosen.append(k)
        if len(chosen) >= n_crosses:
            break
    return chosen


def plan_from_indices(df, idx, name):
    p = df.iloc[idx].copy()
    p.insert(0, "strategy", name)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bfile", required=True,
                    help="PLINK prefix (.bed/.bim/.fam) of the "
                         "QC'd, pruned genotypes")
    ap.add_argument("--trait", required=True)
    ap.add_argument("--pheno")
    ap.add_argument("--pairs", nargs="+", required=True)
    ap.add_argument("--type", default="MULT")
    ap.add_argument("--strategies", nargs="+",
                    default=["two_stage", "weighted", "add_floor"],
                    choices=["two_stage", "weighted", "add_floor"])
    ap.add_argument("--add_quantile", type=float, default=0.70,
                    help="two_stage stage 1: additive quantile required to be eligible")
    ap.add_argument("--floor_quantile", type=float, default=0.80,
                    help="add_floor: selected crosses must not fall below this additive quantile of the ADD arm")
    ap.add_argument("--w_dom", type=float, default=-1,
                    help="weighted: dominance weight; a negative value uses the variance fraction")
    ap.add_argument("--w_epi", type=float, default=-1,
                    help="weighted: epistatic weight; a negative value uses the variance fraction")
    ap.add_argument("--n_candidates", type=int, default=100)
    ap.add_argument("--n_crosses", type=int, default=50)
    ap.add_argument("--max_use", type=int, default=4)
    ap.add_argument("--f_max", type=float, default=0.10)
    ap.add_argument("--cache_dir", default="adept_kernel_cache",
                    help="where exact kernels are cached; they depend "
                         "only on the genotypes, so use one directory "
                         "per dataset")
    ap.add_argument("--out", default="allocation_strategies")
    ap.add_argument("--seed", type=int, default=2024)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    # ---- load genotype + pheno ----
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
    Xc = xc / sd; del Xx, xc

    name_to_idx = {s: k for k, s in enumerate(snp_names)}
    sel = []
    for spec in a.pairs:
        s1, s2 = [x.strip() for x in spec.split(",")]
        if s1 not in name_to_idx or s2 not in name_to_idx:
            raise SystemExit(f"SNP not in the marker set: {spec}")
        sel.append((1.0, name_to_idx[s1], name_to_idx[s2], a.type))
    say(f"[pairs] {len(sel)} validated {a.type} pair(s)")

    qc = Q.CFG(); qc.CACHE_DIR = a.cache_dir; qc.DO_CIS = False
    K, meta = Q.build_kernels(qc, X, Ha, Hd, chrom, keep_xc=False)
    KT = Q.pairs_kernel(sel, a.type, X, Ha, Hd, Xc)
    if KT is None:
        raise SystemExit("could not build the pair kernel")
    K = {k: K[k] for k in ("A", "D")}; K["TOP"] = KT
    Xfix = np.ones((n, 1))
    keys = ["A", "D", "TOP"]
    mm, alpha, var = Q.offspring_components(a, X, Ha, Hd, chrom, K, meta,
                                            y, Xfix, keys)
    tot = sum(v for k, v in var.items() if k != "_resid") + var["_resid"]
    var_frac = {k: var[k] / tot for k in keys}
    say("    variance fractions: " + "  ".join(f"{k}={var_frac[k]:.4f}" for k in keys))
    json.dump(var_frac, open(os.path.join(a.out, "variance_fraction.json"), "w"),
              indent=2)

    mu = X.mean(0)
    a_hat = var["A"] * (Ha.T @ alpha) / (meta["sA"] * (
        2 * np.sum((mu / 2) * (1 - mu / 2)) + 1e-12))
    d_hat = (var["D"] * (Hd.T @ alpha) / (meta["sD"] * (
        np.sum(np.var(Hd, 0)) + 1e-12)) if "D" in var else np.zeros(m))
    pair_idx = [(i, j) for _, i, j, _ in sel]
    feat_mean = np.array([(Ha[:, i] * Ha[:, j]).mean() for i, j in pair_idx])
    feat_std = np.array([(Ha[:, i] * Ha[:, j]).std() for i, j in pair_idx])
    feat_std[feat_std < 1e-10] = 1.0
    Z = np.column_stack([Q.pair_feature(a.type, tg, X, Ha, Hd, Xc, i, j)
                         for (_, i, j, tg) in sel])
    top_eff = var["TOP"] * (Z.T @ alpha) / (Z.shape[1] + 1e-12)

    rng = np.random.default_rng(a.seed)
    cand = np.sort(rng.choice(n, min(a.n_candidates, n), replace=False))
    nc = len(cand)
    p_aa = np.mean(X == 1, axis=0)
    var_A = np.var(Ha, axis=0)
    beta_d = np.where(var_A > 1e-12, (p_aa * (1 - mu)) / np.maximum(var_A, 1e-12), 0.0)

    pairs = list(combinations(range(nc), 2))
    npair = len(pairs)
    ii = np.array([p[0] for p in pairs]); jj = np.array([p[1] for p in pairs])
    ai = np.array([p[0] for p in pair_idx]); aj = np.array([p[1] for p in pair_idx])
    add = np.empty(npair); dom = np.empty(npair); epi = np.zeros(npair)
    inb = np.empty(npair); het = np.empty(npair)
    CH = 2000
    for s in range(0, npair, CH):
        e = min(s + CH, npair)
        a_ = X[cand][ii[s:e]] / 2.0; b_ = X[cand][jj[s:e]] / 2.0
        e_ha = (a_ + b_) - mu[None, :]
        p_het = a_ * (1 - b_) + (1 - a_) * b_
        e_hd = p_het - p_aa[None, :] - beta_d[None, :] * e_ha
        add[s:e] = e_ha @ a_hat
        dom[s:e] = e_hd @ d_hat
        het[s:e] = p_het.mean(1)
        if a.type == "MULT":
            feat = e_ha[:, ai] * e_ha[:, aj]
            feat = (feat - feat_mean[None, :]) / feat_std[None, :]
            epi[s:e] = feat @ top_eff
        inb[s:e] = meta["KA_raw"][cand[ii[s:e]], cand[jj[s:e]]] / 2.0

    df = pd.DataFrame(dict(
        sire=cand[ii], dam=cand[jj], gv_add=add, gv_dom=dom, gv_epi=epi,
        gv_nonadd=dom + epi, gv_add_dom=add + dom, gv_full=add + dom + epi,
        progeny_F=inb, exp_heterozygosity=het))
    df.to_csv(os.path.join(a.out, f"{a.trait}_all_crosses.csv"), index=False)

    plans = {}
    # ---- baselines ----
    plans["ADD"] = plan_from_indices(
        df, greedy_pick(df, df.gv_add.values, a.n_crosses, a.max_use, a.f_max),
        "ADD")
    plans["ADD+DOM"] = plan_from_indices(
        df, greedy_pick(df, df.gv_add_dom.values, a.n_crosses, a.max_use, a.f_max),
        "ADD+DOM")
    plans["FULL_naive"] = plan_from_indices(
        df, greedy_pick(df, df.gv_full.values, a.n_crosses, a.max_use, a.f_max),
        "FULL_naive")

    # ---- systematic strategies ----
    if "two_stage" in a.strategies:
        thr = np.quantile(df.gv_add.values, a.add_quantile)
        elig = df[df.gv_add >= thr].copy()
        say(f"    [two_stage] additive {a.add_quantile:.0%} quantile threshold={thr:.4f}, "
            f"{len(elig)}/{npair} eligible")
        # rank the eligible subset by non-additive merit; the greedy constraints
        # are applied on the indices of that subset
        idx_sub = greedy_pick(elig.reset_index(drop=True),
                              elig.gv_nonadd.values, a.n_crosses,
                              a.max_use, a.f_max)
        chosen_global = elig.index.values[idx_sub]
        plans["two_stage"] = plan_from_indices(
            df, [df.index.get_loc(g) for g in chosen_global], "two_stage")

    if "weighted" in a.strategies:
        w_d = a.w_dom if a.w_dom >= 0 else var_frac["D"] / max(var_frac["A"], 1e-9)
        w_e = a.w_epi if a.w_epi >= 0 else var_frac["TOP"] / max(var_frac["A"], 1e-9)
        say(f"    [weighted] w_dom={w_d:.3f}  w_epi={w_e:.3f} "
            f"(negative values use the variance fraction relative to the additive share)")
        score = df.gv_add.values + w_d * df.gv_dom.values + w_e * df.gv_epi.values
        plans["weighted"] = plan_from_indices(
            df, greedy_pick(df, score, a.n_crosses, a.max_use, a.f_max),
            "weighted")

    if "add_floor" in a.strategies:
        add_in_ADD = plans["ADD"].gv_add.values
        floor = np.quantile(add_in_ADD, a.floor_quantile)
        say(f"    [add_floor] additive floor={floor:.4f} "
            f"(the {a.floor_quantile:.0%} quantile of the pure ADD arm)")
        mask = df.gv_add.values >= floor
        sub = df[mask].copy()
        idx_sub = greedy_pick(sub.reset_index(drop=True), sub.gv_full.values,
                              a.n_crosses, a.max_use, a.f_max)
        chosen_global = sub.index.values[idx_sub]
        plans["add_floor"] = plan_from_indices(
            df, [df.index.get_loc(g) for g in chosen_global], "add_floor")

    pd.concat(plans.values(), ignore_index=True).to_csv(
        os.path.join(a.out, f"{a.trait}_mating_plan.csv"), index=False)

    base = set(zip(plans["ADD"].sire, plans["ADD"].dam))
    add_ADD = plans["ADD"].gv_add.mean()
    rows = []
    for name, p in plans.items():
        s = set(zip(p.sire, p.dam))
        rows.append(dict(
            trait=a.trait, strategy=name, n_crosses=len(p),
            mean_gv_add=float(p.gv_add.mean()),
            mean_gv_dom=float(p.gv_dom.mean()),
            mean_gv_epi=float(p.gv_epi.mean()),
            mean_gv_full=float(p.gv_full.mean()),
            add_retained_vs_ADD=float(p.gv_add.mean() / add_ADD),
            mean_F=float(p.progeny_F.mean()),
            overlap_with_ADD=len(s & base) / max(len(base), 1),
            crosses_changed=len(base) - len(s & base)))
    summ = pd.DataFrame(rows)
    summ.to_csv(os.path.join(a.out, f"{a.trait}_strategy_summary.csv"),
                index=False)
    say("\n" + "=" * 78)
    say(summ.round(4).to_string(index=False))
    say("=" * 78)
    say("\n  How to read this table:")
    say("    mean_gv_add          how much additive merit the arm kept; closer to ADD is better")
    say("    add_retained_vs_ADD  the fraction retained; 1.0 means nothing was given up")
    say("    mean_gv_full         total predicted gain; higher is better")
    say("    The ideal arm has add_retained near 1 and gv_full clearly above ADD.")
    say("    FULL_naive usually has the highest gv_full and the lowest add_retained,")
    say("    i.e. it trades certain additive merit for epistasis. two_stage and")
    say("    add_floor buy non-additive gain at a small additive cost, which is the\n"
        "    form that can be defended.")
    # one automatic summary line
    if "two_stage" in plans:
        ts = summ[summ.strategy == "two_stage"].iloc[0]
        fn = summ[summ.strategy == "FULL_naive"].iloc[0]
        say(f"\n  [two_stage] additive retained {ts.add_retained_vs_ADD:.1%}, "
            f"total gain {ts.mean_gv_full:.4f}; FULL_naive retains only "
            f"{fn.add_retained_vs_ADD:.1%}. two_stage buys its non-additive gain "
            f"for {(1-ts.add_retained_vs_ADD)*100:.1f}% of additive merit.")
    say(f"\n-> {a.out}/{a.trait}_strategy_summary.csv, mating_plan.csv, "
        f"all_crosses.csv")


if __name__ == "__main__":
    main()
