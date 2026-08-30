#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 allocation_core.py -- the six fixed allocation arms, called by
                       trait_epistasis_to_allocation.py
================================================================================

 sel empty            -> only the additive arm is produced (a trait with no
                         validated pair degrades to a purely additive plan,
                         which is the system's decision, not a failure)
 sel non-empty        -> all six arms are produced as fixed comparison arms

 no_dominance = True  -> the dominance module is blocked by the material
                         (mean heterozygosity at or below the threshold, i.e.
                         inbred lines or doubled haploids). The baseline
                         becomes A + aggregate instead of A + D + aggregate,
                         the ADD+DOM arm is not produced, and the weighted arm
                         carries no dominance term. This is recorded as a
                         property of the germplasm; it is NOT a finding that
                         dominance is zero. Two inbred lines crossed produce a
                         fully heterozygous F1 in which dominance is expressed
                         in full -- what is missing is any information in the
                         reference panel with which to estimate it.

 The six arms and what each is for:
   ADD          additive breeding value only; the reference plan
   ADD+DOM      adds the expected dominance deviation of each cross
   FULL_naive   adds the validated epistatic pairs, unweighted
   two_stage    additive percentile filter first, then rank on non-additive
   weighted     each component weighted by its estimated variance fraction
   add_floor    rank on FULL subject to a hard additive floor; this arm may
                return fewer than n_crosses plans, and its means are then
                averages over a smaller, more highly selected set
================================================================================
"""
from __future__ import annotations

import os
from itertools import combinations

import numpy as np
import pandas as pd

import pig_epistasis_pipeline as Q

# Mean heterozygosity at or below this value blocks the dominance module.
HET_BLOCK_THRESHOLD = 0.02


def mean_heterozygosity(X):
    """Mean fraction of heterozygous calls per individual."""
    return float(np.mean(X == 1))


def _greedy(df, score, n_crosses, max_use, f_max):
    order = np.argsort(-score)
    use, chosen = {}, []
    F = df.progeny_F.values
    S = df.sire.values
    D = df.dam.values
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


def _plan(df, idx, name):
    p = df.iloc[idx].copy()
    p.insert(0, "strategy", name)
    return p


def run_strategies(a, X, y, chrom, pos, snp_names, Ha, Hd, Xc, Kfull, meta,
                   agg, sel, out_dir, no_dominance=None):
    n, m = X.shape
    mu = X.mean(0)
    Xfix = np.ones((n, 1))
    has_epi = len(sel) > 0

    # ---- capability: can this material support a dominance term at all? ----
    if no_dominance is None:
        no_dominance = getattr(a, "no_dominance", False)
    het = mean_heterozygosity(X)
    if not no_dominance and het <= HET_BLOCK_THRESHOLD:
        no_dominance = True
        print(f"    [capability] mean heterozygosity {het:.4f} <= "
              f"{HET_BLOCK_THRESHOLD}: the dominance module is BLOCKED for "
              f"this material. Fitting it here would divide by a variance of "
              f"essentially zero. This is a property of the germplasm, not a "
              f"result about the trait.")
    if no_dominance:
        var_hd = float(np.sum(np.var(Hd, axis=0)))
        print(f"    [capability] dominance blocked (sum of per-marker "
              f"dominance-code variance = {var_hd:.3e}); baseline is "
              f"A{'+aggregate' if agg else ''}, the ADD+DOM arm is not "
              f"produced, and the weighted arm carries no dominance term.")

    base_keys = ["A"] if no_dominance else ["A", "D"]

    if has_epi:
        ptype = sel[0][3]
        KT = Q.pairs_kernel(sel, ptype, X, Ha, Hd, Xc)
        K = {k: Kfull[k] for k in base_keys}
        K["TOP"] = KT
        keys = base_keys + ["TOP"]
    else:
        ptype = None
        K = {k: Kfull[k] for k in base_keys}
        keys = list(base_keys)

    mm, alpha, var = Q.offspring_components(a, X, Ha, Hd, chrom, K, meta,
                                            y, Xfix, keys)
    tot = sum(v for k, v in var.items() if k != "_resid") + var["_resid"]
    var_frac = {k: var[k] / tot for k in keys}
    print("    variance fractions: " +
          "  ".join(f"{k}={var_frac[k]:.4f}" for k in keys) +
          f"  resid={var['_resid'] / tot:.4f}")

    a_hat = var["A"] * (Ha.T @ alpha) / (meta["sA"] * (
        2 * np.sum((mu / 2) * (1 - mu / 2)) + 1e-12))
    if "D" in var:
        d_hat = var["D"] * (Hd.T @ alpha) / (meta["sD"] * (
            np.sum(np.var(Hd, 0)) + 1e-12))
    else:
        d_hat = np.zeros(m)

    if has_epi:
        pair_idx = [(i, j) for _, i, j, _ in sel]
        feat_mean = np.array([(Ha[:, i] * Ha[:, j]).mean()
                              for i, j in pair_idx])
        feat_std = np.array([(Ha[:, i] * Ha[:, j]).std() for i, j in pair_idx])
        feat_std[feat_std < 1e-10] = 1.0
        # pair_feature(typ, tag, ...): the first argument is the interaction
        # family and the second the direction tag. Passing the tag twice
        # happens to work for MULT, where the tag is unused, and silently
        # mis-encodes the asymmetric AD family.
        Z = np.column_stack([Q.pair_feature(ptype, tg, X, Ha, Hd, Xc, i, j)
                             for (_, i, j, tg) in sel])
        top_eff = var["TOP"] * (Z.T @ alpha) / (Z.shape[1] + 1e-12)
        ai = np.array([p[0] for p in pair_idx])
        aj = np.array([p[1] for p in pair_idx])

    rng = np.random.default_rng(a.seed)
    cand = np.sort(rng.choice(n, min(a.n_candidates, n), replace=False))
    nc = len(cand)
    p_aa = np.mean(X == 1, axis=0)
    var_A = np.var(Ha, axis=0)
    beta_d = np.where(var_A > 1e-12,
                      (p_aa * (1 - mu)) / np.maximum(var_A, 1e-12), 0.0)

    pairs = list(combinations(range(nc), 2))
    npair = len(pairs)
    ii = np.array([p[0] for p in pairs])
    jj = np.array([p[1] for p in pairs])
    add = np.empty(npair)
    dom = np.zeros(npair)
    epi = np.zeros(npair)
    inb = np.empty(npair)
    het_exp = np.empty(npair)
    CH = 2000
    for s in range(0, npair, CH):
        e = min(s + CH, npair)
        a_ = X[cand][ii[s:e]] / 2.0
        b_ = X[cand][jj[s:e]] / 2.0
        e_ha = (a_ + b_) - mu[None, :]
        p_het = a_ * (1 - b_) + (1 - a_) * b_
        add[s:e] = e_ha @ a_hat
        if not no_dominance:
            e_hd = p_het - p_aa[None, :] - beta_d[None, :] * e_ha
            dom[s:e] = e_hd @ d_hat
        het_exp[s:e] = p_het.mean(1)
        if has_epi and ptype == "MULT":
            feat = e_ha[:, ai] * e_ha[:, aj]
            feat = (feat - feat_mean[None, :]) / feat_std[None, :]
            epi[s:e] = feat @ top_eff
        inb[s:e] = meta["KA_raw"][cand[ii[s:e]], cand[jj[s:e]]] / 2.0
    if has_epi and ptype != "MULT":
        print(f"    [note] the closed-form expected offspring value is "
              f"implemented for the multiplicative family only; pairs of "
              f"type {ptype} contribute 0 to gv_epi here.")

    df = pd.DataFrame(dict(
        sire=cand[ii], dam=cand[jj], gv_add=add, gv_dom=dom, gv_epi=epi,
        gv_nonadd=dom + epi, gv_add_dom=add + dom, gv_full=add + dom + epi,
        progeny_F=inb, exp_heterozygosity=het_exp))
    df.to_csv(os.path.join(out_dir, f"{a.trait}_all_crosses.csv"), index=False)

    plans = {"ADD": _plan(df, _greedy(df, df.gv_add.values, a.n_crosses,
                                      a.max_use, a.f_max), "ADD")}

    if not has_epi:
        summ = pd.DataFrame([dict(
            trait=a.trait, strategy="ADD", n_crosses=len(plans["ADD"]),
            n_pairs_used=0, dominance_blocked=bool(no_dominance),
            mean_heterozygosity=het,
            mean_gv_add=float(plans["ADD"].gv_add.mean()),
            mean_gv_full=float(plans["ADD"].gv_full.mean()),
            add_retained_vs_ADD=1.0, overlap_with_ADD=1.0, crosses_changed=0,
            epi_var_frac=0.0)])
        summ.to_csv(os.path.join(out_dir, f"{a.trait}_strategy_summary.csv"),
                    index=False)
        pd.concat(plans.values(), ignore_index=True).to_csv(
            os.path.join(out_dir, f"{a.trait}_mating_plan.csv"), index=False)
        return summ

    if not no_dominance:
        plans["ADD+DOM"] = _plan(df, _greedy(df, df.gv_add_dom.values,
                                             a.n_crosses, a.max_use, a.f_max),
                                 "ADD+DOM")
    plans["FULL_naive"] = _plan(df, _greedy(df, df.gv_full.values, a.n_crosses,
                                            a.max_use, a.f_max), "FULL_naive")

    # two_stage: additive percentile filter, then rank on non-additive within
    thr = np.quantile(df.gv_add.values, a.add_quantile)
    elig = df[df.gv_add >= thr].copy()
    idx_sub = _greedy(elig.reset_index(drop=True), elig.gv_nonadd.values,
                      a.n_crosses, a.max_use, a.f_max)
    gsel = elig.index.values[idx_sub]
    plans["two_stage"] = _plan(df, [df.index.get_loc(g) for g in gsel],
                               "two_stage")

    # weighted: each component scaled by its variance fraction
    w_d = 0.0 if no_dominance else var_frac["D"] / max(var_frac["A"], 1e-9)
    w_e = var_frac["TOP"] / max(var_frac["A"], 1e-9)
    if var_frac["A"] < 1e-6:
        print(f"    [warn] the additive variance fraction is "
              f"{var_frac['A']:.2e}; the weighted arm divides by it and its "
              f"weights are not interpretable for this trait. Report the "
              f"weighted arm as NA rather than as a result.")
    score = (df.gv_add.values + w_d * df.gv_dom.values
             + w_e * df.gv_epi.values)
    plans["weighted"] = _plan(df, _greedy(df, score, a.n_crosses, a.max_use,
                                          a.f_max), "weighted")

    # add_floor: hard additive floor taken from the ADD plan itself
    floor = np.quantile(plans["ADD"].gv_add.values, a.floor_quantile)
    sub = df[df.gv_add.values >= floor].copy()
    idx_sub = _greedy(sub.reset_index(drop=True), sub.gv_full.values,
                      a.n_crosses, a.max_use, a.f_max)
    gsel = sub.index.values[idx_sub]
    plans["add_floor"] = _plan(df, [df.index.get_loc(g) for g in gsel],
                               "add_floor")
    if len(plans["add_floor"]) < a.n_crosses:
        print(f"    [note] add_floor met its additive floor by planning "
              f"{len(plans['add_floor'])} of {a.n_crosses} crosses; its means "
              f"are averages over a smaller and more highly selected set.")

    pd.concat(plans.values(), ignore_index=True).to_csv(
        os.path.join(out_dir, f"{a.trait}_mating_plan.csv"), index=False)

    base = set(zip(plans["ADD"].sire, plans["ADD"].dam))
    add_ADD = plans["ADD"].gv_add.mean()
    rows = []
    for name, p in plans.items():
        s = set(zip(p.sire, p.dam))
        rows.append(dict(
            trait=a.trait, strategy=name, n_crosses=len(p),
            n_pairs_used=len(sel), epi_var_frac=var_frac.get("TOP", 0.0),
            dominance_blocked=bool(no_dominance), mean_heterozygosity=het,
            mean_gv_add=float(p.gv_add.mean()),
            mean_gv_dom=float(p.gv_dom.mean()),
            mean_gv_epi=float(p.gv_epi.mean()),
            mean_gv_full=float(p.gv_full.mean()),
            add_retained_vs_ADD=float(p.gv_add.mean() / add_ADD)
            if abs(add_ADD) > 1e-12 else np.nan,
            mean_F=float(p.progeny_F.mean()),
            overlap_with_ADD=len(s & base) / max(len(base), 1),
            crosses_changed=len(base) - len(s & base)))
    summ = pd.DataFrame(rows)
    summ.to_csv(os.path.join(out_dir, f"{a.trait}_strategy_summary.csv"),
                index=False)
    return summ
