#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dominance and epistasis deltas FROM THE EXISTING GEBV STORE. No refitting.

Every <trait>_gebv_store.csv.gz already on disk (in pig_bias_<trait>/) has one
row per (trait, seed, model), with the test-set individual indices, GEBVs, and
adjusted phenotype all stored as ';'-joined lists. That is exactly what a
paired bootstrap delta needs. This script reads it and computes:

    dominance: A+D vs A         (never computed anywhere before)
    epistasis: A+D+MULT_t vs A+D
    epistasis: A+D+HAD vs A+D

using the SAME convention already established in the pipeline: the adjusted
phenotype (y_adj) of the BASELINE model in each comparison is used as the
common reference for both sides of the pair.

Runs in seconds. Writes dominance_delta_pcc.csv into the SAME pig_bias_<trait>/
directory the GEBVs came from -- no new folder.

    python dominance_from_gebv.py --traits bf lmd lmp ltn rtn tpd ttn
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pig_epistasis_pipeline as Q


def load_store(path):
    """trait,seed,model,idx,gebv,y_adj -> {(seed, model): (idx, gebv, y_adj)}"""
    df = pd.read_csv(path)
    out = {}
    for _, r in df.iterrows():
        idx = np.array(str(r["idx"]).split(";"), dtype=int)
        gebv = np.array(str(r["gebv"]).split(";"), dtype=float)
        yadj = np.array(str(r["y_adj"]).split(";"), dtype=float)
        out[(int(r["seed"]), str(r["model"]))] = (idx, gebv, yadj)
    return out


def paired_delta(store, seeds, model_lo, model_hi, boot):
    """
    Pair by seed, align by idx (defensive -- do not assume ordering matches
    even though it should, since the split does not depend on the model), and
    compute Q.boot_delta using model_lo's y_adj as the common reference.
    """
    rows = []
    for seed in seeds:
        lo = store.get((seed, model_lo))
        hi = store.get((seed, model_hi))
        if lo is None or hi is None:
            continue
        idx_lo, g_lo, ya_lo = lo
        idx_hi, g_hi, _ = hi
        if not np.array_equal(idx_lo, idx_hi):
            # defensive path: align by individual index rather than trust
            # positional order, in case a future run ever reorders the test
            # set between models
            common, pos_lo, pos_hi = np.intersect1d(idx_lo, idx_hi,
                                                     return_indices=True)
            if len(common) < 10:
                continue
            g_lo, g_hi, ya_lo = g_lo[pos_lo], g_hi[pos_hi], ya_lo[pos_lo]
        d, lo_ci, hi_ci = Q.boot_delta(ya_lo, g_lo, g_hi, boot, seed)
        rows.append(dict(seed=seed, dPCC=d, ci_lo=lo_ci, ci_hi=hi_ci))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traits", nargs="+", required=True)
    ap.add_argument("--dir_pattern", default="pig_bias_{trait}",
                    help="directory holding <trait>_gebv_store.csv.gz")
    ap.add_argument("--boot", type=int, default=1000)
    a = ap.parse_args()

    all_rows = []
    for trait in a.traits:
        d = a.dir_pattern.format(trait=trait)
        p = os.path.join(d, f"{trait}_gebv_store.csv.gz")
        if not os.path.exists(p):
            print(f"[skip] {trait}: {p} not found")
            continue
        store = load_store(p)
        seeds = sorted({s for s, _ in store})
        models_present = sorted({m for _, m in store})
        print(f"\n{trait}: {len(seeds)} seeds, models {models_present}")

        comparisons = [("dominance: A+D vs A", "A", "A+D")]
        for hi in ("A+D+MULT_t", "A+D+HAD"):
            if hi in models_present:
                comparisons.append((f"epistasis: {hi} vs A+D", "A+D", hi))

        for name, lo, hi in comparisons:
            if lo not in models_present or hi not in models_present:
                print(f"    [skip] {name}: model missing from store")
                continue
            R = paired_delta(store, seeds, lo, hi, a.boot)
            if R.empty:
                print(f"    [skip] {name}: no seeds could be paired")
                continue
            pos_frac = float(np.mean(R.dPCC > 0))
            mean_d = float(R.dPCC.mean())
            print(f"    {name:28s} dPCC {mean_d:+.5f}  pos_frac {pos_frac:.2f}"
                  f"  (n_seeds={len(R)})")
            for _, r in R.iterrows():
                all_rows.append(dict(trait=trait, comparison=name,
                                     seed=int(r["seed"]), dPCC=r["dPCC"],
                                     ci_lo=r["ci_lo"], ci_hi=r["ci_hi"]))

        if all_rows:
            out_df = pd.DataFrame([r for r in all_rows if r["trait"] == trait])
            out_df.to_csv(os.path.join(d, "dominance_delta_pcc.csv"),
                          index=False)
            print(f"    -> written to {d}/dominance_delta_pcc.csv")

    if all_rows:
        S = (pd.DataFrame(all_rows).groupby(["trait", "comparison"])
             .agg(dPCC=("dPCC", "mean"), sd=("dPCC", "std"),
                  pos_frac=("dPCC", lambda v: float(np.mean(np.asarray(v) > 0))),
                  n_seeds=("dPCC", "size"))
             .reset_index().sort_values(["comparison", "dPCC"],
                                        ascending=[True, False]))
        print("\n" + "=" * 74)
        print(S.round(5).to_string(index=False))
        S.to_csv("dominance_epistasis_summary.csv", index=False)
        print("\n-> dominance_epistasis_summary.csv (this directory)")


if __name__ == "__main__":
    main()
