#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Real IFAM benchmark (Tang et al.) vs our own results, per trait.

Takes the published per-split accuracy table (Model, Trait, Accuracy; 20 rows
per model per trait) and our own <trait>_accuracy_bias.csv, and writes one
comparison table per trait into the SAME pig_bias_<trait>/ directory the rest
of that trait's output already lives in.

    GBLUP  is the published baseline -- compare against our "A" model.
    IFAM   is their non-additive method -- compare against our
           "A+D+MULT_t" / "A+D+HAD".

Trait names differ slightly (their "ELMP" vs our "lmp") and are mapped
EXPLICITLY below rather than guessed by string similarity, because a silent
wrong mapping would misattribute an entire trait's benchmark.

    python ifam_compare.py --ifam_csv ifam_benchmark_raw.csv \
        --traits bf lmd lmp ltn rtn tpd ttn
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

# our trait code -> the label used in the pasted IFAM table
TRAIT_MAP = {"bf": "BF", "lmd": "LMD", "lmp": "ELMP", "ltn": "LTN",
            "rtn": "RTN", "tpd": "TPD", "ttn": "TTN"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ifam_csv", required=True,
                    help="Model,Trait,Accuracy long-format table")
    ap.add_argument("--traits", nargs="+", required=True)
    ap.add_argument("--dir_pattern", default="pig_bias_{trait}")
    a = ap.parse_args()

    raw = pd.read_csv(a.ifam_csv)
    unmapped = set(raw.Trait.unique()) - set(TRAIT_MAP.values())
    if unmapped:
        print(f"[note] IFAM table has trait labels not in TRAIT_MAP, "
              f"ignored: {sorted(unmapped)}")

    bench = (raw.groupby(["Model", "Trait"]).Accuracy
             .agg(mean="mean", sd="std", n="count").reset_index())

    rows = []
    for trait in a.traits:
        ifam_label = TRAIT_MAP.get(trait)
        if ifam_label is None:
            print(f"[skip] {trait}: no entry in TRAIT_MAP; add one rather "
                  f"than guess")
            continue
        d = a.dir_pattern.format(trait=trait)
        p = os.path.join(d, f"{trait}_accuracy_bias.csv")
        if not os.path.exists(p):
            print(f"[skip] {trait}: {p} not found")
            continue
        ours = pd.read_csv(p)

        def ours_row(model):
            m = ours[ours.model == model]
            return (float(m.accuracy.iloc[0]), float(m.acc_sd.iloc[0])) \
                if len(m) else (np.nan, np.nan)

        def bench_row(model_name):
            m = bench[(bench.Model == model_name) & (bench.Trait == ifam_label)]
            return (float(m["mean"].iloc[0]), float(m["sd"].iloc[0]),
                    int(m["n"].iloc[0])) if len(m) else (np.nan, np.nan, 0)

        our_a, our_a_sd = ours_row("A")
        our_hi_name = "A+D+MULT_t" if "A+D+MULT_t" in ours.model.values \
            else "A+D+HAD"
        our_hi, our_hi_sd = ours_row(our_hi_name)
        gblup_m, gblup_sd, gblup_n = bench_row("GBLUP")
        ifam_m, ifam_sd, ifam_n = bench_row("IFAM")

        row = dict(
            trait=trait, ifam_label=ifam_label,
            our_A=our_a, our_A_sd=our_a_sd,
            published_GBLUP=gblup_m, published_GBLUP_sd=gblup_sd,
            published_GBLUP_n=gblup_n,
            diff_A_minus_GBLUP=(our_a - gblup_m
                                if np.isfinite(our_a) and np.isfinite(gblup_m)
                                else np.nan),
            our_best_nonadditive=our_hi_name,
            our_best_nonadditive_acc=our_hi, our_best_nonadditive_sd=our_hi_sd,
            published_IFAM=ifam_m, published_IFAM_sd=ifam_sd,
            published_IFAM_n=ifam_n,
            diff_nonadditive_minus_IFAM=(our_hi - ifam_m
                                        if np.isfinite(our_hi)
                                        and np.isfinite(ifam_m) else np.nan))
        rows.append(row)

        out_path = os.path.join(d, "ifam_comparison.csv")
        pd.DataFrame([row]).to_csv(out_path, index=False)
        print(f"{trait:5s} A={our_a:.4f} vs GBLUP={gblup_m:.4f}"
              f" (diff {row['diff_A_minus_GBLUP']:+.4f})  |  "
              f"{our_hi_name}={our_hi:.4f} vs IFAM={ifam_m:.4f}"
              f" (diff {row['diff_nonadditive_minus_IFAM']:+.4f})"
              f"  -> {out_path}")

    if rows:
        pd.DataFrame(rows).to_csv("ifam_comparison_all_traits.csv", index=False)
        print("\n-> ifam_comparison_all_traits.csv (this directory)")
        print("\n  Read diff_A_minus_GBLUP as: are we starting from the same")
        print("  additive baseline as the published benchmark? A large gap")
        print("  here means the comparison isn't like-for-like (different")
        print("  splits, different marker set, or a real discrepancy) and")
        print("  the non-additive comparison inherits that uncertainty.")


if __name__ == "__main__":
    main()
