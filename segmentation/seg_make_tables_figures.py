#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build the segmentation ablation table and figures from each variant's
results_summary.csv / test_results.csv.

Point --runs at one "name=path" per model variant, where path is the run
directory containing results_summary.csv (and optionally test_results.csv):

    python seg_make_tables_figures.py --out seg_figures \
        --runs "U-Net=./outputs_UNet/run_2026xxxx" \
               "U-Net+FFE=./outputs_UNetFreq/run_2026xxxx" \
               "U-Net+Mamba=./outputs_UNetMamba/run_2026xxxx" \
               "U-Net+ACPM=./outputs_UNetACPM/run_2026xxxx" \
               "U-Net+ACPM+FFE=./outputs_ACPMFreq/run_2026xxxx" \
               "U-Net+ACPM+Mamba=./outputs_ACPMMamba/run_2026xxxx" \
               "Ours (ACPM+FFE+Mamba)=./outputs_FreqMamba/run_2026xxxx"

Order the --runs arguments from simplest to full model; the table and the bar
chart keep that order so the ablation reads as a progression.

Outputs
    Table2_ablation.csv        mean +- sd over folds, plus test-set row
    Fig2a_ablation_bars.png    IoU / Dice with fold-level error bars
    Fig2b_scale_robustness.png IoU vs input scale, one line per variant
    Fig2c_efficiency.png       params vs FPS vs accuracy (bubble)
"""
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "Liberation Serif",
                              "DejaVu Serif"]
plt.rcParams["mathtext.fontset"] = "stix"

SCALE_COLS = ["scale_iou_0.8", "scale_iou_0.9", "scale_iou_1.0",
              "scale_iou_1.1", "scale_iou_1.2"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True,
                    help='one "name=path" per variant, simplest first')
    ap.add_argument("--out", default="seg_figures")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    rows, per_fold, scales = [], {}, {}
    for spec in a.runs:
        if "=" not in spec:
            raise SystemExit(f"expected name=path, got {spec!r}")
        name, path = spec.split("=", 1)
        f = os.path.join(path, "results_summary.csv")
        if not os.path.exists(f):
            print(f"[skip] {name}: {f} not found")
            continue
        R = pd.read_csv(f)
        per_fold[name] = R
        rec = dict(model=name, n_folds=len(R))
        for col, label in [("best_iou", "IoU"), ("best_dice", "Dice"),
                           ("val_hd95", "HD95"), ("val_asd", "ASD"),
                           ("val_area_error_percent", "AreaErr%"),
                           ("val_boundary_f1", "BoundaryF1")]:
            if col in R:
                rec[f"{label}_mean"] = R[col].mean()
                rec[f"{label}_sd"] = R[col].std()
        for col, label in [("params_M", "Params_M"), ("fps", "FPS"),
                           ("latency_ms", "Latency_ms"),
                           ("train_time_min", "TrainTime_min")]:
            if col in R:
                rec[label] = R[col].mean()
        t = os.path.join(path, "test_results.csv")
        if os.path.exists(t):
            T = pd.read_csv(t)
            for c in ("iou", "dice", "hd95", "asd"):
                if c in T:
                    rec[f"test_{c}"] = float(T[c].iloc[0])
        if all(c in R for c in SCALE_COLS):
            scales[name] = [R[c].mean() for c in SCALE_COLS]
        rows.append(rec)

    if not rows:
        raise SystemExit("no run directories found")
    D = pd.DataFrame(rows)
    D.to_csv(os.path.join(a.out, "Table2_ablation.csv"), index=False)
    print(D.round(4).to_string(index=False))
    print(f"\n-> {a.out}/Table2_ablation.csv")

    names = D.model.tolist()
    x = np.arange(len(names))

    # --- Fig 2a: IoU / Dice bars with fold sd ---
    fig, ax = plt.subplots(figsize=(max(7, 1.5 * len(names)), 4.6))
    w = 0.38
    ax.bar(x - w / 2, D.IoU_mean, w, yerr=D.IoU_sd, capsize=3,
           label="IoU", color="#4C72B0")
    ax.bar(x + w / 2, D.Dice_mean, w, yerr=D.Dice_sd, capsize=3,
           label="Dice", color="#DD8452")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Score (5-fold mean $\\pm$ sd)")
    ax.set_ylim(0, 1.0)
    ax.set_title("Ablation: contribution of each module")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    p = os.path.join(a.out, "Fig2a_ablation_bars.png")
    fig.savefig(p, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"-> {p}")

    # --- Fig 2b: scale robustness ---
    if scales:
        fig, ax = plt.subplots(figsize=(6.5, 4.4))
        xs = [0.8, 0.9, 1.0, 1.1, 1.2]
        for name in names:
            if name in scales:
                ax.plot(xs, scales[name], "o-", label=name, linewidth=1.4)
        ax.set_xlabel("Input scale factor")
        ax.set_ylabel("IoU")
        ax.set_title("Robustness to input scale")
        ax.legend(frameon=False, fontsize=8)
        ax.grid(alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        p = os.path.join(a.out, "Fig2b_scale_robustness.png")
        fig.savefig(p, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"-> {p}")

    # --- Fig 2c: accuracy vs cost ---
    if "Params_M" in D and "FPS" in D:
        fig, ax = plt.subplots(figsize=(6.5, 4.6))
        sizes = 40 + 260 * (D.Params_M / max(D.Params_M.max(), 1e-9))
        sc = ax.scatter(D.FPS, D.IoU_mean, s=sizes, c=range(len(D)),
                        cmap="viridis", alpha=0.85, edgecolor="black",
                        linewidth=0.6, zorder=3)
        for xi, yi, nm in zip(D.FPS, D.IoU_mean, names):
            ax.annotate(nm, (xi, yi), textcoords="offset points",
                        xytext=(7, 4), fontsize=8)
        ax.set_xlabel("Inference speed (FPS)")
        ax.set_ylabel("IoU (5-fold mean)")
        ax.set_title("Accuracy vs speed  (bubble area $\\propto$ parameters)")
        ax.grid(alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        p = os.path.join(a.out, "Fig2c_efficiency.png")
        fig.savefig(p, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"-> {p}")

    print("\nFor the qualitative panel, use the already-generated files in")
    print("  <run>/fold_<best>/predictions/sample_*.png  (original/GT/pred/")
    print("  overlay) and diff_*.png (green TP, red FP, blue FN). Pick 3-4")
    print("  shells spanning low to high purple coverage rather than the best")
    print("  ones, and say in the caption how they were chosen.")


if __name__ == "__main__":
    main()
