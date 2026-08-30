#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pig summary table and two figures, built from per-trait files already on disk.

Inputs, all produced by earlier stages of the pipeline:
  pig_bias_<trait>/<trait>_accuracy_bias.csv     per-split accuracy and bias
  pig_bias_<trait>/dominance_delta_pcc.csv       from dominance_from_gebv.py,
                                                  computed from stored GEBVs
                                                  without refitting
  pig_bias_<trait>/ifam_comparison.csv           from ifam_compare.py
  pig_finish_out/calibration.csv                 the completed 6-cell
                                                  calibration grid

No refitting happens here -- this is purely tabulation and plotting.

    python pig_make_tables_figures.py \
        --traits bf lmd lmp rtn \
        --calib_csv pig_finish_out/calibration.csv \
        --out figures
"""
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from trait_names import pretty as _pretty
except ImportError:                                   # pragma: no cover
    def _pretty(t, sp=None): return str(t)
    print("[note] trait_names.py not importable; raw trait codes will be used")

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "Liberation Serif",
                              "DejaVu Serif"]
plt.rcParams["mathtext.fontset"] = "stix"   # a Times-like math font, so the
                                            # Greek/Delta symbols match


def declutter_labels(ax, points, min_sep_pts=30):
    """
    points: list of (x_data, y_data, text, color). Returns nothing; draws the
    labels directly, offsetting any that would otherwise overlap and adding a
    thin leader line back to the true data point when an offset was needed.

    Overlap is resolved in DISPLAY space (points/pixels), not data units,
    because "too close to read" is a rendering-size property, not a data-
    value property -- two stars 0.001 apart on a 0-0.15 axis are visually
    identical to two stars 0.0001 apart once the axis is compressed enough,
    and a purely data-unit threshold would under- or over-correct depending
    on the axis range.
    """
    if not points:
        return
    fig = ax.figure
    fig.canvas.draw()  # positions must be finalised before transforming
    disp = [(ax.transData.transform((x, y))[1], x, y, t, c)
           for x, y, t, c in points]
    disp.sort()  # by display-y, ascending

    # greedy separation: walk once top-to-bottom pushing any label that is
    # too close to the previous one further away; then a second pass the
    # other direction catches pushes that created a new collision below.
    ys = [d[0] for d in disp]
    for _ in range(2):
        for i in range(1, len(ys)):
            if ys[i] - ys[i - 1] < min_sep_pts:
                ys[i] = ys[i - 1] + min_sep_pts
        for i in range(len(ys) - 2, -1, -1):
            if ys[i + 1] - ys[i] < min_sep_pts:
                ys[i] = ys[i + 1] - min_sep_pts

    inv = ax.transData.inverted()
    for (orig_disp_y, x, y, text, color), new_disp_y in zip(disp, ys):
        moved = abs(new_disp_y - orig_disp_y) > 1.0
        label_x_disp, _ = ax.transData.transform((x, y))
        label_x_disp += 10          # small fixed horizontal offset to the
                                    # right of the marker
        label_x, label_y = inv.transform((label_x_disp, new_disp_y))
        ax.annotate(text, xy=(x, y), xytext=(label_x, label_y),
                   textcoords="data", fontsize=8, color=color, va="center",
                   ha="left", clip_on=False,
                   arrowprops=(dict(arrowstyle="-", color=color,
                                    linewidth=0.6, shrinkA=2, shrinkB=4)
                              if moved else None))


def load_trait_tables(traits, dir_pattern):
    acc_rows, dom_rows, ifam_rows = [], [], []
    for t in traits:
        d = dir_pattern.format(trait=t)
        acc_p = os.path.join(d, f"{t}_accuracy_bias.csv")
        dom_p = os.path.join(d, "dominance_delta_pcc.csv")
        ifam_p = os.path.join(d, "ifam_comparison.csv")
        if os.path.exists(acc_p):
            acc_rows.append(pd.read_csv(acc_p))
        else:
            print(f"[warn] {acc_p} not found, skipping accuracy for {t}")
        if os.path.exists(dom_p):
            dom_rows.append(pd.read_csv(dom_p))
        else:
            print(f"[warn] {dom_p} not found -- run dominance_from_gebv.py "
                  f"first for {t}")
        if os.path.exists(ifam_p):
            ifam_rows.append(pd.read_csv(ifam_p))
    acc = pd.concat(acc_rows, ignore_index=True) if acc_rows else None
    dom = pd.concat(dom_rows, ignore_index=True) if dom_rows else None
    ifam = pd.concat(ifam_rows, ignore_index=True) if ifam_rows else None
    return acc, dom, ifam


def build_table1(traits, acc, dom, ifam):
    rows = []
    for t in traits:
        row = dict(trait=_pretty(t, "pig"), trait_code=t)
        if acc is not None:
            a = acc[(acc.trait == t)]
            for model in ("A", "A+D", "A+D+MULT_t"):
                m = a[a.model == model]
                if len(m):
                    row[f"accuracy_{model}"] = round(float(m.accuracy.iloc[0]), 4)
        if dom is not None:
            dsub = dom[dom.trait == t]
            drow = dsub[dsub.comparison.str.startswith("dominance")]
            erow = dsub[dsub.comparison.str.contains("MULT_t")]
            if len(drow):
                row["dominance_n_splits"] = int(len(drow))
                row["dominance_dPCC"] = round(float(drow.dPCC.mean()), 5)
                row["dominance_pos_frac"] = round(
                    float((drow.dPCC > 0).mean()), 2)
            if len(erow):
                row["epistasis_n_splits"] = int(len(erow))
                row["epistasis_dPCC"] = round(float(erow.dPCC.mean()), 5)
                row["epistasis_pos_frac"] = round(
                    float((erow.dPCC > 0).mean()), 2)
        if ifam is not None:
            i = ifam[ifam.trait == t]
            if len(i):
                row["diff_A_minus_GBLUP"] = round(
                    float(i.diff_A_minus_GBLUP.iloc[0]), 4)
                row["diff_nonadditive_minus_IFAM"] = round(
                    float(i.diff_nonadditive_minus_IFAM.iloc[0]), 4)
        rows.append(row)
    return pd.DataFrame(rows)


def fig_dominance_epistasis(T, out_path):
    T = T.sort_values("epistasis_dPCC", ascending=False, na_position="last")
    traits = T.trait.values          # already the paper abbreviations
    dom = T.get("dominance_dPCC", pd.Series([np.nan] * len(T))).values
    epi = T.get("epistasis_dPCC", pd.Series([np.nan] * len(T))).values

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(traits))
    w = 0.35
    ax.bar(x - w / 2, dom, w, label="dominance (A+D vs A)", color="#4C72B0")
    ax.bar(x + w / 2, epi, w, label="epistasis (A+D+MULT_t vs A+D)",
           color="#DD8452")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(traits)
    ax.set_ylabel(r"$\Delta$PCC (paired bootstrap mean)")
    ax.set_title("Dominance and epistasis contributions by trait")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"-> {out_path}")


def fig_calibration(calib_csv, real_points, out_path):
    C = pd.read_csv(calib_csv)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.5), sharex=False)
    for ax, module in zip(axes, ("dominance", "epistasis")):
        g = C[C.module == module].sort_values("injected")
        xs, ys = g.injected.values, g.mean_dPCC.values
        ax.plot(xs, ys, "o-", color="#4C72B0",
               label="calibration curve (simulated)")
        ax.axhline(0, color="gray", linewidth=0.7)
        fpr = g[g.injected == 0].detection_rate
        fpr = float(fpr.iloc[0]) if len(fpr) else float("nan")
        ax.set_title(f"{module}  (false-positive rate {fpr:.2f})")
        ax.set_xlabel("injected variance")
        if module == "dominance":
            ax.set_ylabel(r"mean $\Delta$PCC")

        x_max = xs.max()
        order = np.argsort(ys)
        y_lo, y_hi = ys[order][0], ys[order][-1]
        label_points = []
        for trait, d in real_points.get(module, {}).items():
            # Placement is computed, never hand-tuned per trait: an earlier
            # version carried hard-coded nudges keyed on trait names, which
            # silently did nothing once the trait set changed.
            if d > y_hi:
                x_pos = x_max * 1.06
                ax.scatter([x_pos], [d], marker="*", s=160,
                          color="#C44E52", zorder=5, clip_on=False)
                label_points.append(
                    (x_pos, d, f"{_pretty(trait, 'pig')} "
                               f"(exceeds tested range)", "#C44E52"))
            else:
                x_pos = float(np.interp(np.clip(d, y_lo, y_hi),
                                       ys[order], xs[order]))
                ax.scatter([x_pos], [d], marker="*", s=140,
                          color="#DD8452", zorder=5)
                label_points.append(
                    (x_pos, d, _pretty(trait, "pig"), "#8C4A2F"))
        ax.set_xlim(0, x_max * 1.55)
        declutter_labels(ax, label_points)
        ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle("Detection calibration curves, with real trait values "
                 "positioned by their true equivalent injected variance",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"-> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traits", nargs="+",
                    default=["bf", "lmd", "lmp", "rtn"],
                    help="the four traits carried into the manuscript; the "
                         "codes are the ones used on disk and are converted "
                         "to their paper abbreviations on output "
                         "(lmp -> ELMP)")
    ap.add_argument("--dir_pattern", default="pig_bias_{trait}")
    ap.add_argument("--calib_csv", default="pig_finish_out/calibration.csv")
    ap.add_argument("--out", default="figures")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    acc, dom, ifam = load_trait_tables(a.traits, a.dir_pattern)
    T = build_table1(a.traits, acc, dom, ifam)
    for col in ("dominance_n_splits", "epistasis_n_splits"):
        if col in T.columns and T[col].nunique(dropna=True) > 1:
            print(f"[warn] {col} differs between traits "
                  f"({dict(zip(T.trait, T[col]))}). A positive fraction is "
                  f"only comparable at a fixed split count -- harmonise the "
                  f"per-split files before quoting these rows together.")
    t1_path = os.path.join(a.out, "Table1_summary.csv")
    T.to_csv(t1_path, index=False)
    print(f"-> {t1_path}")
    print("\n" + T.to_string(index=False))

    fig_dominance_epistasis(T, os.path.join(a.out,
                                            "Fig_pig_dominance_epistasis.png"))

    if os.path.exists(a.calib_csv):
        key = T.trait_code if "trait_code" in T.columns else T.trait
        real_points = {
            "dominance": dict(zip(key, T.get("dominance_dPCC", []))),
            "epistasis": dict(zip(key, T.get("epistasis_dPCC", []))),
        }
        # only annotate the traits that actually cleared their own threshold,
        # so the plot doesn't get cluttered with null traits sitting at zero
        real_points = {m: {t: d for t, d in pts.items()
                          if pd.notna(d) and abs(d) > 1e-4}
                      for m, pts in real_points.items()}
        fig_calibration(a.calib_csv, real_points,
                        os.path.join(a.out, "Fig4a_calibration_pig.png"))
    else:
        print(f"[skip] {a.calib_csv} not found -- calibration figure needs it")


if __name__ == "__main__":
    main()