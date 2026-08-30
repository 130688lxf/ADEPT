#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 make_arabidopsis_figure.py -- the four-panel Arabidopsis positive-control figure
================================================================================

 Reads five CSV files from arabidopsis_final/ (column names taken from the
 actual files, not guessed):
   Q2_correction_schemes.csv  rule,scheme,F,p,beta,se,n,h2,note   interaction test
   Q3_main_effect_ruler.csv   rule,gene,snp,scheme,F,p,beta       main effects
   Q4_injection.csv           injected_frac,n_rep,rate_C0..C3     injection test
   Q_confounding_R2.csv       rule,R2_interaction_on_PC,...       confounding with structure
   Q_replication.csv          phenotype,n,F_none,p_none,F_GRM,p_GRM

 The four panels make one point: the apparent epistatic signal is produced by
 the rule used to select the loci.
   (a) interaction F by locus-selection rule and correction scheme. Choosing
       loci by main effect (R2_topmain) gives a large F; choosing them blindly
       as the nearest gene (R1_nearest) gives nothing significant.
   (b) main effects of FRI and FLC under the four corrections. FRI is erased
       by correction while FLC survives.
   (c) injection test: injected variance against detection rate. The zero-
       injection row is the false-positive rate, and C2 lying close to C0
       shows that correction does not destroy real signal.

   (d) replication across eight flowering traits, before and after correction.
       A real interaction should reappear in several of them.

     python make_arabidopsis_figure.py --dir arabidopsis_final --out arab_figs
================================================================================
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
plt.rcParams["font.serif"] = ["Times New Roman", "Liberation Serif", "DejaVu Serif"]
plt.rcParams["mathtext.fontset"] = "stix"

SCHEMES = ["C0_none", "C1_PConly", "C2_GRMonly", "C3_PC_plus_GRM"]
SCHEME_SHORT = {"C0_none": "none", "C1_PConly": "PC", "C2_GRMonly": "GRM",
                "C3_PC_plus_GRM": "PC+GRM"}
RULE_SHORT = {"R1_nearest": "nearest gene\n(blind)",
              "R2_topmain": "strongest main effect\n(selected)",
              "R3_maxMAF": "highest MAF\n(blind)"}
SCOL = {"C0_none": "#4C72B0", "C1_PConly": "#DD8452",
        "C2_GRMonly": "#55A868", "C3_PC_plus_GRM": "#C44E52"}


def say(*a):
    print(*a, flush=True)


def need(d, name):
    p = os.path.join(d, name)
    if not os.path.exists(p):
        raise SystemExit(f"not found: {p}")
    return pd.read_csv(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="the arabidopsis_final directory")
    ap.add_argument("--out", default="arab_figs")
    ap.add_argument("--alpha", type=float, default=0.05)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    Q2 = need(a.dir, "Q2_correction_schemes.csv")
    Q3 = need(a.dir, "Q3_main_effect_ruler.csv")
    Q4 = need(a.dir, "Q4_injection.csv")
    QC = need(a.dir, "Q_confounding_R2.csv")
    QR = need(a.dir, "Q_replication.csv")
    say(f"[read] Q2 {len(Q2)} rows, Q3 {len(Q3)} rows, Q4 {len(Q4)} rows, "
        f"confounding {len(QC)} rows, replication {len(QR)} rows")

    fig = plt.figure(figsize=(13.5, 9.4))
    gs = fig.add_gridspec(2, 2, hspace=.42, wspace=.26)

    # ---------- (a) interaction F by rule and correction ----------
    ax = fig.add_subplot(gs[0, 0])
    rules = [r for r in ["R1_nearest", "R2_topmain", "R3_maxMAF"]
             if r in set(Q2.rule)]
    x = np.arange(len(rules)); w = .8 / len(SCHEMES)
    for k, sc in enumerate(SCHEMES):
        vals = []
        for r in rules:
            g = Q2[(Q2.rule == r) & (Q2.scheme == sc)]
            vals.append(float(g.F.iloc[0]) if len(g) else np.nan)
        b = ax.bar(x + (k - len(SCHEMES) / 2 + .5) * w, vals, w,
                   label=SCHEME_SHORT[sc], color=SCOL[sc],
                   edgecolor="black", linewidth=.4)
        for xi, v, rr in zip(x + (k - len(SCHEMES) / 2 + .5) * w, vals, rules):
            if np.isfinite(v):
                pv = Q2[(Q2.rule == rr) & (Q2.scheme == sc)].p
                star = "*" if len(pv) and float(pv.iloc[0]) < a.alpha else ""
                ax.annotate(f"{v:.1f}{star}", (xi, v), ha="center",
                            va="bottom", fontsize=7,
                            fontweight="bold" if star else "normal")
    # reference line for the F critical value (df1=1, alpha=0.05, about 3.89)
    ax.axhline(3.89, color="grey", ls=":", lw=1.2)
    ax.annotate("F ≈ 3.9 (p = 0.05)", xy=(len(rules) - .5, 3.89),
                xytext=(2, 3), textcoords="offset points", fontsize=7.5,
                color="grey")
    ax.set_xticks(x)
    ax.set_xticklabels([RULE_SHORT.get(r, r) for r in rules], fontsize=8.5)
    ax.set_ylabel("Interaction $F$ ($FRI \\times FLC$)")
    ax.set_title("(a) The interaction signal depends on how the loci were chosen",
                 fontsize=10.5, loc="left")
    ax.legend(title="correction", fontsize=7.5, title_fontsize=8, frameon=False,
              ncol=2)
    ax.grid(axis="y", alpha=.3)
    ax.spines[["top", "right"]].set_visible(False)

    # ---------- (b) main effects: FRI against FLC ----------
    ax = fig.add_subplot(gs[0, 1])
    rule_b = "R2_topmain" if "R2_topmain" in set(Q3.rule) else rules[0]
    sub = Q3[Q3.rule == rule_b]
    genes = [g for g in ["FRI", "FLC"] if g in set(sub.gene)]
    xs = np.arange(len(SCHEMES))
    for gi, gene in enumerate(genes):
        vals = []
        for sc in SCHEMES:
            r = sub[(sub.gene == gene) & (sub.scheme == sc)]
            vals.append(float(r.F.iloc[0]) if len(r) else np.nan)
        ax.plot(xs, vals, "o-", lw=2, ms=7,
                color="#C44E52" if gene == "FRI" else "#4C72B0", label=gene)
        for xi, v in zip(xs, vals):
            if np.isfinite(v):
                ax.annotate(f"{v:.1f}", (xi, v), textcoords="offset points",
                            xytext=(0, 7), ha="center", fontsize=7.5)
    ax.axhline(3.89, color="grey", ls=":", lw=1.2)
    ax.set_xticks(xs)
    ax.set_xticklabels([SCHEME_SHORT[s] for s in SCHEMES], fontsize=9)
    ax.set_xlabel("Population-structure correction")
    ax.set_ylabel("Main-effect $F$")
    ax.set_title(f"(b) Main effects under correction  [{rule_b}]",
                 fontsize=10.5, loc="left")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=.3)
    ax.spines[["top", "right"]].set_visible(False)

    # ---------- (c) injection test ----------
    ax = fig.add_subplot(gs[1, 0])
    xs = Q4.injected_frac.values.astype(float)
    for sc in SCHEMES:
        col = f"rate_{sc}"
        if col not in Q4.columns:
            continue
        ax.plot(xs, Q4[col].values, "o-", lw=1.8, ms=6,
                color=SCOL[sc], label=SCHEME_SHORT[sc])
    ax.axhline(0.05, color="grey", ls=":", lw=1.2)
    ax.annotate("nominal 0.05", xy=(xs.max(), .05), xytext=(-4, 4),
                textcoords="offset points", ha="right", fontsize=7.5, color="grey")
    ax.axhline(0.80, color="grey", ls="--", lw=1.0, alpha=.7)
    ax.annotate("80% power", xy=(xs.min(), .80), xytext=(4, 4),
                textcoords="offset points", fontsize=7.5, color="grey")
    fpr = Q4[Q4.injected_frac == 0]
    if len(fpr):
        txt = "  ".join(f"{SCHEME_SHORT[s]} {float(fpr[f'rate_{s}'].iloc[0]):.2f}"
                        for s in SCHEMES if f"rate_{s}" in Q4.columns)
        ax.annotate(f"false positive rate at zero injection:\n{txt}",
                    xy=(.03, .97), xycoords="axes fraction", va="top",
                    fontsize=7.5, bbox=dict(boxstyle="round,pad=0.35",
                                            fc="#FFF6E5", ec="grey", lw=.6))
    n_rep = int(Q4.n_rep.iloc[0]) if "n_rep" in Q4.columns else None
    ax.set_xlabel("Injected interaction variance"
                  + (f"   ({n_rep} replicates per level)" if n_rep else ""))
    ax.set_ylabel("Detection rate")
    ax.set_ylim(-.03, 1.05)
    ax.set_title("(c) Injection test: structure does not fabricate interactions,\n"
                 "     and correction does not destroy real ones",
                 fontsize=10.5, loc="left")
    ax.legend(title="correction", fontsize=7.5, title_fontsize=8, frameon=False)
    ax.grid(alpha=.3)
    ax.spines[["top", "right"]].set_visible(False)

    # ---------- (d) replication across phenotypes ----------
    ax = fig.add_subplot(gs[1, 1])
    QR = QR.copy()
    QR["nl_none"] = -np.log10(QR.p_none.clip(lower=1e-300))
    QR["nl_grm"] = -np.log10(QR.p_GRM.clip(lower=1e-300))
    yy = np.arange(len(QR))
    ax.barh(yy - .2, QR.nl_none, .38, label="no correction",
            color=SCOL["C0_none"], edgecolor="black", linewidth=.4)
    ax.barh(yy + .2, QR.nl_grm, .38, label="GRM correction",
            color=SCOL["C2_GRMonly"], edgecolor="black", linewidth=.4)
    ax.axvline(-np.log10(a.alpha), color="grey", ls=":", lw=1.2)
    ax.annotate("p = 0.05", xy=(-np.log10(a.alpha), len(QR) - .4),
                xytext=(3, 0), textcoords="offset points", fontsize=7.5,
                color="grey")
    bonf = a.alpha / len(QR)
    ax.axvline(-np.log10(bonf), color="black", ls="--", lw=1.1)
    ax.annotate(f"Bonferroni ({len(QR)} phenotypes)", xy=(-np.log10(bonf), 0),
                xytext=(3, -2), textcoords="offset points", fontsize=7.5)
    ax.set_yticks(yy)
    ax.set_yticklabels([f"{p} (n={int(n)})" for p, n in
                        zip(QR.phenotype, QR.n)], fontsize=8.5)
    ax.set_xlabel("$-\\log_{10}(p)$ for the interaction")
    ax.set_title("(d) The interaction does not replicate across flowering traits",
                 fontsize=10.5, loc="left")
    ax.legend(fontsize=8, frameon=False)
    ax.grid(axis="x", alpha=.3)
    ax.spines[["top", "right"]].set_visible(False)
    ax.invert_yaxis()

    fp = os.path.join(a.out, "Fig_arabidopsis.png")
    fig.savefig(fp, dpi=300, bbox_inches="tight")
    plt.close(fig)
    say(f"-> {fp}")

    # ---------- appendix: the confounding table ----------
    QC.to_csv(os.path.join(a.out, "Table_confounding_R2.csv"), index=False)

    say("\n" + "=" * 76)
    say("Numbers ready to quote in the Results")
    say("=" * 76)
    for r in rules:
        g0 = Q2[(Q2.rule == r) & (Q2.scheme == "C0_none")]
        g2 = Q2[(Q2.rule == r) & (Q2.scheme == "C2_GRMonly")]
        if len(g0):
            say(f"  {r:<12} uncorrected F={float(g0.F.iloc[0]):7.2f} "
                f"p={float(g0.p.iloc[0]):.3g}"
                + (f"   GRM only F={float(g2.F.iloc[0]):6.2f} "
                   f"p={float(g2.p.iloc[0]):.3g}" if len(g2) else ""))
    say("")
    for r in QC.itertuples(index=False):
        say(f"  {r.rule:<12} R2 of the interaction vector on PCs={r.R2_interaction_on_PC:.3f}"
            f"  locus A={r.R2_locusA_on_PC:.3f}  locus B={r.R2_locusB_on_PC:.3f}")
    if len(fpr):
        say(f"\n  false-positive rate (zero injection): " + ", ".join(
            f"{SCHEME_SHORT[s]}={float(fpr[f'rate_{s}'].iloc[0]):.2f}"
            for s in SCHEMES if f"rate_{s}" in Q4.columns))
    say(f"  Bonferroni across phenotypes: smallest p_GRM x {len(QR)} = "
        f"{QR.p_GRM.min() * len(QR):.3f}")


if __name__ == "__main__":
    main()
