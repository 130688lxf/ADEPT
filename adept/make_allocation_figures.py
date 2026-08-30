#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 make_allocation_figures.py -- the allocation figures of the manuscript
================================================================================

 Reads the output of collect_allocation_results.py:
   Table_allocation_full.csv     one row per trait x strategy
   Table_pairs_inventory.csv     candidate and usable pairs per trait (optional)

 and writes three figures.

 Fig_alloc_a  strategy trade-off: x = additive retained relative to ADD,
              y = total gain relative to ADD. One point is one (trait,
              strategy) pair and ADD always sits at (1, 1). The upper-right
              quadrant is where additive merit was kept AND gain was obtained.
              Both axes are normalised against that trait's own ADD arm, which
              is what makes traits comparable: the absolute gv_add of pig LMD
              (2.27) and ELMP (0.82) cannot otherwise be plotted together.

 Fig_alloc_b  usable pairs per trait (bars), annotated with whether the trait
              fell back to a purely additive plan. This is the most direct
              evidence for trait-wise routing.

 Fig_alloc_c  decision-change heatmap: trait x strategy, value =
              crosses_changed, i.e. how many mating decisions the non-additive
              components actually altered.

     python make_allocation_figures.py --tables final_tables --out alloc_figures
================================================================================
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

# ---- trait-code to published-abbreviation map, used in every figure ----
try:
    from trait_names import pretty as _pretty, label as _tlabel, sort_key as _tsort
except ImportError:
    def _pretty(t, sp=None): return str(t)
    def _tlabel(t, sp=None, with_species=True):
        return f"{str(sp)[:3]}:{t}" if with_species and sp else str(t)
    def _tsort(t, sp=None): return (1, str(t))
    print("[note] trait_names.py not found; internal codes (phe1 etc.) will be used")


import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "Liberation Serif", "DejaVu Serif"]
plt.rcParams["mathtext.fontset"] = "stix"

ORDER = ["ADD", "ADD+DOM", "FULL_naive", "two_stage", "weighted", "add_floor"]
COLORS = {"ADD": "#555555", "ADD+DOM": "#4C72B0", "FULL_naive": "#C44E52",
          "two_stage": "#55A868", "weighted": "#DD8452", "add_floor": "#8172B3"}
MARKERS = {"pig": "o", "mussel": "^"}


def say(*a):
    print(*a, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", required=True,
                    help="output directory of collect_allocation_results.py")
    ap.add_argument("--out", default="alloc_figures")
    ap.add_argument("--dpi", type=int, default=300)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    f_full = os.path.join(a.tables, "Table_allocation_full.csv")
    if not os.path.exists(f_full):
        raise SystemExit(f"not found: {f_full}")
    F = pd.read_csv(f_full)
    inv_p = os.path.join(a.tables, "Table_pairs_inventory.csv")
    INV = pd.read_csv(inv_p) if os.path.exists(inv_p) else None
    say(f"[read] {len(F)} rows, {F.trait.nunique()} trait(s)")

    # ---------- normalise against ADD, the precondition for comparing traits ----------
    recs = []
    for (sp, t), g in F.groupby(["species", "trait"], sort=False):
        add = g[g.strategy == "ADD"]
        if not len(add):
            say(f"  [skipped] {sp}/{t}: no ADD baseline row, cannot normalise")
            continue
        base_full = float(add.mean_gv_full.iloc[0])
        if not np.isfinite(base_full) or abs(base_full) < 1e-9:
            say(f"  [skipped] {sp}/{t}: ADD mean_gv_full={base_full}, unusable as denominator")
            continue
        for _, r in g.iterrows():
            recs.append(dict(
                species=sp, trait=_pretty(t, sp), strategy=r.strategy,
                add_ret=float(r.add_retained_vs_ADD),
                gain_rel=float(r.mean_gv_full) / base_full,
                changed=int(r.crosses_changed),
                n_pairs=int(r.get("n_pairs_used", 0) or 0),
                n_strategies=len(g)))
    D = pd.DataFrame(recs)
    if not len(D):
        raise SystemExit("nothing left after normalisation")
    D.to_csv(os.path.join(a.out, "allocation_normalised.csv"), index=False)

    # ================= Fig a: strategy trade-off =================
    fig, ax = plt.subplots(figsize=(8.5, 5.4))

    ax.axhline(1.0, color="grey", lw=.8, ls="--", zorder=1)
    ax.axvline(1.0, color="grey", lw=.8, ls="--", zorder=1)
    for s in ORDER:
        sub = D[D.strategy == s]
        if not len(sub):
            continue
        for sp, sub2 in sub.groupby("species"):
            ax.scatter(sub2.add_ret, sub2.gain_rel, s=64,
                       marker=MARKERS.get(sp, "s"), color=COLORS.get(s, "k"),
                       edgecolor="black", linewidth=.5, alpha=.85, zorder=3,
                       label=s if sp == sub.species.iloc[0] else None)
    ax.set_xlabel("Additive merit retained (relative to ADD plan)")
    ax.set_ylabel("Total genetic gain (relative to ADD plan)")
    ax.set_title("Strategy trade-off: additive merit versus total gain")
    ax.grid(alpha=.25)
    ax.spines[["top", "right"]].set_visible(False)

    h, l = ax.get_legend_handles_labels()
    seen, hh, ll = set(), [], []
    for hi, li in zip(h, l):
        if li not in seen:
            seen.add(li); hh.append(hi); ll.append(li)
    leg1 = ax.legend(hh, ll, title="Strategy", fontsize=8, title_fontsize=8,
                     loc="upper left", bbox_to_anchor=(1.0, 1.0),
                     borderaxespad=0.2, frameon=False)
    ax.add_artist(leg1)

    from matplotlib.lines import Line2D
    shape_h = [Line2D([], [], marker=MARKERS[k], color="none",
                      markerfacecolor="lightgrey", markeredgecolor="black",
                      markersize=8, label=k) for k in MARKERS if k in set(D.species)]
    if shape_h:
        ax.legend(handles=shape_h, title="Species", fontsize=8,
                  title_fontsize=8, loc="lower left",
                  bbox_to_anchor=(1.0, 0.0), borderaxespad=0.2, frameon=False)

    ax.text(.5, -.13, "right of the dashed line: no additive merit sacrificed",
            transform=ax.transAxes, ha="center", va="top", fontsize=7.5,
            style="italic", color="#444444")

    # axes placed by hand, leaving about 22% on the right for the legend
    fig.subplots_adjust(left=0.1, right=0.85, top=0.9, bottom=0.18)

    p = os.path.join(a.out, "Fig_alloc_a_tradeoff.png")
    fig.savefig(p, dpi=a.dpi, pad_inches=0.1)
    plt.close(fig)
    say(f"-> {p}")

    # ================= Fig b: usable pairs per trait =================
    per_t = (D.groupby(["species", "trait"])
               .agg(n_pairs=("n_pairs", "max"),
                    n_strategies=("n_strategies", "max"))
               .reset_index())
    if INV is not None:
        per_t = per_t.merge(
            INV[["species", "trait", "n_candidate_pairs", "n_usable_pairs"]],
            on=["species", "trait"], how="left")
        per_t["n_pairs"] = per_t["n_usable_pairs"].fillna(per_t["n_pairs"])
    per_t = per_t.sort_values(["species", "n_pairs"], ascending=[True, False])

    fig, ax = plt.subplots(figsize=(max(7, .55 * len(per_t)), 4.6))
    xs = np.arange(len(per_t))
    cols = ["#4C72B0" if s == "pig" else "#55A868" for s in per_t.species]
    ax.bar(xs, per_t.n_pairs, color=cols, edgecolor="black", linewidth=.5)
    for i, (_, r) in enumerate(per_t.iterrows()):
        if r.n_strategies == 1:
            ax.text(i, .3, "additive\nonly", ha="center", va="bottom",
                    fontsize=6.5, style="italic", color="#B22222")
        else:
            ax.text(i, r.n_pairs + .4, int(r.n_pairs), ha="center",
                    va="bottom", fontsize=7)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{r.trait}" for _, r in per_t.iterrows()],
                       rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Validated epistatic pairs used")
    ax.set_title("Per-trait outcome of pair validation")
    ax.grid(axis="y", alpha=.25)
    ax.spines[["top", "right"]].set_visible(False)
    hs = [plt.Rectangle((0, 0), 1, 1, fc=c, ec="black", lw=.5)
          for c in ["#4C72B0", "#55A868"]]
    ax.legend(hs, ["pig", "mussel"], fontsize=8, loc="upper left",
              bbox_to_anchor=(1.02, 1.0), borderaxespad=0, frameon=False)
    fig.tight_layout()
    p = os.path.join(a.out, "Fig_alloc_b_pairs_per_trait.png")
    fig.savefig(p, dpi=a.dpi, bbox_inches="tight"); plt.close(fig)
    say(f"-> {p}")

    # ================= Fig c: decision-change heatmap =================
    piv = D.pivot_table(index=["species", "trait"], columns="strategy",
                        values="changed", aggfunc="first")
    piv = piv.reindex(columns=[s for s in ORDER if s in piv.columns])
    fig, ax = plt.subplots(figsize=(1.1 * len(piv.columns) + 3.2,
                                    .34 * len(piv) + 2.2))
    M = piv.values.astype(float)
    im = ax.imshow(M, cmap="YlOrRd", aspect="auto", vmin=0,
                   vmax=np.nanmax(M) if np.isfinite(np.nanmax(M)) else 1)
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels(piv.columns, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(piv)))
    ax.set_yticklabels([_tlabel(t, sp).replace(":", " ") for sp, t in piv.index], fontsize=7.5)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if np.isfinite(v):
                ax.text(j, i, int(v), ha="center", va="center", fontsize=6.8,
                        color="white" if v > np.nanmax(M) * .6 else "black")
    ax.set_title("Crosses changed relative to the ADD plan (of 50)")
    fig.colorbar(im, ax=ax, shrink=.8, label="crosses changed")
    fig.tight_layout()
    p = os.path.join(a.out, "Fig_alloc_c_decision_change.png")
    fig.savefig(p, dpi=a.dpi, bbox_inches="tight"); plt.close(fig)
    say(f"-> {p}")

    # ---------- console summary, quotable in the manuscript ----------
    say("\n" + "=" * 74)
    say("Median performance of each strategy across traits")
    say("=" * 74)
    for s in ORDER:
        g = D[D.strategy == s]
        if not len(g):
            continue
        say(f"  {s:<12} additive retained {g.add_ret.median():6.3f}   "
            f"relative gain {g.gain_rel.median():6.3f}   "
            f"crosses changed {g.changed.median():5.1f}   (n={len(g)})")
    deg = per_t[per_t.n_strategies == 1]
    say(f"\nTraits that fell back to a purely additive plan: {len(deg)}/{len(per_t)}: "
        + (", ".join(f"{r.species}/{r.trait}" for _, r in deg.iterrows())
           if len(deg) else "none"))
    say(f"\n-> {a.out}/allocation_normalised.csv")


if __name__ == "__main__":
    main()