#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 collect_allocation_results.py -- gather the allocation results of every trait
                                  into the manuscript tables
================================================================================

 Reads <trait>_strategy_summary.csv from each trait directory (the column names
 are identical across species, so no field mapping is needed), together with
 usable_pairs.csv / pair_summary.csv, and writes:

   Table_allocation_full.csv     every trait x every arm, long format
   Table_allocation_main.csv     one row per trait, the arms spread across columns
   Table_pairs_inventory.csv     candidate pairs, retained pairs, pass rate
   allocation_summary.txt        the same content as printed to screen

 Directory convention:
   pig      pipe_<trait>/     <trait>_strategy_summary.csv, usable_pairs.csv
   mussel   alloc_<trait>/    <trait>_strategy_summary.csv
            pairval_<trait>/  usable_pairs.csv, pair_summary.csv

 A trait with no usable pair produces a single ADD row and fewer columns; a
 material whose dominance module was blocked produces no ADD+DOM row. Both are
 handled per column rather than by assuming a fixed column set, because both
 are ordinary outcomes of the system rather than missing data.

   python collect_allocation_results.py --species pig \
       --alloc_glob "pipe_*" --pairs_glob "pipe_*" --out pig_tables

   python collect_allocation_results.py --species mussel \
       --alloc_glob "alloc_phe*" --pairs_glob "pairval_phe*" --out mussel_tables

   python collect_allocation_results.py --merge pig_tables mussel_tables \
       --out final_tables
================================================================================
"""
from __future__ import annotations

import argparse
import glob
import os
import re

import numpy as np
import pandas as pd

STRATEGY_ORDER = ["ADD", "ADD+DOM", "FULL_naive", "two_stage",
                  "weighted", "add_floor"]


def say(*a):
    print(*a, flush=True)


def trait_from_dir(d):
    return re.sub(r"^(alloc_|pairval_|pipe_)", "", os.path.basename(d.rstrip("/")))


def collect_alloc(pattern, species):
    rows = []
    for d in sorted(glob.glob(pattern)):
        if not os.path.isdir(d):
            continue
        t = trait_from_dir(d)
        f = os.path.join(d, f"{t}_strategy_summary.csv")
        if not os.path.exists(f):
            cand = glob.glob(os.path.join(d, "*_strategy_summary.csv"))
            if not cand:
                say(f"  [skip] {d}: no *_strategy_summary.csv")
                continue
            f = cand[0]
        df = pd.read_csv(f)
        df.insert(0, "species", species)
        if "trait" not in df.columns:
            df.insert(1, "trait", t)
        rows.append(df)
        say(f"  [read] {f}  ({len(df)} strategies)")
    return pd.concat(rows, ignore_index=True) if rows else None


def collect_pairs(pattern, species):
    rows = []
    for d in sorted(glob.glob(pattern)):
        if not os.path.isdir(d):
            continue
        t = trait_from_dir(d)
        ps = os.path.join(d, "pair_summary.csv")
        up = os.path.join(d, "usable_pairs.csv")
        n_cand = n_pass = np.nan
        if os.path.exists(ps) and os.path.getsize(ps):
            S = pd.read_csv(ps)
            n_cand = len(S)
        if os.path.exists(up) and os.path.getsize(up):
            U = pd.read_csv(up)
            n_pass = len(U)
        elif not np.isnan(n_cand):
            n_pass = 0
        if np.isnan(n_cand) and np.isnan(n_pass):
            continue
        rows.append(dict(species=species, trait=t,
                         n_candidate_pairs=n_cand, n_usable_pairs=n_pass,
                         pass_rate=(n_pass / n_cand
                                    if n_cand and not np.isnan(n_cand) else np.nan)))
        say(f"  [pairs] {t}: {n_pass:.0f}/{n_cand:.0f}")
    return pd.DataFrame(rows) if rows else None


def make_main_table(full):
    """One row per trait, the six arms spread across columns."""
    out = []
    for (sp, t), g in full.groupby(["species", "trait"], sort=False):
        g = g.set_index("strategy")
        rec = {"species": sp, "trait": t}
        rec["n_pairs_used"] = (int(g["n_pairs_used"].iloc[0])
                               if "n_pairs_used" in g else np.nan)
        if "dominance_blocked" in g.columns:
            rec["dominance_blocked"] = bool(g["dominance_blocked"].iloc[0])
        for s in STRATEGY_ORDER:
            if s not in g.index:
                continue
            r = g.loc[s]
            rec[f"{s}|add_ret"] = round(float(r.add_retained_vs_ADD), 3)
            rec[f"{s}|gv_full"] = round(float(r.mean_gv_full), 4)
            rec[f"{s}|changed"] = int(r.crosses_changed)
        out.append(rec)
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--species", default=None)
    ap.add_argument("--alloc_glob", default=None)
    ap.add_argument("--pairs_glob", default=None)
    ap.add_argument("--merge", nargs="+", default=None,
                    help="merge several previously generated output directories into the final tables")
    ap.add_argument("--out", default="alloc_tables")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    if a.merge:
        fulls, invs = [], []
        for d in a.merge:
            f = os.path.join(d, "Table_allocation_full.csv")
            i = os.path.join(d, "Table_pairs_inventory.csv")
            if os.path.exists(f):
                fulls.append(pd.read_csv(f)); say(f"  [merge] {f}")
            if os.path.exists(i):
                invs.append(pd.read_csv(i))
        if not fulls:
            raise SystemExit("no Table_allocation_full.csv in the --merge directories")
        full = pd.concat(fulls, ignore_index=True)
        inv = pd.concat(invs, ignore_index=True) if invs else None
    else:
        if not (a.species and a.alloc_glob):
            raise SystemExit("--species and --alloc_glob are required, or use --merge")
        say(f"[allocation] scanning {a.alloc_glob}")
        full = collect_alloc(a.alloc_glob, a.species)
        if full is None:
            raise SystemExit("no strategy_summary.csv was read")
        inv = None
        if a.pairs_glob:
            say(f"[pairs] scanning {a.pairs_glob}")
            inv = collect_pairs(a.pairs_glob, a.species)

    full.to_csv(os.path.join(a.out, "Table_allocation_full.csv"), index=False)
    main_t = make_main_table(full)
    main_t.to_csv(os.path.join(a.out, "Table_allocation_main.csv"), index=False)
    if inv is not None:
        inv.to_csv(os.path.join(a.out, "Table_pairs_inventory.csv"), index=False)

    lines = []
    def P(s=""):
        lines.append(s); say(s)

    P("\n" + "=" * 88)
    P("Per trait x strategy: additive retained / total gain / crosses changed")
    P("=" * 88)
    for (sp, t), g in full.groupby(["species", "trait"], sort=False):
        npu = int(g["n_pairs_used"].iloc[0]) if "n_pairs_used" in g else 0
        evf = float(g["epi_var_frac"].iloc[0]) if "epi_var_frac" in g else float("nan")
        blocked = ("dominance_blocked" in g.columns
                   and bool(g["dominance_blocked"].iloc[0]))
        tag = ("  <- no usable pair; the plan degrades to purely additive"
               if len(g) == 1 else "")
        if blocked:
            tag += "  [dominance module blocked by the material]"
        P(f"\n{sp}  {t}   (using {npu} pairs, epi_var_frac={evf:.4f}){tag}")
        g2 = g.set_index("strategy").reindex(
            [s for s in STRATEGY_ORDER if s in set(g.strategy)])
        for s, r in g2.iterrows():
            # A trait with no usable pair emits a single ADD row with fewer
            # columns (no mean_gv_dom / epi / F), so every column is read
            # defensively rather than assuming one fixed column set.
            def gv(col, default=float("nan")):
                v = r.get(col, default) if hasattr(r, "get") else default
                try:
                    return float(v)
                except (TypeError, ValueError):
                    return default
            P(f"   {s:<12} additive retained {gv('add_retained_vs_ADD'):6.3f}   "
              f"total gain {gv('mean_gv_full'):9.4f}   "
              f"changed {int(gv('crosses_changed', 0)):3d}   "
              f"overlap {gv('overlap_with_ADD'):.2f}   F {gv('mean_F'):.4f}")

    if inv is not None and len(inv):
        P("\n" + "=" * 88)
        P("Validated epistatic pairs, per trait")
        P("=" * 88)
        for _, r in inv.iterrows():
            P(f"   {r.species:<7} {r.trait:<8} {r.n_usable_pairs:>3.0f} / "
              f"{r.n_candidate_pairs:>4.0f}  = {r.pass_rate*100:5.2f}%")

    # Strategy behaviour across traits -- the most quotable line here
    P("\n" + "=" * 88)
    P("Across traits: median additive merit retained and median crosses changed")
    P("=" * 88)
    for s in STRATEGY_ORDER:
        g = full[full.strategy == s]
        if not len(g):
            continue
        P(f"   {s:<12} median additive retained {g.add_retained_vs_ADD.median():6.3f}"
          f"   median crosses changed {g.crosses_changed.median():5.1f}"
          f"   (n={len(g)} traits)")

    with open(os.path.join(a.out, "allocation_summary.txt"), "w") as f:
        f.write("\n".join(lines))
    P(f"\n-> {a.out}/Table_allocation_full.csv")
    P(f"-> {a.out}/Table_allocation_main.csv")
    if inv is not None:
        P(f"-> {a.out}/Table_pairs_inventory.csv")
    P(f"-> {a.out}/allocation_summary.txt")


if __name__ == "__main__":
    main()
