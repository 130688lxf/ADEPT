#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 make_screening_figures.py -- Fig 3 and Table 3: which effect is usable in
                              which trait
================================================================================

 Answers the upstream question behind trait-wise routing: for each trait, does
 dominance or epistasis raise held-out prediction often enough to be used? The
 allocation figures say what was used; this one says why.

 Inputs (column names are auto-detected and reported, never guessed silently):
   pig     <pig_dir>/pig_bias_<trait>/diffuse_delta_pcc.csv     epistasis, A+D+K vs A+D
           <pig_dir>/pig_bias_<trait>/dominance_delta_pcc.csv   BOTH modules; the
                                                                 `comparison` column
                                                                 separates them
   mussel  files matched by --mussel_glob, either per-split or pre-aggregated

 Outputs:
   Table_screening.csv    one row per trait x module x model, with role and verdict
   Fig3_screening.png     mean dPCC and positive fraction, pig and mussel
   screening_summary.txt

 ------------------------------------------------------------------------------
 THREE RULES THIS SCRIPT ENFORCES, EACH OF WHICH WAS ONCE VIOLATED
 ------------------------------------------------------------------------------
 (1) THE THRESHOLD IS NOT A CONSTANT ACROSS SPLIT COUNTS. A positive fraction
     of 0.80 is a one-sided binomial probability of 0.0059 over 20 splits but
     0.19 over 5. Applying one number to both silently promoted five-split
     diagnostic kernels to "usable". Rows are therefore classified first:

       primary      n_splits == --n_splits and the model is a core model
                    -> judged against the threshold
       sensitivity  n_splits differs from --n_splits (e.g. the 100-split SH
                    dominance run) -> reported, never judged
       diagnostic   a secondary kernel fitted over fewer splits by design
                    -> reported, never judged

     If the input already carries a `role` column (the output of the
     harmonisation step) that column wins, and this script only checks it.

 (2) A FILE MAY HOLD MORE THAN ONE MODULE. dominance_delta_pcc.csv contains
     both `dominance: A+D vs A` and `epistasis: ... vs A+D` rows. Grouping by
     file name rather than by the `comparison` column once averaged the two
     together and made dominance look larger than it is. Module assignment
     comes from the comparison/model column only.

 (3) A PRE-AGGREGATED FILE'S pos_frac IS READ, NEVER RECOMPUTED. Recomputing
     it from a single aggregated row collapses it to "is this row positive",
     which can only be 0 or 1, and turns 1e-7 noise into a usable verdict.

     python make_screening_figures.py \
        --pig_dir  path/to/pig_results \
        --mussel_glob "path/to/mussel_out/*screen*.csv" \
        --n_splits 20 --threshold 0.80 --out screening_figs
================================================================================
"""
from __future__ import annotations

import argparse
import glob
import os
import re
from math import comb

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from trait_names import pretty as _pretty, label as _tlabel, \
        sort_key as _tsort, EXCLUDED_BY_DEFAULT
except ImportError:                                   # pragma: no cover
    def _pretty(t, sp=None): return str(t)
    def _tlabel(t, sp=None, with_species=True):
        return f"{str(sp)[:3]}:{t}" if with_species and sp else str(t)
    def _tsort(t, sp=None): return (1, str(t))
    EXCLUDED_BY_DEFAULT = ["phe9", "phe10", "phe11", "phe12", "phe13"]
    print("[note] trait_names.py not importable; raw codes will be used")

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "Liberation Serif",
                              "DejaVu Serif"]
plt.rcParams["mathtext.fontset"] = "stix"

# Models fitted over the full number of splits. Everything else is a secondary
# diagnostic run over fewer splits by design and must never be judged.
CORE_MODELS = {"A", "A+D", "A+D+MULT_T", "A+D+HAD", "MULT_T", "HAD",
               "A+MULT_T", "A+HAD"}

DCOLS = ["d_epi", "d_dom", "delta_pcc", "dPCC", "d_vs_AD", "d_vs_AGG",
         "d", "delta"]


def say(*a):
    print(*a, flush=True)


def nominal_p(pos_frac, n_splits):
    """One-sided binomial probability of at least this many positive splits
    under p = 0.5. Descriptive only: the splits resample one dataset and are
    not independent, so this is a yardstick for comparing split counts, not a
    significance test."""
    if not np.isfinite(pos_frac) or not n_splits:
        return np.nan
    k = int(round(pos_frac * n_splits))
    return sum(comb(n_splits, x) for x in range(k, n_splits + 1)) / 2 ** n_splits


def normalise_model(text):
    """'epistasis: A+D+MULT_t vs A+D' -> 'A+D+MULT_T'.

    The comparison column holds the whole nested-model statement. Matching it
    against CORE_MODELS verbatim would classify every real core model as a
    secondary diagnostic, so the tested model (the part before ' vs ') is
    extracted first.
    """
    t = str(text).strip()
    if ":" in t:
        t = t.split(":", 1)[1]
    if " vs " in t:
        t = t.split(" vs ", 1)[0]
    return t.upper().replace(" ", "")


def classify(row, n_splits_primary):
    """primary / sensitivity / diagnostic. An explicit role column wins."""
    if row.get("role") and str(row["role"]).strip():
        return str(row["role"]).strip()
    ns = row.get("n_splits")
    model = normalise_model(row.get("model", ""))
    if ns is None or not np.isfinite(ns):
        return "unknown"
    if int(ns) < n_splits_primary:
        return "diagnostic"
    if int(ns) > n_splits_primary:
        return "sensitivity"
    if model and model not in CORE_MODELS:
        return "diagnostic"
    return "primary"


def module_from_comparison(text, fallback):
    """Module comes from the comparison/model string, never from the file
    name -- see rule (2) in the header."""
    t = str(text).lower()
    if "dominance" in t:
        return "dominance"
    if "epistasis" in t or "mult" in t or "had" in t:
        return "epistasis"
    return fallback


def pick_dcol(df, prefer=None):
    if prefer and prefer in df.columns:
        return prefer
    for c in DCOLS:
        if c in df.columns:
            return c
    num = [c for c in df.columns
           if pd.api.types.is_numeric_dtype(df[c])
           and df[c].abs().max() < 1.0
           and c.lower() not in ("seed", "fold", "n_splits", "pos_frac")]
    return num[0] if num else None


def summarise_per_split(df, dcol, trait, module, model, species, src):
    """Per-split format: one row per split, pos_frac computed here."""
    d = pd.to_numeric(df[dcol], errors="coerce").dropna()
    if not len(d):
        return None
    return dict(species=species, trait=_pretty(trait, species),
                trait_raw=str(trait), module=module, model=model,
                n_splits=len(d), mean_dPCC=float(d.mean()),
                sd_dPCC=float(d.std(ddof=1)) if len(d) > 1 else 0.0,
                pos_frac=float((d > 0).mean()),
                source=os.path.basename(src), fmt="per-split",
                role=None)


def take_aggregated(row, trait, module, species, src, n_splits_hint):
    """Pre-aggregated format: dPCC / sd / pos_frac are already in the file and
    are read as they stand -- see rule (3)."""
    ns = row.get("n_seeds", row.get("n_splits", np.nan))
    ns = int(ns) if pd.notna(ns) else n_splits_hint
    return dict(species=species, trait=_pretty(trait, species),
                trait_raw=str(trait), module=module,
                model=str(row.get("model", "")),
                n_splits=ns,
                mean_dPCC=float(row["dPCC"]),
                sd_dPCC=float(row.get("sd", np.nan)),
                pos_frac=float(row["pos_frac"]),
                derived=(bool(row["derived"]) if "derived" in row.index
                         else None),
                source=os.path.basename(src), fmt="aggregated",
                role=(str(row["role"]) if "role" in row.index
                      and pd.notna(row.get("role")) else None))


def read_screen_file(path, module_fallback, species, n_splits_hint):
    df = pd.read_csv(path)
    tcol = next((c for c in ("trait", "phenotype", "phe") if c in df.columns),
                None)
    ccol = next((c for c in ("comparison", "model") if c in df.columns), None)
    out = []

    if "pos_frac" in df.columns and "dPCC" in df.columns:
        if tcol is None:
            say(f"  [skip] {os.path.basename(path)}: aggregated but has no "
                f"trait column (columns: {list(df.columns)})")
            return out
        for _, r in df.iterrows():
            mod = module_from_comparison(r.get(ccol, ""), module_fallback) \
                if ccol else module_fallback
            out.append(take_aggregated(r, r[tcol], mod, species, path,
                                       n_splits_hint))
        return out

    dcol = pick_dcol(df, "d_epi" if module_fallback == "epistasis" else "d_dom")
    if dcol is None:
        say(f"  [skip] {os.path.basename(path)}: no dPCC column recognised "
            f"(columns: {list(df.columns)})")
        return out

    group_cols = [c for c in (tcol, ccol) if c]
    if group_cols:
        for key, g in df.groupby(group_cols):
            key = key if isinstance(key, tuple) else (key,)
            trait = key[0] if tcol else os.path.basename(path)
            comp = key[-1] if ccol else ""
            mod = module_from_comparison(comp, module_fallback)
            r = summarise_per_split(g, dcol, str(trait), mod, str(comp),
                                    species, path)
            if r:
                out.append(r)
    else:
        m = re.findall(r"(phe\d+)", os.path.basename(path))
        r = summarise_per_split(df, dcol,
                                m[0] if m else os.path.basename(path),
                                module_fallback, "", species, path)
        if r:
            out.append(r)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pig_dir", default=None,
                    help="directory containing pig_bias_<trait>/")
    ap.add_argument("--pig_traits", nargs="+",
                    default=["bf", "lmd", "lmp", "rtn"])
    ap.add_argument("--mussel_glob", default=None,
                    help='e.g. "mussel_out/*screen*.csv"')
    ap.add_argument("--threshold", type=float, default=0.80,
                    help="positive-fraction threshold; fixed before analysis")
    ap.add_argument("--n_splits", type=int, default=20,
                    help="the primary number of splits. Rows with a different "
                         "count are labelled sensitivity or diagnostic and are "
                         "NOT judged against the threshold.")
    ap.add_argument("--exclude_traits", nargs="*", default=None,
                    help="default: the mussel colour traits not analysed here")
    ap.add_argument("--only_traits", nargs="*", default=None)
    ap.add_argument("--out", default="screening_figs")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    exclude = (a.exclude_traits if a.exclude_traits is not None
               else list(EXCLUDED_BY_DEFAULT))

    recs = []

    # ---------------- pig ----------------
    if a.pig_dir:
        for t in a.pig_traits:
            d = os.path.join(a.pig_dir, f"pig_bias_{t}")
            if not os.path.isdir(d):
                say(f"[pig] {d} not found, skipping {t}")
                continue
            for fn, fallback in [("diffuse_delta_pcc.csv", "epistasis"),
                                 ("dominance_delta_pcc.csv", "dominance"),
                                 ("dominance_delta_pcc_20.csv", "dominance")]:
                p = os.path.join(d, fn)
                if not (os.path.exists(p) and os.path.getsize(p)):
                    continue
                got = read_screen_file(p, fallback, "pig", a.n_splits)
                for r in got:
                    r["trait"] = _pretty(t, "pig")
                    r["trait_raw"] = t
                recs.extend(got)
                say(f"[pig] {t}: {len(got)} rows from {fn}")

    # ---------------- mussel ----------------
    if a.mussel_glob:
        files = sorted(glob.glob(a.mussel_glob))
        if not files:
            say(f"[mussel] {a.mussel_glob} matched nothing")
        for p_ in files:
            base = os.path.basename(p_)
            fallback = ("epistasis" if "epistasis" in base or "diffuse" in base
                        else "dominance" if "dominance" in base else None)
            if fallback is None:
                continue
            got = read_screen_file(p_, fallback, "mussel", a.n_splits)
            recs.extend(got)
            say(f"[mussel] {base}: {len(got)} rows")

    if not recs:
        raise SystemExit(
            "No screening results were read. Check that --pig_dir contains "
            "pig_bias_<trait>/ and that --mussel_glob points at the mussel "
            "*screen*.csv files. Column names encountered are printed above.")

    S = pd.DataFrame(recs)

    def _keep(row):
        raw, pr = str(row.get("trait_raw", row["trait"])), str(row["trait"])
        if a.only_traits:
            return raw in a.only_traits or pr in a.only_traits
        return not (raw in set(exclude) or pr in set(exclude))

    n0 = len(S)
    S = S[S.apply(_keep, axis=1)].reset_index(drop=True)
    if len(S) < n0:
        say(f"\n[filter] dropped {n0 - len(S)} rows for excluded traits")

    # ---- role, then verdict; only primary rows are judged ----
    S["role"] = S.apply(lambda r: classify(r, a.n_splits), axis=1)
    S["nominal_one_sided_p"] = [nominal_p(pf, int(ns))
                                for pf, ns in zip(S.pos_frac, S.n_splits)]
    S["verdict"] = np.where(
        S.role != "primary",
        np.where(S.role == "sensitivity", "sensitivity only (not judged)",
                 "diagnostic (not judged)"),
        np.where((S.mean_dPCC > 0) & (S.pos_frac >= a.threshold),
                 "usable", "not usable"))

    S = S.sort_values(
        by=["species", "module", "trait"],
        key=lambda col: col.map(lambda v: v) if col.name != "trait"
        else col.map(lambda t: _tsort(t))).reset_index(drop=True)

    p = os.path.join(a.out, "Table_screening.csv")
    S.to_csv(p, index=False)
    say(f"\n-> {p}")
    show = ["species", "trait", "module", "model", "n_splits", "mean_dPCC",
            "pos_frac", "nominal_one_sided_p", "role", "verdict"]
    say("\n" + S[[c for c in show if c in S.columns]]
        .round(5).to_string(index=False))

    n_role = S.role.value_counts().to_dict()
    say(f"\n[roles] {n_role}")
    if n_role.get("diagnostic"):
        say("        diagnostic rows were fitted over fewer splits by design "
            "and carry no verdict; at 5 splits a positive fraction of 0.80 "
            "has a nominal probability of 0.19.")
    if n_role.get("sensitivity"):
        say("        sensitivity rows use a different split count from the "
            "primary analysis and were not used for any decision.")

    # ---------------- figure: primary rows only ----------------
    Pm = S[S.role == "primary"].copy()
    if not len(Pm):
        say("\n[figure] no primary rows to plot")
        return
    Pm["mod2"] = np.where(Pm.module.str.startswith("dominance"),
                          "dominance", "epistasis")
    # within a module keep the single core model per trait (MULT_t over HAD:
    # they correlate at 0.999 and are the same kernel written two ways)
    def _rank(model):
        mu = normalise_model(model)
        return 0 if "MULT" in mu else 1 if "HAD" in mu else 2
    Pm["_r"] = Pm.model.map(_rank)
    Pm = (Pm.sort_values("_r")
            .groupby(["species", "trait", "mod2"], as_index=False).first())

    for sp, tag, fname in [("pig", "(a) Duroc pig", "Fig3a_screening_pig.png"),
                           ("mussel", "(b) Hyriopsis cumingii",
                            "Fig3b_screening_mussel.png")]:
        G = Pm[Pm.species == sp]
        if not len(G):
            continue
        traits = sorted(G.trait.unique(), key=lambda t: _tsort(t, sp))
        x = np.arange(len(traits))
        fig, axes = plt.subplots(2, 1, figsize=(max(6.5, .95 * len(traits)),
                                                6.6), sharex=True)
        for k, (mod, col) in enumerate([("dominance", "#55A868"),
                                        ("epistasis", "#C44E52")]):
            vals, pfs = [], []
            for t in traits:
                r = G[(G.trait == t) & (G.mod2 == mod)]
                vals.append(float(r.mean_dPCC.iloc[0]) if len(r) else np.nan)
                pfs.append(float(r.pos_frac.iloc[0]) if len(r) else np.nan)
            axes[0].bar(x + (k - .5) * .38, vals, .38, label=mod, color=col,
                        edgecolor="black", linewidth=.4)
            axes[1].bar(x + (k - .5) * .38, pfs, .38, color=col,
                        edgecolor="black", linewidth=.4)
        axes[0].axhline(0, color="black", ls="-", lw=.8)
        axes[0].set_ylabel(r"mean $\Delta$PCC")
        axes[0].set_title(f"{tag}: trait-wise screening over {a.n_splits} "
                          f"random 80/20 splits", fontsize=11, loc="left")
        axes[0].legend(frameon=False, fontsize=9)
        axes[1].axhline(a.threshold, color="black", ls="--", lw=1.0)
        axes[1].annotate(
            f"threshold {a.threshold:g} "
            f"({int(round(a.threshold * a.n_splits))} of {a.n_splits}, "
            f"nominal one-sided p = "
            f"{nominal_p(a.threshold, a.n_splits):.4f})",
            xy=(len(traits) - .5, a.threshold), xytext=(3, 4),
            textcoords="offset points", fontsize=7.5, ha="right")
        axes[1].set_ylabel("positive fraction across splits")
        axes[1].set_ylim(0, 1.05)
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(traits, rotation=20, ha="right", fontsize=9)
        for ax in axes:
            ax.grid(axis="y", alpha=.3)
            ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fp = os.path.join(a.out, fname)
        fig.savefig(fp, dpi=300, bbox_inches="tight")
        plt.close(fig)
        say(f"-> {fp}")

    with open(os.path.join(a.out, "screening_summary.txt"), "w") as f:
        f.write(S.round(5).to_string(index=False))
    say(f"-> {a.out}/screening_summary.txt")

    us = S[S.verdict == "usable"]
    say("\n" + "=" * 72)
    say(f"usable (primary rows only, mean dPCC > 0 and pos_frac >= "
        f"{a.threshold:g}): {len(us)}")
    if len(us):
        say(us[["species", "trait", "module", "model", "mean_dPCC",
                "pos_frac"]].round(5).to_string(index=False))
    say("Rows with role != primary appear in Table_screening.csv but were "
        "not judged and must not be quoted as accepted or rejected.")


if __name__ == "__main__":
    main()
