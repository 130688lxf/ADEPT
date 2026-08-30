#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
harmonise_screening.py
======================

Bring every screening result (pig + mussel, dominance + epistasis) onto a single,
common number of random splits, so that one positive-fraction threshold means the
same nominal significance level everywhere in the manuscript.

WHY THIS EXISTS
---------------
The delivered screening tables were produced at three different split counts
(pig epistasis 10, pig dominance 30, mussel 20).  A fixed pos_frac threshold of
0.80 corresponds to a different one-sided nominal p at each of those counts
(9/10 -> 0.011, 16/20 -> 0.0059, 24/30 -> 0.0007), so the same threshold is not
the same test.  This script re-derives every row at ONE split count.

WHAT IT DOES *NOT* DO
---------------------
It never refits a model.  Per-split delta-PCC values already exist on disk; this
script only subsets and re-aggregates them.  Where a file has FEWER splits than
requested it reports the shortfall explicitly and marks the row INCOMPLETE
instead of silently aggregating over whatever is available.

THE SUBSETTING RULE (state this in Methods)
-------------------------------------------
Splits are indexed by their random seed.  When more than N splits are available
the first N seeds *in ascending numeric order* are kept.  The seed is fixed
before the phenotype is seen and the ordering does not depend on the outcome, so
this is a pre-specified rule and not selection on the result.  The same seed set
is used for both arms of every contrast, so all comparisons remain paired.

USAGE
-----
  # step 1 -- ALWAYS run this first and read the output
  python harmonise_screening.py --inspect \
      --pig_root  /path/to/pig  --mussel_root /path/to/mussel_out

  # step 2 -- produce the table
  python harmonise_screening.py \
      --pig_root  /path/to/pig  --mussel_root /path/to/mussel_out \
      --n_splits 20 --posfrac_threshold 0.80 \
      --out Table_screening.csv

  # keep the 100-split phe4 dominance row as a sensitivity row instead of
  # dropping it (it is written with verdict 'sensitivity', never 'usable'):
  #     --keep_offcount_as_sensitivity

Author's note: every numeric column in the output carries the split count it was
computed from, so a reader can never be misled about which rows are comparable.
"""

from __future__ import annotations

import argparse
import glob
import gzip
import os
import re
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Trait naming                                                                 #
# --------------------------------------------------------------------------- #

MUSSEL_ABBREV: Dict[str, str] = {
    "phe1": "SL", "phe2": "TSH", "phe3": "AARRL", "phe4": "SH", "phe5": "ShW",
    "phe6": "BW", "phe7": "sW", "phe8": "SWR", "phe14": "dE", "phe15": "%PP",
}
# Traits deliberately excluded from the manuscript (no production relevance).
MUSSEL_DROP = {"phe9", "phe10", "phe11", "phe12", "phe13"}
# Derived traits: legitimate to analyse singly, but a non-linear transform of
# other traits can manufacture statistical epistasis, so they are flagged.
MUSSEL_DERIVED = {"phe8", "phe12", "phe13", "phe14"}

PIG_TRAITS = ("bf", "lmd", "lmp", "rtn")

# Models that carry a manuscript claim.  Everything else in the delta-PCC files
# is a SECONDARY DIAGNOSTIC model, run at a deliberately reduced number of splits
# under the model-tiering design (core models get the full split budget, the
# diagnostics get five).  Those rows are not deficient -- they were never meant
# to reach the common split count -- so they are labelled 'diagnostic' rather
# than 'INCOMPLETE', and they are excluded from the usable/not-usable verdict.
CORE_MODELS = ("A+D vs A", "A+D+MULT_t", "A+D+HAD")
_CORE: Tuple[str, ...] = CORE_MODELS

# --------------------------------------------------------------------------- #
# Tolerant column detection                                                    #
# --------------------------------------------------------------------------- #

SEED_KEYS = ("seed", "split", "rep", "replicate", "iter", "fold_seed", "n_seed")
DELTA_KEYS = ("delta_pcc", "dpcc", "d_pcc", "delta", "dPCC", "diff", "delta_acc")
MODEL_KEYS = ("model", "module", "contrast", "comparison", "arm")
TRAIT_KEYS = ("trait", "phenotype", "phe", "trait_raw")


def _find_col(df: pd.DataFrame, keys: Sequence[str]) -> Optional[str]:
    """Return the first column whose lowercased name matches or contains a key."""
    lower = {c.lower().strip(): c for c in df.columns}
    for k in keys:
        if k.lower() in lower:
            return lower[k.lower()]
    for k in keys:
        for lc, orig in lower.items():
            if k.lower() in lc:
                return orig
    return None


def _read_any(path: str) -> pd.DataFrame:
    """Read a csv/csv.gz, tolerating CRLF and stray whitespace in headers."""
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as fh:
        df = pd.read_csv(fh)
    df.columns = [str(c).strip() for c in df.columns]
    return df


# --------------------------------------------------------------------------- #
# Aggregation                                                                  #
# --------------------------------------------------------------------------- #


_CMP_RE = re.compile(r"^\s*(dominance|epistasis)\s*:\s*(.+?)\s*$", re.I)


def normalise_contrast(label: str, file_module: str) -> Tuple[str, str]:
    """
    Turn a contrast label into (module, model).

    The pig dominance file stores three different contrasts in one file --
    'dominance: A+D vs A' plus two 'epistasis: ... vs A+D' rows -- so the module
    must be taken from the label, not from the file name.  Aggregating across
    them (which an earlier summary did) mixes a dominance gain with two epistatic
    gains and inflates the apparent dominance effect.

    Model names are normalised so that the same contrast arriving from two
    different files collapses to one row: 'epistasis: A+D+HAD vs A+D' and the
    diffuse file's bare 'A+D+HAD' both become ('epistasis', 'A+D+HAD').
    """
    lab = str(label).strip()
    m = _CMP_RE.match(lab)
    if m:
        module, rest = m.group(1).lower(), m.group(2).strip()
    else:
        module, rest = file_module, lab
    if module == "epistasis":
        rest = re.sub(r"\s+vs\s+A\+D\s*$", "", rest, flags=re.I).strip()
    return module, rest


def aggregate(deltas: np.ndarray) -> Tuple[float, float, float]:
    """mean, sd (ddof=1), and fraction of splits with a strictly positive delta."""
    d = np.asarray(deltas, dtype=float)
    d = d[np.isfinite(d)]
    if d.size == 0:
        return float("nan"), float("nan"), float("nan")
    sd = float(np.std(d, ddof=1)) if d.size > 1 else float("nan")
    return float(np.mean(d)), sd, float(np.mean(d > 0.0))


def nominal_p(n: int, pos_frac: float) -> float:
    """
    One-sided binomial tail P(X >= k) with X ~ Bin(n, 0.5), k = round(pos_frac*n).

    This is the nominal level of the positive-fraction rule.  It is reported so
    that readers can see the threshold means the same thing at every split count.
    It is NOT a valid p-value for the underlying effect: repeated splits of the
    same individuals are not independent experiments.  Methods must say so.
    """
    if not np.isfinite(pos_frac) or n <= 0:
        return float("nan")
    from math import comb
    k = int(round(pos_frac * n))
    return float(sum(comb(n, i) for i in range(k, n + 1)) / (2.0 ** n))


def verdict_for(mean_d: float, pos_frac: float, n_used: int, n_req: int,
                thr: float, model: Optional[str] = None) -> str:
    if model is not None and _CORE and model not in _CORE:
        return "diagnostic"
    if n_used < n_req:
        return "INCOMPLETE"
    if not np.isfinite(mean_d) or not np.isfinite(pos_frac):
        return "not usable"
    return "usable" if (mean_d > 0.0 and pos_frac >= thr) else "not usable"


def _per_split_rows(path: str, module: str, species: str, n_splits: int,
                    thr: float, trait_map, derived_set, drop_set,
                    emit_sensitivity: bool) -> List[dict]:
    """
    Aggregate one per-split delta-PCC file at the target split count.

    Emits, for each (trait, model):
      * one 'primary'     row using the first n_splits seeds in ascending order;
      * one 'sensitivity' row using ALL available seeds, but only when more than
        n_splits exist.  The sensitivity row can never be labelled 'usable'.
    """
    out: List[dict] = []
    df = _read_any(path)
    c_seed = _find_col(df, SEED_KEYS)
    c_delta = _find_col(df, DELTA_KEYS)
    c_model = _find_col(df, MODEL_KEYS)
    c_trait = _find_col(df, TRAIT_KEYS)
    if c_seed is None or c_delta is None:
        print(f"[{species[:3]} ] cannot parse {path}: seed={c_seed} "
              f"delta={c_delta}. Run with --inspect.", file=sys.stderr)
        return out

    trait_groups = ([(str(t), g) for t, g in df.groupby(c_trait)]
                    if c_trait is not None else [("__all__", df)])

    for raw, gt in trait_groups:
        raw = raw.strip()
        if raw in drop_set:
            continue
        label = trait_map.get(raw, raw) if trait_map else raw
        if trait_map and raw not in trait_map:
            print(f"[{species[:3]} ] unknown trait {raw!r} in "
                  f"{os.path.basename(path)}; skipped", file=sys.stderr)
            continue

        model_groups = ([(str(m), g) for m, g in gt.groupby(c_model)]
                        if c_model is not None else [(module, gt)])
        for model_label, g in model_groups:
            g = g.dropna(subset=[c_seed, c_delta])
            seeds = sorted(pd.to_numeric(g[c_seed],
                                         errors="coerce").dropna().unique())
            if not seeds:
                continue
            seed_num = pd.to_numeric(g[c_seed], errors="coerce")

            mod_here, model_here = normalise_contrast(model_label, module)

            def _agg(keep_seeds, role):
                sub = g[seed_num.isin(keep_seeds)]
                per_seed = (sub.groupby(pd.to_numeric(sub[c_seed],
                                                      errors="coerce"))[c_delta]
                              .mean().values)
                m, sd, pf = aggregate(per_seed)
                n_used = int(len(per_seed))
                v = (verdict_for(m, pf, n_used, n_splits, thr, model_here)
                     if role == "primary" else "sensitivity")
                if v == "diagnostic":
                    role = "diagnostic"
                return dict(
                    species=species, trait=label, trait_raw=raw,
                    module=mod_here, model=model_here, role=role,
                    n_splits_available=int(len(seeds)), n_splits_used=n_used,
                    mean_dPCC=m, sd_dPCC=sd, pos_frac=pf,
                    nominal_p=nominal_p(n_used, pf),
                    derived=(raw in derived_set) if derived_set else False,
                    source=os.path.basename(path), fmt="per-split", verdict=v,
                )

            out.append(_agg(seeds[:n_splits], "primary"))
            if emit_sensitivity and len(seeds) > n_splits:
                out.append(_agg(seeds, "sensitivity"))
    return out


# --------------------------------------------------------------------------- #
# Pig: per-split files                                                         #
# --------------------------------------------------------------------------- #

PIG_FILES = {
    "epistasis": ("diffuse_delta_pcc.csv",),
    "dominance": ("dominance_delta_pcc.csv",),
}


def collect_pig(root: str, n_splits: int, thr: float,
                inspect: bool, emit_sensitivity: bool = True) -> List[dict]:
    rows: List[dict] = []
    dirs = sorted(glob.glob(os.path.join(root, "pig_bias_*"))) or \
           sorted(glob.glob(os.path.join(root, "**", "pig_bias_*"), recursive=True))
    if not dirs:
        print(f"[pig ] no pig_bias_* directories under {root}", file=sys.stderr)
        return rows

    for d in dirs:
        trait = os.path.basename(d).replace("pig_bias_", "")
        if trait not in PIG_TRAITS:
            print(f"[pig ] skipping {trait} (not in manuscript scope)")
            continue
        for module, fnames in PIG_FILES.items():
            for fname in fnames:
                path = os.path.join(d, fname)
                if not os.path.exists(path):
                    continue
                if inspect:
                    df = _read_any(path)
                    print(f"\n--- {path}")
                    print(f"    rows={len(df)}  columns={list(df.columns)}")
                    print(df.head(3).to_string(index=False))
                    continue
                got = _per_split_rows(path, module, "pig", n_splits, thr,
                                      None, set(), set(), emit_sensitivity)
                for r in got:              # pig files carry no trait column
                    if r["trait_raw"] == "__all__":
                        r["trait"] = r["trait_raw"] = trait
                    r["source"] = os.path.join(os.path.basename(d), fname)
                rows += got
    return rows


# --------------------------------------------------------------------------- #
# Mussel: already-aggregated files                                             #
# --------------------------------------------------------------------------- #

MUSSEL_PERSPLIT = {"dominance": "dominance_delta_pcc_phe*.csv",
                   "epistasis": "diffuse_delta_pcc_phe*.csv"}


def collect_mussel(root: str, n_splits: int, thr: float,
                   splits_default: int, overrides: Dict[str, int],
                   keep_offcount: bool, inspect: bool,
                   emit_sensitivity: bool = True) -> List[dict]:
    """
    Prefer the per-split files (mussel_out/*_delta_pcc_phe*.csv), which can be
    trimmed to the common split count.  Only fall back to the already-aggregated
    *_screen_phe*.csv when no per-split file covers a trait.

    NEVER recompute pos_frac from an aggregated row: an earlier collation script
    did that, which collapsed the positive fraction to the sign of one number and
    produced spurious 'usable' verdicts for effects of order 1e-7.
    """
    rows: List[dict] = []

    if inspect:
        pats = list(MUSSEL_PERSPLIT.values()) + ["*screen*phe*.csv"]
        for pat in pats:
            for path in sorted(glob.glob(os.path.join(root, pat))):
                df = _read_any(path)
                print(f"\n--- {path}")
                print(f"    rows={len(df)}  columns={list(df.columns)}")
                print(df.head(3).to_string(index=False))
        return rows

    covered = set()
    for module, pat in MUSSEL_PERSPLIT.items():
        for path in sorted(glob.glob(os.path.join(root, pat))):
            got = _per_split_rows(path, module, "mussel", n_splits, thr,
                                  MUSSEL_ABBREV, MUSSEL_DERIVED, MUSSEL_DROP,
                                  emit_sensitivity)
            rows += got
            covered |= {(r["trait_raw"], r["module"]) for r in got}

    # fallback: aggregated files for anything the per-split files missed
    for path in sorted(glob.glob(os.path.join(root, "*screen*phe*.csv"))):
        base = os.path.basename(path)
        module = ("dominance" if "dominance" in base.lower()
                  else "epistasis" if "epistasis" in base.lower() else None)
        if module is None:
            continue
        df = _read_any(path)
        c_trait = _find_col(df, TRAIT_KEYS)
        c_delta = _find_col(df, DELTA_KEYS)
        c_pf = _find_col(df, ("pos_frac", "posfrac", "frac_pos"))
        c_sd = _find_col(df, ("sd_dpcc", "sd", "std"))
        if c_trait is None or c_delta is None or c_pf is None:
            continue
        for _, r in df.iterrows():
            raw = str(r[c_trait]).strip()
            if raw in MUSSEL_DROP or raw not in MUSSEL_ABBREV:
                continue
            if (raw, module) in covered:
                continue
            stem = base.rsplit(".", 1)[0]
            n_here = int(overrides.get(stem, overrides.get(raw, splits_default)))
            mean_d, pf = float(r[c_delta]), float(r[c_pf])
            sd_d = (float(r[c_sd]) if c_sd is not None and pd.notna(r[c_sd])
                    else float("nan"))
            if n_here != n_splits and not keep_offcount:
                print(f"[mus ] dropping {raw}/{module} from {base}: "
                      f"{n_here} splits != target {n_splits}")
                continue
            v = ("sensitivity" if n_here != n_splits
                 else verdict_for(mean_d, pf, n_here, n_splits, thr))
            rows.append(dict(
                species="mussel", trait=MUSSEL_ABBREV[raw], trait_raw=raw,
                module=module,
                model=("A+D vs A" if module == "dominance"
                       else "A+D+MULT_t vs A+D"),
                role=("primary" if v != "sensitivity" else "sensitivity"),
                n_splits_available=n_here, n_splits_used=n_here,
                mean_dPCC=mean_d, sd_dPCC=sd_d, pos_frac=pf,
                nominal_p=nominal_p(n_here, pf),
                derived=(raw in MUSSEL_DERIVED),
                source=base, fmt="aggregated(fallback)", verdict=v,
            ))
    return rows


# --------------------------------------------------------------------------- #
# De-duplication                                                               #
# --------------------------------------------------------------------------- #

def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """
    One row per (species, trait, module, model).  When a trait appears in more
    than one source file, prefer the row whose split count matches the target
    (already enforced upstream); among remaining ties prefer the file whose name
    is trait-specific (e.g. *_phe4_phe4.csv) over the batch file, because the
    trait-specific file is the later, deliberate rerun.

    Every dropped duplicate is printed, never silently discarded.
    """
    key = ["species", "trait", "module", "model", "role"]
    df = df.copy()
    if "role" not in df.columns:
        df["role"] = "primary"
    _spec = re.compile(r"_(phe\d+)_\1\.")
    df["_specific"] = df["source"].astype(str).map(
        lambda s: bool(_spec.search(s)))
    # Prefer the BATCH file for the primary row: every trait was screened under
    # the same batch protocol, whereas a trait-specific rerun exists only for the
    # trait that looked promising.  Using the batch row as primary is what keeps
    # the table free of a winner-only rerun.  The trait-specific file still
    # contributes its all-splits sensitivity row.
    df["_rank"] = df["_specific"].astype(int)
    df = df.sort_values(key + ["_rank"], kind="mergesort")
    dup = df.duplicated(subset=key, keep="first")
    for _, r in df[dup].iterrows():
        print(f"[dedup] duplicate {r['species']}/{r['trait']}/{r['module']}"
              f" [{r['role']}] from {r['source']}: "
              f"dPCC={r['mean_dPCC']:+.5f}, pos_frac={r['pos_frac']:.2f} "
              f"(n={r['n_splits_used']}) -- not written. If this is the same "
              f"split count as the row that was kept, the two numbers should "
              f"agree; a large discrepancy means the seed schemes differ.")
    out = df[~dup].drop(columns=["_specific", "_rank"])
    return out


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #

def parse_overrides(items: Optional[List[str]]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for it in items or []:
        if "=" not in it:
            raise SystemExit(f"--mussel_splits_override expects key=value, got {it!r}")
        k, v = it.split("=", 1)
        out[k.strip()] = int(v)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pig_root", default=None,
                    help="directory containing pig_bias_<trait>/ subdirectories")
    ap.add_argument("--mussel_root", default=None,
                    help="directory containing *_screen_phe*.csv")
    ap.add_argument("--n_splits", type=int, default=20,
                    help="target number of random splits for every row")
    ap.add_argument("--posfrac_threshold", type=float, default=0.80)
    ap.add_argument("--mussel_splits_default", type=int, default=20)
    ap.add_argument("--mussel_splits_override", nargs="*", default=None,
                    metavar="KEY=N",
                    help="e.g. dominance_screen_phe4_phe4=100")
    ap.add_argument("--keep_offcount_as_sensitivity", action="store_true",
                    help="retain rows whose split count differs from the target, "
                         "labelled verdict='sensitivity' (never 'usable')")
    ap.add_argument("--core_models", nargs="*", default=list(CORE_MODELS),
                    help="models that carry a manuscript claim; every other "
                         "model is reported as 'diagnostic'. Pass an empty list "
                         "to score every model.")
    ap.add_argument("--no_sensitivity_rows", action="store_true",
                    help="do not emit the extra all-splits row for files that "
                         "contain more splits than the target")
    ap.add_argument("--inspect", action="store_true",
                    help="print the real columns of every file found and exit")
    ap.add_argument("--out", default="Table_screening.csv")
    args = ap.parse_args()

    if not args.pig_root and not args.mussel_root:
        raise SystemExit("give at least one of --pig_root / --mussel_root")

    overrides = parse_overrides(args.mussel_splits_override)
    global _CORE
    _CORE = tuple(args.core_models or ())

    rows: List[dict] = []
    if args.pig_root:
        rows += collect_pig(args.pig_root, args.n_splits,
                            args.posfrac_threshold, args.inspect,
                            not args.no_sensitivity_rows)
    if args.mussel_root:
        rows += collect_mussel(args.mussel_root, args.n_splits,
                               args.posfrac_threshold,
                               args.mussel_splits_default, overrides,
                               args.keep_offcount_as_sensitivity, args.inspect,
                               not args.no_sensitivity_rows)

    if args.inspect:
        print("\n[inspect] done -- send the output above and the column mapping "
              "will be fixed if anything was mis-detected.")
        return

    if not rows:
        raise SystemExit("nothing collected; run with --inspect")

    df = deduplicate(pd.DataFrame(rows))
    df = df.sort_values(["species", "module", "trait"], kind="mergesort")

    cols = ["species", "trait", "trait_raw", "module", "model", "role",
            "n_splits_available", "n_splits_used", "mean_dPCC", "sd_dPCC",
            "pos_frac", "nominal_p", "derived", "source", "fmt", "verdict"]
    df = df[[c for c in cols if c in df.columns]]
    df.to_csv(args.out, index=False)

    # ---- audit block: everything the manuscript must be able to state --------
    print("\n================ harmonisation report ================")
    print(f"target splits           : {args.n_splits}")
    print(f"pos_frac threshold      : {args.posfrac_threshold} "
          f"(nominal one-sided p = {nominal_p(args.n_splits, args.posfrac_threshold):.4g})")
    print(f"rows written            : {len(df)}  ->  {args.out}")

    diag = df[df["verdict"] == "diagnostic"]
    if len(diag):
        print(f"\n[diagnostic rows: {len(diag)} secondary models at a reduced "
              f"split budget by design -- not scored, not a gap]")
        for (sp, mdl), g in diag.groupby(["species", "model"]):
            print(f"    {sp:6s} {mdl:12s} {len(g)} traits, "
                  f"{int(g['n_splits_used'].max())} splits")

    inc = df[df["verdict"] == "INCOMPLETE"]
    if len(inc):
        print("\n*** INCOMPLETE -- these need more splits before they can be "
              "reported at the common count ***")
        for _, r in inc.iterrows():
            print(f"    {r['species']:6s} {r['trait']:6s} {r['module']:10s} "
                  f"{r['model']:20s} have {r['n_splits_available']:>3d}, "
                  f"need {args.n_splits}")

    sens = df[df["verdict"] == "sensitivity"]
    if len(sens):
        print("\n[sensitivity rows retained -- must not be reported as 'usable']")
        for _, r in sens.iterrows():
            print(f"    {r['species']:6s} {r['trait']:6s} {r['module']:10s} "
                  f"{r['n_splits_used']} splits, dPCC={r['mean_dPCC']:+.5f}, "
                  f"pos_frac={r['pos_frac']:.2f}")

    print("\n[usable]")
    us = df[df["verdict"] == "usable"]
    if len(us) == 0:
        print("    (none)")
    for _, r in us.iterrows():
        print(f"    {r['species']:6s} {r['trait']:6s} {r['module']:10s} "
              f"dPCC={r['mean_dPCC']:+.5f}  pos_frac={r['pos_frac']:.2f}  "
              f"n={r['n_splits_used']}")
    print("======================================================\n")


# --------------------------------------------------------------------------- #
# Self-test                                                                    #
# --------------------------------------------------------------------------- #

def _selftest() -> int:
    """Synthetic end-to-end check of the behaviours that matter."""
    import tempfile
    ok = True
    rng = np.random.default_rng(0)

    with tempfile.TemporaryDirectory() as td:
        pig = os.path.join(td, "pig")
        mus = os.path.join(td, "mussel_out")
        os.makedirs(mus)

        # --- pig: 30-split dominance, 10-split epistasis -------------------- #
        for trait, dom_mean, epi_n in (("bf", 0.013, 30), ("lmd", 0.0009, 10)):
            d = os.path.join(pig, f"pig_bias_{trait}")
            os.makedirs(d)
            pd.DataFrame({"seed": np.arange(1, 31), "model": "A+D vs A",
                          "delta_pcc": rng.normal(dom_mean, 0.005, 30)}
                         ).to_csv(os.path.join(d, "dominance_delta_pcc.csv"),
                                  index=False)
            pd.DataFrame({"seed": np.arange(1, epi_n + 1),
                          "model": "A+D+MULT_t",
                          "delta_pcc": rng.normal(0.015, 0.004, epi_n)}
                         ).to_csv(os.path.join(d, "diffuse_delta_pcc.csv"),
                                  index=False)

        # --- mussel per-split: batch file (20 splits, phe1+phe4) and a
        #     trait-specific rerun for phe4 at 100 splits ------------------- #
        base = np.concatenate([rng.normal(-0.003, 0.007, 20),
                               rng.normal(0.023, 0.044, 20)])
        pd.DataFrame({"trait": ["phe1"] * 20 + ["phe4"] * 20,
                      "seed": list(range(1, 21)) * 2,
                      "model": "A+D vs A", "delta_pcc": base}
                     ).to_csv(os.path.join(mus,
                              "dominance_delta_pcc_phe1_phe5.csv"), index=False)
        pd.DataFrame({"trait": "phe4", "seed": np.arange(1, 101),
                      "model": "A+D vs A",
                      "delta_pcc": rng.normal(0.040, 0.036, 100)}
                     ).to_csv(os.path.join(mus,
                              "dominance_delta_pcc_phe4_phe4.csv"), index=False)
        # an excluded trait must never surface
        pd.DataFrame({"trait": "phe13", "seed": np.arange(1, 21),
                      "model": "A+D+MULT_t vs A+D",
                      "delta_pcc": rng.normal(0.010, 0.004, 20)}
                     ).to_csv(os.path.join(mus,
                              "diffuse_delta_pcc_phe11_phe15.csv"), index=False)

        rows = collect_pig(pig, 20, 0.80, False, True)
        rows += collect_mussel(mus, 20, 0.80, 20, {}, False, False, True)
        df = deduplicate(pd.DataFrame(rows))
        prim = df[df.role == "primary"]

        # (1) 30-split pig file trimmed to exactly 20 in the primary row
        r = prim[(prim.trait == "bf") & (prim.module == "dominance")].iloc[0]
        if r["n_splits_used"] != 20 or r["n_splits_available"] != 30:
            print("FAIL: pig 30-split not trimmed"); ok = False

        # (2) and it also produced an all-splits sensitivity row
        sens = df[(df.trait == "bf") & (df.module == "dominance")
                  & (df.role == "sensitivity")]
        if len(sens) != 1 or sens.iloc[0]["n_splits_used"] != 30:
            print("FAIL: pig sensitivity row missing"); ok = False
        if len(sens) and sens.iloc[0]["verdict"] != "sensitivity":
            print("FAIL: sensitivity row not labelled"); ok = False

        # (3) 10-split epistasis flagged INCOMPLETE, never silently aggregated
        r = prim[(prim.trait == "lmd") & (prim.module == "epistasis")].iloc[0]
        if r["verdict"] != "INCOMPLETE":
            print(f"FAIL: lmd verdict {r['verdict']!r}"); ok = False

        # (4) SH: primary from 20 seeds, sensitivity from 100
        sh = df[(df.trait == "SH") & (df.module == "dominance")]
        shp = sh[sh.role == "primary"]
        shs = sh[sh.role == "sensitivity"]
        if len(shs) != 1 or shs.iloc[0]["n_splits_used"] != 100:
            print("FAIL: SH 100-split sensitivity row missing"); ok = False
        if not len(shp) or (shp["n_splits_used"] != 20).any():
            print("FAIL: SH primary not at 20 splits"); ok = False
        if len(shs) and shs.iloc[0]["verdict"] == "usable":
            print("FAIL: sensitivity row marked usable"); ok = False

        # (5) excluded trait absent
        if (df.trait_raw == "phe13").any():
            print("FAIL: phe13 present"); ok = False

        # (6) pig rows carry a real trait name, not the placeholder
        if (df.trait_raw == "__all__").any():
            print("FAIL: pig trait placeholder leaked"); ok = False

    # (7) nominal p arithmetic
    if abs(nominal_p(20, 0.80) - 0.005909) > 1e-5:
        print(f"FAIL: nominal_p(20,0.80)={nominal_p(20,0.80)}"); ok = False
    if abs(nominal_p(30, 0.80) - 0.000717) > 1e-5:
        print(f"FAIL: nominal_p(30,0.80)={nominal_p(30,0.80)}"); ok = False

    print("SELFTEST: " + ("ALL PASS" if ok else "FAILURES ABOVE"))
    return 0 if ok else 1


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(_selftest())
    main()
