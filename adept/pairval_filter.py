#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
pairval_filter.py
=================

Restrict a trait's pair-validation results to the pairs that trait's OWN Stage-1
actually nominated, and recompute the candidate count, the retained count and
the pass rate from the existing per-fold results.

WHY
---
For the mussel trait phe8 (SWR) the pair inventory records 255 candidate pairs,
while every other trait records 116-133.  With --n_candidates k = 30 per seed
and five seeds the arithmetic ceiling is 5 x 30 = 150 unique pairs, so 255 is
impossible from phe8's own nominations alone: pool files belonging to other
traits were present in the _scores directory when Stage-1 ran, and their pairs
entered phe8's candidate set.

NOTHING IS REFITTED.  per_pair_per_fold.csv already holds every fold's result
for every pair that was tested; testing a pair nominated by another trait is not
in itself wrong, it only makes the pass rate incomparable across traits.  So the
fix is to drop the rows whose pair does not appear in this trait's own pool
files and re-derive the summary.  That is cheaper than rerunning Stage-1 and it
keeps the numbers identical to the ones already computed.

PAIR IDENTITY
-------------
Pairs are matched as an unordered pair of SNP indices, (min, max), because the
pool files and the per-fold table need not store them in the same order.  Seed
is part of the key: a pair only counts as nominated for a seed if it appears in
that seed's pool file.

USAGE
-----
    python pairval_filter.py --dir hcu/agent/mating/pairval_phe8 \\
        --trait phe8 --inspect
    python pairval_filter.py --dir hcu/agent/mating/pairval_phe8 \\
        --trait phe8 --out_prefix phe8_own
    python pairval_filter.py --selftest
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

Pair = Tuple[str, str]


def _pk(a, b) -> Pair:
    """Unordered pair key. SNP identifiers are strings (e.g. 'A7955'), so they
    are compared as strings; casting to int silently fails on real data."""
    a, b = str(a).strip(), str(b).strip()
    return (a, b) if a <= b else (b, a)


def _col(df: pd.DataFrame, keys: Sequence[str]) -> Optional[str]:
    low = {c.lower(): c for c in df.columns}
    for k in keys:
        if k.lower() in low:
            return low[k.lower()]
    for k in keys:
        for lc, orig in low.items():
            if k.lower() in lc:
                return orig
    return None


def seed_from_name(path: str) -> Optional[int]:
    """Extract the seed from a name like phe8_pool_seed2024_k30.json."""
    base = os.path.basename(path)
    tag = "seed"
    i = base.find(tag)
    if i < 0:
        return None
    j = i + len(tag)
    digits = ""
    while j < len(base) and base[j].isdigit():
        digits += base[j]
        j += 1
    return int(digits) if digits else None


def load_bim(path: str) -> List[str]:
    """SNP identifiers in file order (column 2 of a PLINK .bim)."""
    b = pd.read_csv(path, sep=r"\s+", header=None, dtype=str,
                    usecols=[1], names=["chr", "snp", "cm", "pos", "a1", "a2"])
    return [str(x).strip() for x in b["snp"].tolist()]


def resolve_offset(pool_idx: Set[int], names: List[str],
                   observed: Set[str]) -> Tuple[int, float]:
    """
    Decide whether pool files store 0-based or 1-based column indices.

    The pool files hold integer positions into the genotype matrix, while the
    per-fold table holds SNP identifiers, so the two can only be matched through
    the .bim.  Guessing the base wrong shifts every pair by one marker and
    produces a plausible-looking but wrong overlap, so both conventions are
    scored against the identifiers actually present and the better one is
    reported with its match rate.  A low or ambiguous rate aborts rather than
    picking one.
    """
    best = []
    for off in (0, 1):
        mapped = {names[i - off] for i in pool_idx
                  if 0 <= i - off < len(names)}
        rate = len(mapped & observed) / max(len(mapped), 1)
        best.append((rate, off))
    best.sort(reverse=True)
    (r0, o0), (r1, o1) = best[0], best[1]
    if r0 < 0.90:
        raise SystemExit(
            f"cannot map pool indices onto SNP identifiers: best match rate "
            f"{r0:.2f} ({o0}-based). Wrong .bim? It must be the marker set the "
            f"kernels were built on ({len(names)} markers here).")
    if r0 - r1 < 0.10:
        raise SystemExit(
            f"0-based and 1-based indexing match almost equally "
            f"({r0:.2f} vs {r1:.2f}); refusing to guess.")
    return o0, r0


def load_pools(scores_dir: str, trait: str) -> Dict[int, Set[Tuple[int, int]]]:
    """
    Read this trait's own pool files only.

    A pool file is a dict keyed by kernel type, each value a list of
    [score, snp_i, snp_j, tag].  Files whose basename does not start with
    '<trait>_' belong to another trait and are ignored -- that is the whole
    point of this script, so their presence is reported rather than tolerated
    silently.
    """
    own: Dict[int, Set[Tuple[int, int]]] = {}
    foreign: List[str] = []
    for path in sorted(glob.glob(os.path.join(scores_dir, "*_pool_seed*.json"))):
        base = os.path.basename(path)
        if not base.startswith(f"{trait}_"):
            foreign.append(base)
            continue
        seed = seed_from_name(path)
        if seed is None:
            print(f"[pool] cannot read a seed from {base}; skipped",
                  file=sys.stderr)
            continue
        with open(path) as fh:
            obj = json.load(fh)
        pairs: Set[Tuple[int, int]] = set()
        for _kernel, entries in obj.items():
            for e in entries:
                # [score, snp_i, snp_j, tag]
                if len(e) < 3:
                    continue
                i, j = int(e[1]), int(e[2])
                pairs.add((min(i, j), max(i, j)))
        own.setdefault(seed, set()).update(pairs)

    if foreign:
        print(f"[pool] {len(foreign)} pool file(s) in this directory belong to "
              f"another trait and were EXCLUDED: {foreign[:6]}"
              f"{' ...' if len(foreign) > 6 else ''}")
    return own


def pools_to_names(pools: Dict[int, Set[Tuple[int, int]]], names: List[str],
                   off: int) -> Dict[int, Set[Pair]]:
    out: Dict[int, Set[Pair]] = {}
    for seed, prs in pools.items():
        conv = set()
        for i, j in prs:
            a, b = i - off, j - off
            if 0 <= a < len(names) and 0 <= b < len(names):
                conv.add(_pk(names[a], names[b]))
        out[seed] = conv
    return out


def filter_perfold(pf: pd.DataFrame, pools: Dict[int, Set[Pair]],
                   match_seed: bool = False
                   ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split per-fold rows into (own, foreign).

    DEFAULT (match_seed=False): match on the PAIR only, against the union of
    this trait's pool files, and keep every row of a matched pair.

    That is the correct granularity.  In this table the seed column is the FOLD
    a pair was evaluated in, not the pool that nominated it: a pair nominated by
    one seed is then evaluated under all five, and the retention rule is that it
    improves in every one of them.  Matching on (pair, seed) would keep only the
    fold that nominated the pair, leaving one fold to take a minimum over, which
    inflates the pass rate -- for phe1 it turned 9 retained pairs into 17.  The
    negative control (traits with no foreign pools must reproduce their
    delivered counts exactly) is what exposes this, so run it after any change.

    match_seed=True restores the stricter (pair, seed) matching for diagnosis.
    """
    c_i, c_j = _col(pf, ("snp_i",)), _col(pf, ("snp_j",))
    c_seed = _col(pf, ("seed",))
    missing = [n for n, c in (("snp_i", c_i), ("snp_j", c_j), ("seed", c_seed))
               if c is None]
    if missing:
        raise ValueError(f"per-fold table lacks {missing}; "
                         f"columns = {list(pf.columns)}")
    if match_seed:
        sd = pd.to_numeric(pf[c_seed], errors="coerce")
        keep = np.array([
            (pd.notna(z) and _pk(x, y) in pools.get(int(z), set()))
            for x, y, z in zip(pf[c_i], pf[c_j], sd)])
    else:
        union: Set[Pair] = set().union(*pools.values()) if pools else set()
        keep = np.array([_pk(x, y) in union
                         for x, y in zip(pf[c_i], pf[c_j])])
    return pf[keep].copy(), pf[~keep].copy()


def summarise(pf: pd.DataFrame, min_d_floor: float = 0.0) -> dict:
    """
    Candidate pairs, retained pairs and pass rate under the stated rule:
    a pair is retained when its worst fold still improves prediction, i.e.
    min(delta) > min_d_floor.

    If the table carries no per-fold delta column the retained count cannot be
    recomputed here and is reported as None rather than guessed.
    """
    c_i, c_j = _col(pf, ("snp_i",)), _col(pf, ("snp_j",))
    pairs = {_pk(x, y) for x, y in zip(pf[c_i], pf[c_j])}
    c_d = _col(pf, ("d_vs_AGG", "delta", "dpcc", "d_pcc", "gain", "improve"))
    retained = None
    if c_d is not None:
        key = [_pk(x, y) for x, y in zip(pf[c_i], pf[c_j])]
        tmp = pf.assign(_k=key)
        worst = tmp.groupby("_k")[c_d].min()
        retained = int((worst > min_d_floor).sum())
    return dict(n_rows=len(pf), n_candidate_pairs=len(pairs),
                delta_col=c_d, n_usable_pairs=retained,
                pass_rate=(None if retained is None or not pairs
                           else retained / len(pairs)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=("--selftest" not in sys.argv), help="a pairval_<trait> directory")
    ap.add_argument("--trait", required=("--selftest" not in sys.argv), help="e.g. phe8")
    ap.add_argument("--bim", default=None,
                    help="PLINK .bim for the marker set the kernels were built "
                         "on; needed because pool files store column indices "
                         "while the per-fold table stores SNP identifiers")
    ap.add_argument("--match_seed", action="store_true",
                    help="diagnosis only: also require the pair to have been "
                         "nominated by that particular seed. This drops folds "
                         "and inflates the pass rate; not for the manuscript.")
    ap.add_argument("--min_d_floor", type=float, default=0.0)
    ap.add_argument("--out_prefix", default=None,
                    help="write <prefix>_per_pair_per_fold.csv and "
                         "<prefix>_summary.csv; nothing is overwritten")
    ap.add_argument("--inspect", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        sys.exit(_selftest())

    pf_path = os.path.join(a.dir, "per_pair_per_fold.csv")
    pf = pd.read_csv(pf_path)
    pf.columns = [c.strip() for c in pf.columns]
    pools = load_pools(os.path.join(a.dir, "_scores"), a.trait)

    if a.inspect:
        print(f"{pf_path}\n  rows={len(pf)}  columns={list(pf.columns)}")
        print(pf.head(3).to_string(index=False))
        print(f"\npool files: {len(pools)} seed(s)")
        for s, p in sorted(pools.items()):
            print(f"  seed {s}: {len(p)} nominated pairs")
        tot = set().union(*pools.values()) if pools else set()
        print(f"  union across seeds: {len(tot)} unique pairs")
        return

    if not pools:
        raise SystemExit(f"no pool files for trait {a.trait!r} under "
                         f"{a.dir}/_scores")
    if not a.bim:
        raise SystemExit("--bim is required (pool files store column indices, "
                         "the per-fold table stores SNP identifiers)")

    names = load_bim(a.bim)
    c_i, c_j = _col(pf, ("snp_i",)), _col(pf, ("snp_j",))
    observed = set(map(str, pf[c_i])) | set(map(str, pf[c_j]))
    all_idx = {i for prs in pools.values() for p in prs for i in p}
    off, rate = resolve_offset(all_idx, names, observed)
    print(f"[bim ] {len(names)} markers; pool indices are {off}-based "
          f"(match rate {rate:.3f})")
    pools = pools_to_names(pools, names, off)

    own, foreign = filter_perfold(pf, pools)
    before, after = summarise(pf, a.min_d_floor), summarise(own, a.min_d_floor)

    print("\n================ pair-validation filter ================")
    print(f"trait                     : {a.trait}")
    print(f"seeds with a pool file    : {len(pools)}  "
          f"(k per seed: {[len(v) for _, v in sorted(pools.items())]})")
    print(f"arithmetic ceiling        : {sum(len(v) for v in pools.values())} "
          f"pair-slots, {len(set().union(*pools.values()))} unique")
    print(f"\n                            {'as delivered':>14s}  "
          f"{'own pool only':>14s}")
    print(f"  rows                    : {before['n_rows']:>14d}  "
          f"{after['n_rows']:>14d}")
    print(f"  candidate pairs         : {before['n_candidate_pairs']:>14d}  "
          f"{after['n_candidate_pairs']:>14d}")
    if after["n_usable_pairs"] is None:
        print(f"  retained pairs          : {'n/a':>14s}  {'n/a':>14s}"
              f"   (no per-fold delta column found; columns = "
              f"{list(pf.columns)})")
    else:
        print(f"  retained pairs          : {before['n_usable_pairs']:>14d}  "
              f"{after['n_usable_pairs']:>14d}")
        print(f"  pass rate               : {before['pass_rate']:>14.4f}  "
              f"{after['pass_rate']:>14.4f}")
    print(f"\n  rows dropped (foreign)  : {len(foreign)}")
    print("========================================================\n")

    if a.out_prefix:
        p1 = os.path.join(a.dir, f"{a.out_prefix}_per_pair_per_fold.csv")
        p2 = os.path.join(a.dir, f"{a.out_prefix}_summary.csv")
        own.to_csv(p1, index=False)
        pd.DataFrame([dict(trait=a.trait, scope="own_pool_only", **after),
                      dict(trait=a.trait, scope="as_delivered", **before)]
                     ).to_csv(p2, index=False)
        print(f"written: {p1}\n         {p2}")


# --------------------------------------------------------------------------- #

def _selftest() -> int:
    import tempfile
    ok = True
    with tempfile.TemporaryDirectory() as td:
        sc = os.path.join(td, "_scores"); os.makedirs(sc)
        names = [f"A{i}" for i in range(100)]
        bim = os.path.join(td, "m.bim")
        with open(bim, "w") as fh:
            for k, nm in enumerate(names):
                fh.write(f"1\t{nm}\t0\t{1000+k}\tA\tG\n")
        if load_bim(bim) != names:
            print("FAIL: bim parse"); ok = False

        # pools store 1-BASED indices here
        own_idx = {2024: [(2, 3), (4, 5)], 2125: [(6, 7)]}
        for seed, prs in own_idx.items():
            json.dump({"MULT": [[0.3, i, j, "MULT"] for i, j in prs]},
                      open(os.path.join(sc, f"phe8_pool_seed{seed}_k30.json"), "w"))
        json.dump({"MULT": [[0.3, 90, 91, "MULT"]]},
                  open(os.path.join(sc, "phe14_pool_seed2024_k30.json"), "w"))

        # every own pair is evaluated under ALL seeds, not only the one that
        # nominated it -- this is what the real table looks like
        all_seeds = sorted(own_idx)
        rows = []
        for i, j in [p for prs in own_idx.values() for p in prs]:
            for sd_ in all_seeds:
                # pair (4,5) loses in a fold that did NOT nominate it: kept under the
                # default rule (so the pair fails), dropped under match_seed
                d = (-0.01 if (i, j) == (4, 5) and sd_ == all_seeds[1] else 0.02)
                rows.append(dict(snp_i=names[j-1], snp_j=names[i-1],
                                 seed=sd_, fold=0, d_vs_AGG=d))
        for sd_ in all_seeds:        # foreign pair from another trait's pool
            rows.append(dict(snp_i=names[89], snp_j=names[90], seed=sd_,
                             fold=0, d_vs_AGG=0.01))
        pf = pd.DataFrame(rows)

        pools = load_pools(sc, "phe8")
        if set(pools) != {2024, 2125}: print("FAIL: seeds"); ok = False
        obs = set(pf.snp_i) | set(pf.snp_j)
        allidx = {i for prs in pools.values() for p in prs for i in p}
        off, rate = resolve_offset(allidx, names, obs)
        if off != 1: print(f"FAIL: offset {off} != 1 (rate {rate})"); ok = False
        pn = pools_to_names(pools, names, off)
        if _pk("A89", "A90") in pn[2024]:
            print("FAIL: foreign pool leaked"); ok = False

        own, foreign = filter_perfold(pf, pn)          # default: pair-only
        if len(foreign) != 2:
            print(f"FAIL: foreign rows {len(foreign)} != 2"); ok = False
        if len(own) != 6:
            print(f"FAIL: own rows {len(own)} != 6 (all folds kept)"); ok = False
        # the stricter mode must keep fewer rows and inflate retention
        own_s, _ = filter_perfold(pf, pn, match_seed=True)
        if len(own_s) >= len(own):
            print("FAIL: match_seed did not drop folds"); ok = False
        if summarise(own_s)["n_usable_pairs"] <= summarise(own)["n_usable_pairs"]:
            print("FAIL: match_seed should inflate the pass rate here"); ok = False
        if not ((own.snp_i == "A2") & (own.snp_j == "A1")).any():
            print("FAIL: unordered string pair matching"); ok = False

        s_ = summarise(own)
        if s_["delta_col"] != "d_vs_AGG":
            print(f"FAIL: delta col {s_['delta_col']}"); ok = False
        if s_["n_candidate_pairs"] != 3:
            print(f"FAIL: candidates {s_['n_candidate_pairs']} != 3"); ok = False
        # (A3,A4) under seed 2125 has a negative fold -> not retained
        if s_["n_usable_pairs"] != 2:
            print(f"FAIL: retained {s_['n_usable_pairs']} != 2"); ok = False
        if summarise(own.drop(columns=["d_vs_AGG"]))["n_usable_pairs"] is not None:
            print("FAIL: retained guessed without delta"); ok = False

        # a wrong bim must abort, not silently mismatch
        bad = os.path.join(td, "bad.bim")
        with open(bad, "w") as fh:
            for k in range(100):
                fh.write(f"1\tZ{k}\t0\t{k}\tA\tG\n")
        try:
            resolve_offset(allidx, load_bim(bad), obs)
            print("FAIL: wrong bim accepted"); ok = False
        except SystemExit:
            pass

    print("SELFTEST: " + ("ALL PASS" if ok else "FAILURES ABOVE"))
    return 0 if ok else 1


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(_selftest())
    main()
