#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 validate_agent_on_pig.py -- does mateagent's own pipeline reach the right
 answer on real data we already know the answer to?
================================================================================

 This is the first time mateagent.py sees real data. Everything it has passed
 so far is a synthetic self-test, and synthetic self-tests have already missed
 three real bugs in this project (covariate scope, audit directory, calibrate
 checkpointing) -- all only surfaced on the actual machine, with actual data.
 This script is designed to catch a fourth one before another material does.

 It runs the trait bf through mateagent's REAL pipeline (ingest -> capability
 detection -> screen -> decide), NOT a re-implementation, and checks the
 output against numbers already established and cross-checked earlier in this
 project:

     dominance  A+D vs A          dPCC +0.00807, pos_frac 0.85
     epistasis  A+D+MULT_t vs A+D dPCC +0.01578, pos_frac 1.00
                (bf's epistasis effect exceeds the calibrated range --
                 calibration.csv only reaches dPCC +0.0038 at injected 0.15)

 CALIBRATION IS REUSED, NOT RECOMPUTED. The real calibration.csv (6 cells,
 5.5+ hours of compute) already exists for this exact dataset. Recomputing it
 here would repeat exactly the kind of redundant multi-hour run this project
 has already paid for twice. --calib_csv loads it instead.

 WHAT THIS SCRIPT DOES NOT COVER
 ------------------------------------------------------------------------------
 mateagent.py currently fits every trait with an intercept-only model; it has
 no covariate-diagnosis mechanism of its own. For pig this happens to be the
 right answer (covariates are established as confounded with family and
 excluded), but that is a coincidence of this dataset, not evidence that
 mateagent would get a covariate decision right on a species where covariates
 legitimately belong in the model. That is a real, separate gap -- flagged
 here, not silently absorbed into a "PASS".

     python validate_agent_on_pig.py \
         --bfile <plink_prefix> \
         --pheno_wide pig_pheno_wide.csv \
         --calib_csv pig_finish_out/calibration.csv \
         --out agent_pig_validation
================================================================================
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mateagent as M

# ---------------------------------------------------------------------------
# Known answers, established independently of the agent.
#
# These are the final 20-split, intercept-only numbers reported in the paper.
# The pig phenotypes are already adjusted upstream, so fitting covariates here
# would adjust them a second time and depress accuracy (BF additive accuracy
# near 0.31 instead of 0.40). A run that reproduces the 0.31 figure has not
# applied the intercept-only convention and should be stopped rather than
# interpreted.
#
# Bands are deliberately wider than the point estimates: the same command in
# a different computing environment can move dPCC by up to 0.009, because
# REML lands on a different point of a nearly flat likelihood ridge when the
# kernels are strongly correlated. Within one environment the pipeline is
# exactly reproducible.
# ---------------------------------------------------------------------------
KNOWN = {
    # BF: the clearest positive. Both modules accepted; the epistatic effect
    # is larger than anything the calibration grid reached, so the verdict
    # text must say so rather than report a spuriously precise equivalent.
    "bf": dict(
        dominance_dpcc=(0.00807, 0.003, 0.014), dominance_posfrac_min=0.75,
        epistasis_dpcc=(0.01578, 0.010, 0.022), epistasis_posfrac_min=0.90,
        epistasis_exceeds_calibration=True, expect_model="FULL"),
    # ELMP (code lmp): epistasis only.
    "lmp": dict(
        dominance_dpcc=(-0.00076, -0.005, 0.004), dominance_posfrac_min=0.0,
        epistasis_dpcc=(0.00589, 0.001, 0.011), epistasis_posfrac_min=0.75,
        epistasis_exceeds_calibration=None, expect_model=None),
    # RTN: epistasis accepted at exactly the threshold (16 of 20). Its band is
    # wider than the estimate itself, which is the honest statement: this
    # trait's decision is threshold-sensitive and the paper says so.
    "rtn": dict(
        dominance_dpcc=(0.00120, -0.003, 0.005), dominance_posfrac_min=0.0,
        epistasis_dpcc=(0.00337, -0.001, 0.008), epistasis_posfrac_min=0.65,
        epistasis_exceeds_calibration=None, expect_model=None),
    # LMD: the clean known negative. This tests the opposite failure mode from
    # BF -- does the system hold back when there is nothing there? Most of the
    # paper's claims rest on negatives being trustworthy, so a false positive
    # here would matter more than a missed call on BF.
    "lmd": dict(
        dominance_dpcc=(0.00279, -0.002, 0.008), dominance_posfrac_min=0.0,
        epistasis_dpcc=(0.00020, -0.004, 0.004), epistasis_posfrac_min=0.0,
        expect_model="ADD", expect_no_false_positive=True),
}


def check(label, ok, detail):
    mark = "PASS" if ok else "FAIL"
    print(f"  [{mark}] {label}: {detail}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bfile", required=True,
                    help="PLINK prefix of the QC'd, pruned genotypes")
    ap.add_argument("--pheno_wide", default="pig_pheno_wide.csv",
                    help="output of pig_pheno_convert.py")
    ap.add_argument("--pheno_for_geno", default="bf.phe",
                    help="a GCTA .phe file, used only to drive load_real()'s "
                         "row filter when reading the PLINK genotypes")
    ap.add_argument("--calib_csv", required=True,
                    help="the already-completed calibration.csv for THIS "
                         "dataset -- not recomputed")
    ap.add_argument("--traits", nargs="+", default=["bf"],
                    help="must have an entry in KNOWN; run bf first (the "
                         "clearest positive), then lmd (the clean known "
                         "negative) before trusting the system more broadly")
    ap.add_argument("--out", default="agent_pig_validation")
    a = ap.parse_args()

    if not os.path.exists(a.pheno_wide):
        raise SystemExit(
            f"{a.pheno_wide} not found. Build it first:\n"
            f"  python pig_pheno_convert.py --traits {' '.join(a.traits)} "
            f"--out {a.pheno_wide}")
    unknown = [t for t in a.traits if t not in KNOWN]
    if unknown:
        raise SystemExit(f"no known-answer entry for {unknown}; add one to "
                         f"KNOWN before validating against it")

    cfg = dict(M.DEFAULTS)
    cfg.update(
        geno=a.bfile, pheno=a.pheno_wide, traits=a.traits,
        pheno_for_geno=a.pheno_for_geno, pheno_for_geno_trait=a.traits[0],
        calib_csv=a.calib_csv, out=a.out, cache=f"{a.out}_cache", tag="pigval",
        splits=20, boot=1000, min_n=100, allocate=False, multitrait=False,
    )

    print("=" * 74)
    print(f"Running mateagent's real pipeline on real pig data: {a.traits}")
    print("=" * 74)
    M.run(cfg)

    import pandas as pd
    decisions_path = os.path.join(a.out, "decisions.csv")
    screen_path = os.path.join(a.out, "screen.csv")
    if not os.path.exists(decisions_path) or not os.path.exists(screen_path):
        raise SystemExit(f"expected outputs missing in {a.out}/; the run did "
                         f"not reach stage [3]. Check the log above.")
    D = pd.read_csv(decisions_path)
    S = pd.read_csv(screen_path)

    all_results = {}
    for trait in a.traits:
        exp = KNOWN[trait]
        drow = D[D.trait == trait]
        srow = S[S.trait == trait]
        if len(drow) == 0 or len(srow) == 0:
            print(f"\n[{trait}] MISSING from output -- did not reach decide")
            all_results[trait] = {"present": False}
            continue
        drow, srow = drow.iloc[0], srow.iloc[0]

        print(f"\n{'='*74}\n{trait}\n{'='*74}")
        results = {}
        lo, hi = exp["dominance_dpcc"][1], exp["dominance_dpcc"][2]
        results["dominance_dpcc_in_band"] = check(
            "dominance dPCC in expected band",
            lo <= srow.dPCC_dominance <= hi,
            f"got {srow.dPCC_dominance:+.5f}, expected [{lo:+.4f}, {hi:+.4f}]")
        results["dominance_posfrac"] = check(
            "dominance pos_frac at or above known minimum",
            srow.posfrac_dominance >= exp["dominance_posfrac_min"],
            f"got {srow.posfrac_dominance:.2f}, "
            f"need >= {exp['dominance_posfrac_min']:.2f}")

        lo, hi = exp["epistasis_dpcc"][1], exp["epistasis_dpcc"][2]
        results["epistasis_dpcc_in_band"] = check(
            "epistasis dPCC in expected band",
            lo <= srow.dPCC_epistasis <= hi,
            f"got {srow.dPCC_epistasis:+.5f}, expected [{lo:+.4f}, {hi:+.4f}]")
        results["epistasis_posfrac"] = check(
            "epistasis pos_frac at or above known minimum",
            srow.posfrac_epistasis >= exp["epistasis_posfrac_min"],
            f"got {srow.posfrac_epistasis:.2f}, "
            f"need >= {exp['epistasis_posfrac_min']:.2f}")
        if exp.get("epistasis_exceeds_calibration"):
            results["epistasis_exceeds_calibration"] = check(
                "epistasis verdict notes it exceeds the calibrated range",
                "exceeds the calibrated range" in str(drow.epistasis_verdict),
                drow.epistasis_verdict)
        if exp.get("expect_model"):
            results["model_routing"] = check(
                f"routed to {exp['expect_model']}",
                drow.model == exp["expect_model"],
                f"got model={drow.model}")
        if exp.get("expect_no_false_positive"):
            # The failure mode that matters most for a known-null trait: the
            # agent must NOT claim "detected" here. A wrong "detected" on a
            # trait this project has independently confirmed null, twice
            # (the original 20-seed screen and dominance_from_gebv.py), would
            # undermine every negative result the paper's narrative leans on.
            claimed_dom = "detected (" in str(drow.dominance_verdict)
            claimed_epi = "detected (" in str(drow.epistasis_verdict)
            results["no_false_positive"] = check(
                "does NOT falsely claim detection on this known-null trait",
                not (claimed_dom or claimed_epi),
                f"dominance_verdict claims detected={claimed_dom}, "
                f"epistasis_verdict claims detected={claimed_epi}")
        all_results[trait] = results

    print("\n" + "=" * 74)
    flat = {f"{t}.{k}": v for t, r in all_results.items()
           for k, v in r.items() if k != "present"}
    ok = all(flat.values()) and all(
        r.get("present", True) for r in all_results.values())
    print("OVERALL: " + ("ALL CHECKS PASS" if ok else "SOME CHECKS FAILED --"
                         " read above before trusting the system on the other "
                         "materials"))
    print("=" * 74)
    print("\nKNOWN GAP (not tested here, not a pass/fail item): mateagent "
          "currently fits every trait intercept-only, with no covariate "
          "diagnosis of its own. That happens to be correct for pig. It is "
          "untested whether the agent would correctly INCLUDE covariates on "
          "a species where they legitimately belong -- there is currently no "
          "mechanism in mateagent to do so at all.")

    json.dump(dict(results={t: {k: bool(v) for k, v in r.items()}
                            for t, r in all_results.items()},
                   overall_pass=bool(ok)),
              open(os.path.join(a.out, "validation_verdict.json"), "w"),
              indent=2)


if __name__ == "__main__":
    main()
