#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 MANIFEST -- what you have, what you do not, and what that costs
================================================================================

 Printed BEFORE any computation. Six sections:

   A  inputs detected, with their specifications
   B  inputs absent, and the consequence of each one
   C  the execution path this implies
   D  the analyses that will run and the files they produce
   E  the conclusions that will NOT be available
   F  the provenance label every artefact will carry

 Section E is the point of the whole stage. Listing up front what cannot be
 concluded is more useful than explaining it afterwards, and it is the part a
 breeder needs in order to know how far to trust the plan.

 SUBSTITUTION POLICY
 ------------------------------------------------------------------------------
 Missing inputs are handled in one of three ways, never silently:

   SIMULATE      legitimate and necessary. Calibration curves are simulation by
                 construction; future generations and map-density sensitivity
                 have no empirical counterpart to begin with.
   DEGRADE       the analysis still runs but answers a weaker question, and is
                 relabelled. Missing F1 genotypes turn a prospective validation
                 into a demonstration; it does not stay called validation.
   REFUSE        no substitute exists. Reference genotypes and phenotypes are
                 the analysis. Economic weights encode what the breeder wants
                 and are a value judgement the agent has no standing to invent.

 HARD GUARD
 ------------------------------------------------------------------------------
 Any mating plan whose inputs include a SIMULATED component is written as a
 demonstration file and is refused as a recommendation. A breeder handed a plan
 built on simulated marker effects, not knowing it was simulated, would act on
 it. The filename, the provenance field and the report all have to say so.
================================================================================
"""
from __future__ import annotations

import gzip
import json
import os

import numpy as np
import pandas as pd

SIMULATE, DEGRADE, REFUSE, PRESENT = "SIMULATE", "DEGRADE", "REFUSE", "PRESENT"
EMPIRICAL, SIMULATED, HYBRID = "EMPIRICAL", "SIMULATED", "HYBRID"


def _peek_vcf(path, max_lines=2000):
    """Cheap header + sample scan: no full parse, no genotype matrix."""
    op = gzip.open if str(path).endswith(".gz") else open
    n_ind = 0
    ph = un = 0
    het = tot = 0
    seen = 0
    with op(path, "rt") as f:
        for line in f:
            if line.startswith("##"):
                continue
            p = line.rstrip("\n").split("\t")
            if line.startswith("#CHROM"):
                n_ind = len(p) - 9
                continue
            for g in p[9:]:
                gt = g.split(":", 1)[0]
                sep = "|" if "|" in gt else ("/" if "/" in gt else None)
                if sep is None:
                    continue
                ph += sep == "|"
                un += sep == "/"
                a, b = gt.split(sep)
                if a != "." and b != ".":
                    het += a != b
                    tot += 1
            seen += 1
            if seen >= max_lines:
                break
    return dict(n_individuals=n_ind,
                phased_fraction=ph / max(ph + un, 1),
                heterozygosity=het / max(tot, 1),
                markers_scanned=seen)


class Manifest:
    def __init__(self, cfg):
        self.cfg = cfg
        self.present = []
        self.absent = []
        self.path = None
        self.analyses = []
        self.unavailable = []
        self.provenance = {}
        self.blocking = []

    # ------------------------------------------------------------------ A/B
    def have(self, name, detail):
        self.present.append(dict(input=name, detail=detail))

    def lack(self, name, policy, consequence, substitute=None):
        self.absent.append(dict(input=name, policy=policy,
                                consequence=consequence,
                                substitute=substitute))
        if policy == REFUSE:
            self.blocking.append(f"{name}: {consequence}")

    def analysis(self, name, outputs, provenance=EMPIRICAL):
        self.analyses.append(dict(analysis=name, outputs=outputs,
                                  provenance=provenance))
        self.provenance[name] = provenance

    def cannot(self, statement):
        self.unavailable.append(statement)

    # -------------------------------------------------------------- inspect
    def inspect(self):
        cfg = self.cfg
        g = cfg.get("geno")

        def geno_exists(path):
            if not path:
                return False
            if os.path.exists(str(path)):
                return True
            return all(os.path.exists(str(path) + ext)
                      for ext in (".bed", ".bim", ".fam"))

        if not geno_exists(g):
            self.lack("reference genotypes", REFUSE,
                      "these are the analysis; nothing can substitute for them")
        else:
            info = _peek_vcf(g) if str(g).lower().endswith((".vcf", ".vcf.gz")) \
                else dict(n_individuals="?", phased_fraction=0.0,
                          heterozygosity=float("nan"), markers_scanned=0)
            self.have("reference genotypes",
                      f"{g}: ~{info['n_individuals']} individuals, "
                      f"phased fraction {info['phased_fraction']:.3f}, "
                      f"heterozygosity {info['heterozygosity']:.4f} "
                      f"(from the first {info['markers_scanned']} markers)")
            if info["phased_fraction"] < 0.99:
                self.lack("phased haplotypes", DEGRADE,
                          "segregation variance cannot be simulated, so "
                          "usefulness (mean + i x SD) is unavailable and "
                          "selection falls back to the cross mean alone",
                          "cross-mean selection only")
                self.cannot("no within-cross variance, so no usefulness "
                            "criterion and no five-generation projection")
            if np.isfinite(info["heterozygosity"]) and \
                    info["heterozygosity"] <= 0.02:
                self.lack("heterozygous individuals", DEGRADE,
                          f"heterozygosity is {info['heterozygosity']:.4f}; in "
                          "inbred or doubled-haploid material dominance "
                          "deviations are not estimable",
                          "dominance module disabled")
                self.cannot("nothing can be concluded about dominance in this "
                            "material -- absence of the module is a property "
                            "of the material, not a result")

        p = cfg.get("pheno")
        if not p or not os.path.exists(str(p)):
            self.lack("phenotypes", REFUSE,
                      "these are the analysis; nothing can substitute for them")
        else:
            df = pd.read_csv(p, nrows=5)
            self.have("phenotypes",
                      f"{p}: columns {list(df.columns)[:12]}"
                      + (" ..." if len(df.columns) > 12 else ""))

        if cfg.get("breeding_geno"):
            self.have("breeding panel", str(cfg["breeding_geno"]))
        else:
            self.lack("breeding panel", DEGRADE,
                      "allocation runs inside the reference population, so "
                      "candidates and training individuals are the same people",
                      "allocate within reference")

        if cfg.get("sex"):
            self.have("sex assignments", str(cfg["sex"]))
        else:
            self.lack("sex assignments", DEGRADE,
                      "all pairwise combinations are considered; the resulting "
                      "grouping is structural and NOT biological",
                      "all pairwise crosses")

        if cfg.get("cm_per_mb"):
            self.have("genetic map", f"{cfg['cm_per_mb']} cM/Mb (supplied)")
        else:
            self.lack("measured genetic map", SIMULATE,
                      "1.0 cM/Mb is assumed and a 0.5 / 1.0 / 2.0 sensitivity "
                      "sweep becomes mandatory; every multi-generation number "
                      "is reported as a range, never a point estimate",
                      "assumed map + sensitivity sweep")
            self.cannot("no point estimate of multi-generation gain; only a "
                        "range across map-density assumptions")

        if cfg.get("multitrait"):
            if cfg.get("weights"):
                self.have("economic weights", str(cfg["weights"]))
            else:
                self.lack("economic weights", REFUSE,
                          "weights encode what the breeding programme values. "
                          "That is a value judgement, not a statistical "
                          "quantity, and the agent has no standing to invent "
                          "one and then hand back a plan built on it")
            if cfg.get("step1_npz"):
                self.have("multi-trait variance components",
                          str(cfg["step1_npz"]))
            else:
                self.lack("multi-trait variance components", REFUSE,
                          "the selection index needs the genetic and "
                          "phenotypic covariance matrices across traits; run "
                          "the multi-trait REML step first")

        if cfg.get("f1_pheno"):
            self.have("F1 families", str(cfg["f1_pheno"]))
        else:
            self.lack("F1 progeny records", DEGRADE,
                      "simulated progeny replace observed ones. The output is "
                      "a DEMONSTRATION of the workflow, not a validation of "
                      "it, and is labelled as such throughout",
                      "simulated progeny (demonstration only)")
            self.cannot("no prospective test against real offspring")

        if not cfg.get("calibrate"):
            self.lack("module calibration", DEGRADE,
                      "without injected-variance calibration there is no "
                      "false-positive rate and no detection limit, so every "
                      "negative is uninterpretable",
                      "verdicts labelled UNCALIBRATED")
            self.cannot("no negative result may be reported as evidence that "
                        "an effect is absent")

    # ----------------------------------------------------------------- plan
    def plan(self):
        cfg = self.cfg
        mt = bool(cfg.get("multitrait"))
        self.path = ("multi-trait: selection index over traits, usefulness "
                     "criterion, Pareto front (mussel step2 logic)" if mt else
                     "single-trait: per-trait effect screen then "
                     "sex-structured allocation (pig pipeline logic)")
        if cfg.get("calibrate"):
            self.analysis("module calibration",
                          ["calibration.csv"], SIMULATED)
        self.analysis("effect screen (dominance and epistasis per trait)",
                      ["screen.csv"], EMPIRICAL)
        self.analysis("allocation model decision per trait",
                      ["decisions.csv"],
                      HYBRID if cfg.get("calibrate") else EMPIRICAL)
        sim_inputs = any(x["policy"] == SIMULATE for x in self.absent)
        alloc_prov = HYBRID if sim_inputs else EMPIRICAL
        if mt:
            self.analysis("Smith-Hazel index (correct b = P^-1 G a, with the "
                          "naive b = a/sd reported alongside)",
                          ["index_weights.csv"], EMPIRICAL)
            self.analysis("multi-trait allocation",
                          ["all_crosses.csv", "pareto_front.csv",
                           "recommended_crosses.csv"], alloc_prov)
        else:
            self.analysis("single-trait allocation",
                          ["*_mating_plan.csv", "allocation_summary.csv"],
                          alloc_prov)
        self.demo_only = sim_inputs

    # --------------------------------------------------------------- render
    def render(self, say, out_dir):
        bar = "=" * 74
        say("\n" + bar)
        say("MANIFEST -- read this before the run starts")
        say(bar)

        say("\nA. INPUTS DETECTED")
        for x in self.present:
            say(f"    [have] {x['input']}")
            say(f"           {x['detail']}")
        if not self.present:
            say("    (none)")

        say("\nB. INPUTS ABSENT, AND WHAT EACH COSTS")
        for x in self.absent:
            say(f"    [{x['policy']}] {x['input']}")
            say(f"           {x['consequence']}")
            if x["substitute"]:
                say(f"           -> substituted with: {x['substitute']}")
        if not self.absent:
            say("    (nothing missing)")

        say("\nC. EXECUTION PATH")
        say(f"    {self.path}")

        say("\nD. ANALYSES THAT WILL RUN")
        for x in self.analyses:
            say(f"    [{x['provenance']:9s}] {x['analysis']}")
            say(f"                 -> {', '.join(x['outputs'])}")

        say("\nE. CONCLUSIONS THAT WILL NOT BE AVAILABLE")
        for x in self.unavailable:
            say(f"    - {x}")
        if not self.unavailable:
            say("    (none; every planned conclusion is supported by the "
                "supplied data)")

        say("\nF. PROVENANCE")
        say("    EMPIRICAL  derived entirely from the data you supplied")
        say("    SIMULATED  generated; carries no information about your "
            "population")
        say("    HYBRID     empirical data combined with a simulated component")
        if getattr(self, "demo_only", False):
            say("\n    [GUARD] a simulated component feeds the allocation, so "
                "the mating plan is written as")
            say("            DEMONSTRATION_NOT_FOR_USE and is refused as a "
                "recommendation.")

        if self.blocking:
            say("\n" + bar)
            say("BLOCKED -- the run cannot proceed")
            say(bar)
            for b in self.blocking:
                say(f"    {b}")

        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "manifest.json"), "w") as f:
            json.dump(dict(present=self.present, absent=self.absent,
                           path=self.path, analyses=self.analyses,
                           unavailable=self.unavailable,
                           blocking=self.blocking,
                           demonstration_only=getattr(self, "demo_only", False)),
                      f, indent=2, ensure_ascii=False)
        say("\n" + bar)
        return len(self.blocking) == 0


def build_manifest(cfg, say, out_dir):
    m = Manifest(cfg)
    m.inspect()
    m.plan()
    ok = m.render(say, out_dir)
    return m, ok
