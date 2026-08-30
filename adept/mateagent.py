#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 MateAgent -- a species-agnostic decision pipeline for genomic mate allocation
================================================================================

 The pipeline decides, per trait and from the data, WHICH genetic effects are
 worth acting on, and then allocates matings using only those. Nothing is
 assumed about the species: what the data can support is detected, and every
 capability that is missing is reported rather than silently skipped.

     ADDITIVE ONLY        no non-additive effect clears its detection limit
     ADDITIVE + DOMINANCE dominance clears it; epistasis does not
     FULL                 epistasis clears it as well

 STAGES, each of which emits a report section
 --------------------------------------------------------------------------
   [0] INGEST      load data, detect capabilities, refuse impossible requests
   [1] CALIBRATE   inject known variance; derive the false-positive rate and
                   the detection limit of every module BEFORE any verdict
   [2] SCREEN      per trait, test dominance (A+D vs A) and epistasis
                   (A+D+E vs A+D) with paired bootstrap deltas
   [3] DECIDE      map each trait to an allocation model, and qualify every
                   negative as either "absent" or "below detection limit"
   [4] ALLOCATE    sex-structured or unrestricted allocation under use and
                   inbreeding constraints; compare against the additive plan
   [5] REPORT      consolidated machine-readable and human-readable output

 WHY CALIBRATION COMES BEFORE SCREENING
 --------------------------------------------------------------------------
 A module that never reports a positive and a module that is broken produce
 identical output. Until the false-positive rate and the detection limit are
 known, "no epistasis in this trait" carries no information and must not be
 handed to a breeder. Stage [3] therefore refuses to issue a calibrated verdict
 unless stage [1] has run, and labels its output UNCALIBRATED if it has not.

 CAPABILITY DETECTION -- what makes this species-agnostic
 --------------------------------------------------------------------------
   heterozygosity ~ 0   inbred material: dominance is not estimable. The
                        dominance module is disabled and the report says why,
                        rather than returning a meaningless number.
   genotypes unphased   segregation variance cannot be simulated; any stage
                        needing gametes is refused.
   no sex column        all pairwise crosses instead of sire x dam.
   no breeding panel    allocation runs within the reference population.
   no genetic map       recombination-aware simulation refused.

 A capability that is absent never degrades quietly into a wrong number.

 USAGE
     python mateagent.py --selftest
     python mateagent.py --config run.json
     python mateagent.py --geno ref_all_phased.vcf --pheno trait_imputed.csv \\
         --traits phe4 phe15 --breeding_geno breed_all_phased.vcf \\
         --sex cand_sex.csv --calibrate --out agent_out
================================================================================
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mateagent_manifest as MAN
import mussel_pipeline as MP
import power_curve as PW
import pig_epik_pipeline as P
import pig_epistasis_pipeline as Q

ADD, ADD_DOM, FULL = "ADD", "ADD+DOM", "FULL"


# ============================================================================
# Reporting
# ============================================================================
class Report:
    """
    Accumulates every stage's findings. Two consumers: a human reading the
    console, and a downstream process reading the JSON. Both see the same
    facts, including the ones that block progress.
    """

    def __init__(self, out_dir):
        self.out_dir = out_dir
        self.sections = []
        self.started = datetime.now(timezone.utc).isoformat()
        os.makedirs(out_dir, exist_ok=True)

    def section(self, key, title):
        s = dict(key=key, title=title, facts=[], warnings=[], blocks=[],
                 tables={})
        self.sections.append(s)
        MP.say("\n" + "=" * 74)
        MP.say(f"[{key}] {title}")
        MP.say("=" * 74)
        return s

    def fact(self, s, text, **kv):
        s["facts"].append(dict(text=text, **kv))
        MP.say(f"    {text}")

    def warn(self, s, text):
        s["warnings"].append(text)
        MP.say(f"    [warning] {text}")

    def block(self, s, text):
        """A capability that is unavailable. Recorded, never silently ignored."""
        s["blocks"].append(text)
        MP.say(f"    [unavailable] {text}")

    def table(self, s, name, df):
        s["tables"][name] = json.loads(df.to_json(orient="records"))
        path = os.path.join(self.out_dir, f"{name}.csv")
        df.to_csv(path, index=False)
        MP.say("\n" + df.round(5).to_string(index=False))

    def save(self):
        doc = dict(started=self.started,
                   finished=datetime.now(timezone.utc).isoformat(),
                   sections=self.sections)
        with open(os.path.join(self.out_dir, "agent_report.json"), "w") as f:
            json.dump(doc, f, indent=2, ensure_ascii=False)
        lines = ["MateAgent report", "=" * 74]
        for s in self.sections:
            lines.append(f"\n[{s['key']}] {s['title']}")
            for f_ in s["facts"]:
                lines.append(f"  - {f_['text']}")
            for w in s["warnings"]:
                lines.append(f"  ! {w}")
            for b in s["blocks"]:
                lines.append(f"  x unavailable: {b}")
        with open(os.path.join(self.out_dir, "agent_report.txt"), "w") as f:
            f.write("\n".join(lines) + "\n")


# ============================================================================
# [0] Ingest and capability detection
# ============================================================================
def phased_fraction(path, max_lines=2000):
    """
    Fraction of calls using the phased separator, from a sample of the file.
    read_phased_vcf prints this but does not return it, and a VCF full of '/'
    calls still yields a haplotype array -- so without this check an unphased
    file would be treated as phased and gamete simulation would silently run on
    arbitrary haplotype assignments.
    """
    import gzip
    op = gzip.open if str(path).endswith(".gz") else open
    ph = un = 0
    with op(path, "rt") as f:
        seen = 0
        for line in f:
            if line.startswith("#"):
                continue
            for g in line.rstrip("\n").split("\t")[9:]:
                gt = g.split(":", 1)[0]
                if "|" in gt:
                    ph += 1
                elif "/" in gt:
                    un += 1
            seen += 1
            if seen >= max_lines:
                break
    return ph / max(ph + un, 1)


def read_genotypes(path, pheno_hint=None):
    """
    Dispatch on what `path` actually is. Returns ids, chrom, pos, snp, H (or
    None), G.

    Three formats are supported:
      .vcf / .vcf.gz    phased or unphased
      .npz              must carry G, ids, chrom (optionally pos, snp)
      a PLINK prefix     i.e. path.bed / path.bim / path.fam all exist. This
                         is genotype-only in principle, but pig_epik_pipeline_
                         v6.load_real() couples genotype and phenotype loading
                         (row filtering depends on phenotype/covariate
                         missingness), so a phenotype file is required to
                         drive that filter. pheno_hint should be the FIRST
                         trait column available from --pheno if that file is
                         GCTA-style per-trait, or the wide multi-trait CSV
                         itself if it already has a compatible first column.
                         See pig_pheno_convert.py to build a compatible file
                         from GCTA .phe files.
    """
    low = str(path).lower()
    if low.endswith((".vcf", ".vcf.gz")):
        ids, chrom, pos, snp, H = MP.read_phased_vcf(path)
        G = MP.hap_to_geno(H)
        if phased_fraction(path) < 0.99:
            H = None          # haplotypes are not trustworthy; see stage 0
        return ids, chrom, pos, snp, H, G
    if low.endswith(".npz"):
        z = np.load(path, allow_pickle=True)
        need = ("G", "ids", "chrom")
        miss = [k for k in need if k not in z]
        if miss:
            raise SystemExit(f"{path} is missing arrays {miss}; "
                             f"present: {list(z.keys())}")
        G = np.asarray(z["G"], float)
        n_snp = G.shape[1]
        pos = np.asarray(z["pos"]) if "pos" in z else np.arange(n_snp)
        snp = (np.asarray(z["snp"], dtype=object) if "snp" in z
               else np.array([f"s{i}" for i in range(n_snp)], dtype=object))
        return (np.asarray(z["ids"], dtype=object), np.asarray(z["chrom"]),
                pos, snp, None, G)
    if all(os.path.exists(path + ext) for ext in (".bed", ".bim", ".fam")):
        if pheno_hint is None:
            raise SystemExit(
                f"{path} is a PLINK prefix (.bed/.bim/.fam found), but "
                "load_real() filters individuals by phenotype/covariate "
                "missingness, so it needs a phenotype column to drive that "
                "filter. Pass --pheno_for_geno <path>:<column> (a GCTA .phe "
                "file and the trait to key on), or convert the phenotype "
                "file to the wide id+trait CSV format first.")
        pheno_path, trait_col = pheno_hint
        pc = P.CFG()
        pc.BFILE, pc.PHENO_FILE, pc.TRAIT = path, pheno_path, trait_col
        pc.COV_FILE = pc.QCOV_FILE = None   # filtering only; no covariates
        X, _, chrom, pos, snp, _ = P.load_real(pc)
        ids = P.read_fam(path)["iid"].astype(str).values
        # load_real already row-filtered X to the individuals with a finite
        # phenotype for trait_col; ids must be filtered the same way to stay
        # aligned, so re-derive the keep mask exactly as load_real does.
        yv = P.load_pheno_gcta(pheno_path, ids, trait_col)
        keep = np.isfinite(yv)
        ids = ids[keep]
        if len(ids) != X.shape[0]:
            raise SystemExit(
                f"row count mismatch after PLINK load: {len(ids)} ids vs "
                f"{X.shape[0]} genotype rows. load_real()'s filtering must "
                f"have changed; do not proceed with a silent misalignment.")
        return ids, chrom, pos, snp, None, X.astype(float)
    raise SystemExit(
        f"unsupported genotype input: {path}\n"
        "  supported: .vcf / .vcf.gz (phased or unphased), .npz carrying "
        "G/ids/chrom, or a PLINK prefix with matching .bed/.bim/.fam.")


def detect_capabilities(G, H, chrom, pos, sex, Gb, rep, sec):
    """
    Establish what this dataset can and cannot support. Every False here
    disables a downstream stage explicitly instead of letting it return a
    number that looks fine and means nothing.
    """
    het = float(np.mean(G == 1))
    cap = dict(
        n=int(G.shape[0]), n_snp=int(G.shape[1]),
        heterozygosity=het,
        dominance_estimable=het > 0.02,
        phased=H is not None,
        sexed=sex is not None,
        separate_breeding_panel=Gb is not None,
        mapped=bool(pos is not None and np.ptp(pos) > 0),
    )
    rep.fact(sec, f"individuals {cap['n']}, markers {cap['n_snp']}")
    rep.fact(sec, f"mean heterozygosity {het:.4f}")

    if not cap["dominance_estimable"]:
        rep.block(sec, f"dominance module: heterozygosity is {het:.4f}. In "
                       "inbred or doubled-haploid material there are almost no "
                       "heterozygotes, so dominance deviations are not "
                       "estimable. This is a property of the material, not a "
                       "result: it must not be reported as 'no dominance'.")
    if not cap["phased"]:
        rep.block(sec, "gamete simulation: genotypes are unphased, so "
                       "segregation variance cannot be simulated. Allocation "
                       "on expected cross values is unaffected.")
    if not cap["sexed"]:
        rep.fact(sec, "no sex information: all pairwise crosses considered")
    else:
        rep.fact(sec, f"sexes present: {(sex=='M').sum()} male, "
                      f"{(sex=='F').sum()} female -> sire x dam only")
    if not cap["separate_breeding_panel"]:
        rep.fact(sec, "no separate breeding panel: allocation runs within the "
                      "reference population")
    return cap


def stage_ingest(cfg, rep):
    sec = rep.section("0", "INGEST and capability detection")
    hint = ((cfg["pheno_for_geno"], cfg["pheno_for_geno_trait"])
            if cfg.get("pheno_for_geno") else None)
    ids, chrom, pos, snp, H, G = read_genotypes(cfg["geno"], hint)
    phe = pd.read_csv(cfg["pheno"])
    ia, ib = MP.align_ids(ids, phe.iloc[:, 0].values)
    if len(ia) == 0:
        raise SystemExit("genotype ids do not match column 1 of the phenotype "
                         "file; check the id formatting in both")
    rep.fact(sec, f"matched {len(ia)}/{len(ids)} genotyped individuals "
                  f"to phenotypes")
    ids, G, phe = ids[ia], G[ia], phe.iloc[ib].reset_index(drop=True)
    H = H[ia] if H is not None else None

    sex = Gb = ids_b = None
    if cfg.get("breeding_geno"):
        ids_b, chrom_b, pos_b, snp_b, Hb, Gb = read_genotypes(
            cfg["breeding_geno"], hint)
        if len(snp_b) != len(snp) or not np.array_equal(np.asarray(snp_b),
                                                        np.asarray(snp)):
            raise SystemExit(
                "the breeding panel does not carry the same markers, in the "
                "same order, as the reference panel. Cross-panel scoring is a "
                "matrix product over marker columns and returns plausible "
                "numbers even when the columns disagree, so this is refused "
                "rather than warned about.")
        rep.fact(sec, f"breeding panel: {len(ids_b)} individuals, "
                      f"marker set identical")
        if cfg.get("sex"):
            sex = MP.read_sex(cfg["sex"], ids_b)

    missing = [t for t in cfg["traits"] if t not in phe.columns]
    if missing:
        raise SystemExit(f"traits not present in {cfg['pheno']}: {missing}\n"
                         f"  available: {list(phe.columns)}")

    cap = detect_capabilities(G, H, chrom, pos, sex, Gb, rep, sec)
    return dict(ids=ids, G=G, H=H, chrom=chrom, pos=pos, snp=snp, phe=phe,
                ids_b=ids_b, Gb=Gb, sex=sex, cap=cap)


def build_kernels(data, cfg, rep):
    sec = rep.section("0b", "KERNELS")
    G, chrom = data["G"], data["chrom"]
    Ha, Hd = P.noia(G)
    qcfg = Q.CFG()
    qcfg.BFILE = cfg.get("tag", "agent")
    qcfg.CACHE_DIR = cfg["cache"]
    K, meta = Q.build_kernels(qcfg, G, Ha, Hd, chrom, keep_xc=False)
    keep = ["A", "D", cfg["epi_kernel"]]
    K = {k: v for k, v in K.items() if k in keep}
    if cfg["epi_kernel"] not in K:
        raise SystemExit(f"epistatic kernel {cfg['epi_kernel']} was not built")
    n = G.shape[0]
    iu = np.triu_indices(n, 1)
    rows = []
    names = list(K)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            rows.append(dict(kernel_a=a, kernel_b=b,
                             r=float(np.corrcoef(K[a][iu], K[b][iu])[0, 1])))
    CM = pd.DataFrame(rows)
    rep.table(sec, "kernel_collinearity", CM)
    for r in rows:
        if abs(r["r"]) > 0.5:
            rep.warn(sec, f"{r['kernel_a']} and {r['kernel_b']} correlate at "
                          f"{r['r']:+.3f}. Their variance components are not "
                          f"separable; report the prediction delta, never the "
                          f"variance share.")
    labels, how = P.structure_clusters(K["A"], None, 10, cfg["seed"])
    rep.fact(sec, f"structure clustering: {how}")
    return K, meta, Ha, Hd, labels


# ============================================================================
# [1] Calibration -- the gate that makes every later verdict interpretable
# ============================================================================
def stage_calibrate(data, K, labels, cfg, rep):
    sec = rep.section("1", "CALIBRATE modules (false-positive rate and "
                           "detection limit)")
    cap = data["cap"]
    modules = ["epistasis"] + (["dominance"] if cap["dominance_estimable"]
                               else [])
    if not cap["dominance_estimable"]:
        rep.block(sec, "dominance calibration skipped: not estimable in this "
                       "material (see stage 0)")
    sampler = PW.Sampler(K, ["A", "D", cfg["epi_kernel"]])
    mcfg = MP.CFG()
    mcfg.SEED, mcfg.TEST_FRAC = cfg["seed"], cfg["test_frac"]
    levels = cfg["levels"]
    rows = []
    t0 = time.time()
    for module in modules:
        for lvl in levels:
            for r_ in range(cfg["calib_reps"]):
                rng = np.random.default_rng(9000 + r_)
                var_d = lvl if module == "dominance" else 0.0
                var_e = lvl if module == "epistasis" else 0.0
                var_a = max(cfg["h2_sim"] - var_d - var_e, 1e-6)
                y = PW.simulate(sampler, rng, var_a, var_d, var_e,
                                epi_key=cfg["epi_kernel"])
                d, pf = PW.screen_once(y, K, labels, mcfg, cfg["splits"],
                                       module, cfg["boot"],
                                       with_dominance=cap["dominance_estimable"],
                                       epi_key=cfg["epi_kernel"])
                rows.append(dict(module=module, injected=lvl, rep=r_,
                                 dPCC=d, pos_frac=pf))
            MP.say(f"    {module} injected={lvl:.2f} done "
                   f"({(time.time()-t0)/60:.1f} min)")
    R = pd.DataFrame(rows)
    th = cfg["posfrac_threshold"]
    recs = []
    for (module, lvl), g in R.groupby(["module", "injected"]):
        recs.append(dict(module=module, injected=lvl, n_reps=len(g),
                         mean_dPCC=float(g.dPCC.mean()),
                         detection_rate=float(np.mean((g.pos_frac >= th)
                                                      & (g.dPCC > 0)))))
    S = pd.DataFrame(recs).sort_values(["module", "injected"])
    rep.table(sec, "calibration", S)

    calib = {}
    for module, g in S.groupby("module"):
        g = g.sort_values("injected")
        zero = g[g.injected == 0.0]
        fpr = float(zero.detection_rate.iloc[0]) if len(zero) else np.nan
        ok = g[(g.injected > 0) & (g.detection_rate >= cfg["power_target"])]
        limit = float(ok.injected.iloc[0]) if len(ok) else np.nan
        calib[module] = dict(
            false_positive_rate=fpr, detection_limit=limit,
            curve=[dict(injected=float(a_), mean_dPCC=float(b_))
                   for a_, b_ in zip(g.injected, g.mean_dPCC)])
        rep.fact(sec, f"{module}: false-positive rate {fpr:.2f} at threshold "
                      f"{th:.2f}; detection limit "
                      + (f"{limit:.2f} injected variance"
                         if np.isfinite(limit)
                         else f"NOT REACHED within {max(levels):.2f}"),
                 module=module, fpr=fpr, detection_limit=limit)
        if not np.isfinite(limit):
            rep.warn(sec, f"{module}: no injected level reached "
                          f"{cfg['power_target']:.0%} detection. Every negative "
                          f"from this module is uninformative in this dataset; "
                          f"report it as undetectable, not as absent.")
        if np.isfinite(fpr) and fpr > 2 * (1 - cfg["posfrac_threshold"]):
            rep.warn(sec, f"{module}: false-positive rate {fpr:.2f} is high. "
                          f"Raise --posfrac_threshold or treat single-trait "
                          f"positives as provisional.")
    rep.fact(sec, "the generating model matches the tested kernel exactly, so "
                  "these are UPPER BOUNDS on power; real effects will be "
                  "harder to detect than this table implies")
    return calib


def implied_injection(calib, module, dpcc):
    """
    Where an observed delta sits on the calibration curve, as (value,
    in_range). np.interp silently clamps anything past the tested range to
    the boundary y-value -- so a real dPCC of 0.0155 against a curve tested
    only up to 0.0038 would be reported as "equivalent to 0.15 injected
    variance", which understates it. in_range=False signals that the true
    equivalent is beyond what was calibrated, and callers must say so rather
    than print the clamped number as if it were precise.
    """
    c = calib.get(module)
    if not c or not c["curve"]:
        return np.nan, False
    xs = np.array([p["injected"] for p in c["curve"]], float)
    ys = np.array([p["mean_dPCC"] for p in c["curve"]], float)
    o = np.argsort(ys)
    in_range = ys[o][0] <= dpcc <= ys[o][-1]
    return float(np.interp(dpcc, ys[o], xs[o])), in_range


# ============================================================================
# [2] Screen
# ============================================================================
def load_calibration_csv(path, th, power_target, rep):
    """
    Reconstruct the calib dict stage_decide expects from an ALREADY-COMPUTED
    calibration.csv, instead of re-running stage_calibrate. Valid ONLY for
    reusing a calibration computed on the SAME dataset (the false-positive
    rate and detection limit are properties of this data's kernels and sample
    size, not portable across species or datasets).
    """
    import pandas as pd
    df = pd.read_csv(path)
    sec = rep.section("1", "CALIBRATE (reused from a prior run)")
    rep.fact(sec, f"loaded from {path}; NOT recomputed")
    calib = {}
    for module, g in df.groupby("module"):
        g = g.sort_values("injected")
        curve = [dict(injected=float(a), mean_dPCC=float(b))
                for a, b in zip(g.injected, g.mean_dPCC)]
        zero = g[g.injected == 0.0]
        fpr = float(zero.detection_rate.iloc[0]) if len(zero) else np.nan
        ok = g[(g.injected > 0) & (g.detection_rate >= power_target)]
        limit = float(ok.injected.iloc[0]) if len(ok) else np.nan
        calib[module] = dict(false_positive_rate=fpr, detection_limit=limit,
                             curve=curve)
        rep.fact(sec, f"{module}: false-positive rate {fpr:.2f}; detection "
                      f"limit " + (f"{limit:.2f}" if np.isfinite(limit)
                                   else f"NOT REACHED within tested levels"))
    return calib


def stage_screen(data, K, labels, cfg, rep):
    sec = rep.section("2", "SCREEN traits for dominance and epistasis")
    cap = data["cap"]
    mcfg = MP.CFG()
    mcfg.SEED, mcfg.TEST_FRAC = cfg["seed"], cfg["test_frac"]
    rows = []
    for t in cfg["traits"]:
        y = data["phe"][t].values.astype(float)
        keep = np.isfinite(y)
        if keep.sum() < cfg["min_n"]:
            rep.warn(sec, f"{t}: only {int(keep.sum())} phenotyped "
                          f"individuals, below --min_n {cfg['min_n']}; skipped")
            continue
        yk = y[keep]
        Kk = {k: v[np.ix_(np.where(keep)[0], np.where(keep)[0])]
              for k, v in K.items()}
        lab = labels[keep]
        rec = dict(trait=t, n=int(keep.sum()))
        for module, avail in (("dominance", cap["dominance_estimable"]),
                              ("epistasis", True)):
            if not avail:
                rec[f"dPCC_{module}"] = np.nan
                rec[f"posfrac_{module}"] = np.nan
                continue
            d, pf = PW.screen_once(yk, Kk, lab, mcfg, cfg["splits"],
                                   module, cfg["boot"],
                                   with_dominance=cap["dominance_estimable"],
                                   epi_key=cfg["epi_kernel"])
            rec[f"dPCC_{module}"] = d
            rec[f"posfrac_{module}"] = pf
        rows.append(rec)
        MP.say(f"    {t}: dominance dPCC "
               f"{rec.get('dPCC_dominance', float('nan')):+.5f} "
               f"pos {rec.get('posfrac_dominance', float('nan')):.2f} | "
               f"epistasis dPCC {rec['dPCC_epistasis']:+.5f} "
               f"pos {rec['posfrac_epistasis']:.2f}")
    S = pd.DataFrame(rows)
    rep.table(sec, "screen", S)
    rep.fact(sec, f"all traits screened at the SAME {cfg['splits']} splits; "
                  f"the pos_frac threshold is only comparable at a fixed split "
                  f"count, so this must not be varied per trait")
    return S


# ============================================================================
# [3] Decide
# ============================================================================
def stage_decide(screen, calib, cfg, rep):
    sec = rep.section("3", "DECIDE the allocation model per trait")
    th = cfg["posfrac_threshold"]
    rows = []
    for _, r in screen.iterrows():
        verdicts, use = {}, {}
        for module in ("dominance", "epistasis"):
            d, pf = r.get(f"dPCC_{module}"), r.get(f"posfrac_{module}")
            if not np.isfinite(d):
                verdicts[module] = "not estimable in this material"
                use[module] = False
                continue
            passed = (d > 0) and (pf >= th)
            use[module] = bool(passed)
            if passed:
                verdicts[module] = f"detected (dPCC {d:+.5f}, pos {pf:.2f})"
                if calib:
                    imp, in_range = implied_injection(calib, module, d)
                    if np.isfinite(imp) and not in_range:
                        verdicts[module] += (
                            f"; exceeds the calibrated range (curve tested up "
                            f"to dPCC {calib[module]['curve'][-1]['mean_dPCC']:+.4f} "
                            f"at injected variance "
                            f"{calib[module]['curve'][-1]['injected']:.2f} -- "
                            f"this signal is stronger than that, extend "
                            f"--levels to pin an exact equivalent)")
                    elif np.isfinite(imp):
                        verdicts[module] += (f"; equivalent to about "
                                            f"{imp:.3f} injected variance")
            elif calib:
                lim = calib.get(module, {}).get("detection_limit", np.nan)
                imp, _ = implied_injection(calib, module, max(d, 0.0))
                if not np.isfinite(lim):
                    # No injected level reached the power target, so this
                    # module could not have found ANYTHING in this dataset.
                    # A negative here is maximally uninformative -- the
                    # opposite of evidence for absence.
                    verdicts[module] = (
                        "not detected, but this module reached no usable "
                        "detection limit in this dataset (no injected level "
                        "hit the power target). The negative carries NO "
                        "information about whether the effect exists. Do not "
                        "report it as absent; report the dataset as "
                        "underpowered for this module.")
                elif np.isfinite(imp) and imp < lim:
                    verdicts[module] = (f"not detected; the observed delta "
                                        f"corresponds to about {imp:.3f} "
                                        f"injected variance, BELOW the "
                                        f"{lim:.2f} detection limit -- "
                                        f"undetermined, not absent")
                else:
                    verdicts[module] = (f"not detected at or above the "
                                        f"{lim:.2f} detection limit -- "
                                        f"absent for practical purposes")
            else:
                verdicts[module] = "not detected -- UNCALIBRATED, no detection "\
                                   "limit available, so this is not evidence "\
                                   "of absence"
        model = FULL if use["epistasis"] else (
            ADD_DOM if use["dominance"] else ADD)
        rows.append(dict(trait=r["trait"], model=model,
                         dominance_verdict=verdicts["dominance"],
                         epistasis_verdict=verdicts["epistasis"],
                         rule_false_positive_rate=(
                             calib.get("epistasis", {}).get(
                                 "false_positive_rate", np.nan)
                             if calib else np.nan)))
        MP.say(f"    {r['trait']:10s} -> {model}")
        MP.say(f"        dominance: {verdicts['dominance']}")
        MP.say(f"        epistasis: {verdicts['epistasis']}")
    D = pd.DataFrame(rows)
    rep.table(sec, "decisions", D)
    if not calib:
        rep.warn(sec, "stage 1 did not run. Every negative above is "
                      "UNCALIBRATED and must not be reported as evidence that "
                      "an effect is absent. Re-run with --calibrate.")
    return D


# ============================================================================
# [4] Allocate
# ============================================================================
def stage_allocate(data, K, meta, Ha, Hd, decisions, cfg, rep):
    sec = rep.section("4", "ALLOCATE matings")
    cap = data["cap"]
    if cap["separate_breeding_panel"]:
        Gb, ids_b = data["Gb"], data["ids_b"]
    else:
        Gb, ids_b = data["G"], data["ids"]
        rep.fact(sec, "allocating within the reference population")
    sex = data["sex"]
    if sex is None:
        sex = np.array(["M"] * len(ids_b), dtype=object)
        sex[len(ids_b) // 2:] = "F"
        rep.warn(sec, "no sex information; individuals were split arbitrarily "
                      "into two mating groups so that a plan can be produced. "
                      "The resulting plan is structurally valid but the group "
                      "assignment is NOT biological -- supply --sex for a "
                      "usable plan.")
    p_ref = data["G"].mean(0) / 2.0
    KA_b = MP.grm_a_fixed_freq(Gb, p_ref)
    rep.fact(sec, "breeding-panel relationships computed on REFERENCE allele "
                  "frequencies so that progeny F is on the same scale as the "
                  "inbreeding ceiling")

    mcfg = MP.CFG()
    for k, v in (("N_CROSSES", cfg["n_crosses"]),
                 ("MAX_USE_SIRE", cfg["max_use_sire"]),
                 ("MAX_USE_DAM", cfg["max_use_dam"]),
                 ("F_MAX", cfg["f_max"]), ("SEED", cfg["seed"])):
        setattr(mcfg, k, v)

    out = []
    for _, d in decisions.iterrows():
        t = d["trait"]
        MP.say(f"\n    --- {t} (model {d['model']}) ---")
        gm, n_used = MP.fit_reference_model(mcfg, data, K, meta, Ha, Hd, t)
        epi_fast = MP.EpiFast(gm, Gb) if gm.use_epi else None
        _, summ = MP.mate_allocation(mcfg, gm, Gb, ids_b, sex, KA_b,
                                     rep.out_dir, t, epi_fast=epi_fast)
        summ = summ.copy()
        summ.insert(1, "chosen_model", d["model"])
        out.append(summ)
    if not out:
        rep.warn(sec, "no trait reached allocation")
        return None
    A = pd.concat(out, ignore_index=True)
    rep.table(sec, "allocation_summary", A)
    for _, r in A[A.strategy == A.chosen_model].iterrows():
        ov = r["overlap_with_ADD"]
        rep.fact(sec, f"{r['trait']}: the chosen model shares {ov:.0%} of its "
                      f"crosses with the additive plan"
                      + ("; the non-additive terms changed nothing, so "
                         "additive allocation is sufficient for this trait"
                         if ov >= 0.999 else ""),
                 trait=r["trait"], overlap_with_ADD=float(ov))
    return A


# ============================================================================
# Multi-trait -- declared, not implemented
# ============================================================================
def stage_allocate_multitrait(*_a, **_k):
    raise NotImplementedError(
        "Multi-trait allocation is not implemented.\n"
        "  What exists: per-trait weights are parsed and validated at stage 0, "
        "so the input contract is fixed.\n"
        "  What does not exist: an index combining traits whose chosen models "
        "differ. A trait allocated on FULL and a trait allocated on ADD do not "
        "share a scale, and the covariance between their non-additive "
        "components is not estimated anywhere in this pipeline. Producing a "
        "weighted sum of them would return a number with no defined meaning.\n"
        "  This is raised rather than approximated on purpose: an earlier "
        "version of this project accepted stage names it did not implement and "
        "silently ran something else.")


def load_weights(path, traits):
    """Weights are validated early so a multi-trait run cannot start half-set."""
    if not path:
        return None
    w = (json.load(open(path)) if str(path).endswith(".json")
         else dict(zip(*pd.read_csv(path).iloc[:, :2].values.T)))
    w = {str(k): float(v) for k, v in w.items()}
    missing = [t for t in traits if t not in w]
    extra = [k for k in w if k not in traits]
    if missing:
        raise SystemExit(f"--weights is missing entries for {missing}")
    if extra:
        raise SystemExit(f"--weights has entries for traits not requested: "
                         f"{extra}")
    if all(abs(v) < 1e-12 for v in w.values()):
        raise SystemExit("--weights are all zero")
    return w


# ============================================================================
# Driver
# ============================================================================
DEFAULTS = dict(
    geno=None, pheno=None, breeding_geno=None, sex=None, weights=None,
    traits=None, out="agent_out", cache="agent_kernel_cache", tag="agent",
    seed=2024, test_frac=0.20, splits=30, boot=400, min_n=100,
    epi_kernel="MULT_t", posfrac_threshold=0.80, power_target=0.80,
    pheno_for_geno=None, pheno_for_geno_trait=None, calib_csv=None,
    calibrate=False, levels=[0.0, 0.02, 0.05, 0.10, 0.20], calib_reps=10,
    h2_sim=0.50, n_crosses=60, max_use_sire=3, max_use_dam=2, f_max=0.0625,
    allocate=True, multitrait=False, manifest_only=False,
    cm_per_mb=None, step1_npz=None, f1_pheno=None,
    n_screen=500, off_per_cross=5, max_use_per_parent=5,
    inbreed_cap=0.10, n_recommend=50, n_gametes=200,
)


def run(cfg):
    rep = Report(cfg["out"])
    t0 = time.time()
    man, ok = MAN.build_manifest(cfg, MP.say, cfg["out"])
    if not ok:
        raise SystemExit("run blocked by the manifest above; supply the "
                         "refused inputs or change the requested path")
    if cfg.get("manifest_only"):
        MP.say("\n--manifest_only: stopping before any computation")
        return rep
    data = stage_ingest(cfg, rep)
    load_weights(cfg.get("weights"), cfg["traits"])
    K, meta, Ha, Hd, labels = build_kernels(data, cfg, rep)
    if cfg.get("calib_csv"):
        calib = load_calibration_csv(cfg["calib_csv"], cfg["posfrac_threshold"],
                                     cfg["power_target"], rep)
    else:
        calib = (stage_calibrate(data, K, labels, cfg, rep)
                 if cfg["calibrate"] else None)
    screen = stage_screen(data, K, labels, cfg, rep)
    decisions = stage_decide(screen, calib, cfg, rep)
    if cfg["multitrait"]:
        stage_allocate_multitrait()
    if cfg["allocate"]:
        stage_allocate(data, K, meta, Ha, Hd, decisions, cfg, rep)
    sec = rep.section("5", "SUMMARY")
    rep.fact(sec, f"completed in {(time.time()-t0)/60:.1f} min")
    rep.fact(sec, f"outputs written to {cfg['out']}/")
    rep.save()
    return rep


def selftest():
    """Three species profiles, exercising the capability branches."""
    import shutil
    ok = True
    tmp = "/tmp/agent_st"
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    rng = np.random.default_rng(5)
    n, nb, m, nc = 180, 40, 600, 6
    chrom = np.repeat(np.arange(1, nc + 1), m // nc)
    m = len(chrom)
    pos = np.sort(rng.integers(1, 6e7, m))
    freq = rng.uniform(0.2, 0.8, m)

    def vcf(path, ids, inbred=False, phased=True):
        with open(path, "w") as f:
            f.write("##fileformat=VCFv4.2\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\t"
                    "FILTER\tINFO\tFORMAT\t" + "\t".join(ids) + "\n")
            for k in range(m):
                if inbred:
                    a = (rng.random(len(ids)) < freq[k]).astype(int)
                    h = np.stack([a, a], 1)
                else:
                    h = (rng.random((len(ids), 2)) < freq[k]).astype(int)
                sep = "|" if phased else "/"
                f.write(f"{chrom[k]}\t{pos[k]}\tsnp{k}\tA\tG\t.\t.\t.\tGT\t"
                        + "\t".join(f"{x}{sep}{y}" for x, y in h) + "\n")

    rid = [f"R{i}" for i in range(n)]
    bid = [f"B{i}" for i in range(nb)]
    vcf(f"{tmp}/ref.vcf", rid)
    vcf(f"{tmp}/breed.vcf", nb and bid)
    vcf(f"{tmp}/inbred.vcf", rid, inbred=True)
    vcf(f"{tmp}/unphased.vcf", rid, phased=False)
    for nm, ids_ in (("pheno", rid),):
        pd.DataFrame({"id": ids_, "t1": rng.normal(0, 1, len(ids_)),
                      "t2": rng.normal(0, 1, len(ids_))}
                     ).to_csv(f"{tmp}/{nm}.csv", index=False)
    pd.DataFrame({"id": bid, "sex": ["M"] * (nb // 2) + ["F"] * (nb // 2)}
                 ).to_csv(f"{tmp}/sex.csv", index=False)

    def base(**kw):
        c = dict(DEFAULTS)
        c.update(pheno=f"{tmp}/pheno.csv", traits=["t1"], splits=6,
                 boot=100, min_n=50, n_crosses=8, calibrate=False,
                 cache=f"{tmp}/ck", allocate=True)
        c.update(kw)
        return c

    MP.say("\n########## [1] outbred + sexed + separate breeding panel")
    r1 = run(base(geno=f"{tmp}/ref.vcf", breeding_geno=f"{tmp}/breed.vcf",
                  sex=f"{tmp}/sex.csv", out=f"{tmp}/o1", tag="a1"))
    caps = [s for s in r1.sections if s["key"] == "0"][0]
    good = not caps["blocks"]
    MP.say(f"    no capabilities blocked: {'OK' if good else 'FAIL'}")
    ok &= good
    d1 = pd.read_csv(f"{tmp}/o1/decisions.csv")
    good = "not detected" in str(d1.epistasis_verdict.iloc[0]) and \
           "UNCALIBRATED" in str(d1.epistasis_verdict.iloc[0])
    MP.say(f"    uncalibrated negatives are labelled as such: "
           f"{'OK' if good else 'FAIL'}")
    ok &= good

    MP.say("\n########## [2] inbred material: dominance must be DISABLED")
    r2 = run(base(geno=f"{tmp}/inbred.vcf", out=f"{tmp}/o2", tag="a2",
                  allocate=False))
    caps = [s for s in r2.sections if s["key"] == "0"][0]
    good = any("dominance module" in b for b in caps["blocks"])
    MP.say(f"    dominance blocked with a reason: {'OK' if good else 'FAIL'}")
    ok &= good
    d2 = pd.read_csv(f"{tmp}/o2/decisions.csv")
    good = "not estimable" in str(d2.dominance_verdict.iloc[0]) and \
           d2.model.iloc[0] in (ADD, FULL)
    MP.say(f"    inbred trait never routed to ADD+DOM: "
           f"{'OK' if good else 'FAIL'}")
    ok &= good

    MP.say("\n########## [3] unphased genotypes: gamete stages refused")
    r3 = run(base(geno=f"{tmp}/unphased.vcf", out=f"{tmp}/o3", tag="a3",
                  allocate=False))
    caps = [s for s in r3.sections if s["key"] == "0"][0]
    good = any("unphased" in b for b in caps["blocks"])
    MP.say(f"    unphased reported as a blocked capability: "
           f"{'OK' if good else 'FAIL'}")
    ok &= good

    MP.say("\n########## [4] multi-trait must refuse, not approximate")
    try:
        stage_allocate_multitrait()
        MP.say("    FAIL: multi-trait returned instead of raising")
        ok = False
    except NotImplementedError as e:
        good = "not implemented" in str(e).lower()
        MP.say(f"    raises with an explanation: {'OK' if good else 'FAIL'}")
        ok &= good

    MP.say("\n########## [5] weight validation catches partial specs")
    json.dump({"t1": 1.0}, open(f"{tmp}/w.json", "w"))
    try:
        load_weights(f"{tmp}/w.json", ["t1", "t2"])
        MP.say("    FAIL: missing weight accepted")
        ok = False
    except SystemExit as e:
        good = "missing" in str(e)
        MP.say(f"    missing weight refused: {'OK' if good else 'FAIL'}")
        ok &= good

    MP.say("\n########## [6] marker-set mismatch between panels refused")
    with open(f"{tmp}/breed.vcf") as fi, open(f"{tmp}/breed_bad.vcf", "w") as fo:
        for i, line in enumerate(fi):
            fo.write(line if i != 40 else line.replace("snp38", "snpZZZ", 1))
    try:
        run(base(geno=f"{tmp}/ref.vcf", breeding_geno=f"{tmp}/breed_bad.vcf",
                 sex=f"{tmp}/sex.csv", out=f"{tmp}/o6", tag="a6"))
        MP.say("    FAIL: mismatched marker set accepted")
        ok = False
    except SystemExit as e:
        good = "same markers" in str(e)
        MP.say(f"    refused: {'OK' if good else 'FAIL'}")
        ok &= good

    MP.say("\nSELF-TEST: " + ("PASS" if ok else "FAIL"))
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--selftest", action="store_true")
    for k, v in DEFAULTS.items():
        if isinstance(v, bool):
            ap.add_argument(f"--{k}", action="store_true", default=None)
        elif isinstance(v, list):
            ap.add_argument(f"--{k}", nargs="*", type=float)
        elif k == "traits":
            ap.add_argument("--traits", nargs="*")
        elif k == "pheno_for_geno_trait":
            ap.add_argument("--pheno_for_geno_trait")
        elif k == "pheno_for_geno":
            ap.add_argument("--pheno_for_geno")
        elif k == "calib_csv":
            ap.add_argument("--calib_csv")
        elif isinstance(v, int) and not isinstance(v, bool):
            ap.add_argument(f"--{k}", type=int)
        elif isinstance(v, float):
            ap.add_argument(f"--{k}", type=float)
        else:
            ap.add_argument(f"--{k}")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if selftest() else 1)
    cfg = dict(DEFAULTS)
    if a.config:
        cfg.update(json.load(open(a.config)))
    for k in DEFAULTS:
        v = getattr(a, k, None)
        if v is not None:
            cfg[k] = v
    for need in ("geno", "pheno", "traits"):
        if not cfg.get(need):
            raise SystemExit(f"--{need} is required (or supply --config)")
    run(cfg)


if __name__ == "__main__":
    main()
