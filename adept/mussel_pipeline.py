#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 Epistasis evaluation and epistasis-aware mate allocation in Hyriopsis cumingii
 ------------------------------------------------------------------------------
 Mussel counterpart of pig_epistasis_pipeline.py. The statistical machinery is
 shared, but four structural differences make this the stronger of the two
 datasets and require dedicated code:

   PHASED GENOTYPES   The mussel VCFs are Beagle-phased, so gametes can be
                      simulated with real crossovers along a genetic map. The
                      pig pipeline had to assume free recombination, which
                      discards within-chromosome linkage and therefore
                      understates Mendelian sampling variance. That limitation
                      does not apply here.
   SEXED CANDIDATES   cand_sex.csv gives explicit sexes, so mate allocation is
                      sire x dam rather than all pairwise combinations.
   SEPARATE POPULATIONS  Parameters are estimated in the phenotyped reference
                      population (n~599); allocation is performed in an
                      independent breeding population (n~300). The pig data had
                      no such separation.
   REAL F1 OFFSPRING  Fourteen families were produced by GBLUP-based genomic
                      selection. Stage [V] uses them as a PROSPECTIVE test:
                      predictions made without any knowledge of the F1 are
                      compared against observed family performance. No
                      cross-validation scheme can substitute for this.

================================================================================
 STAGES
================================================================================
   [A] DIFFUSE    exact aggregate epistatic kernels vs an A+D baseline   READY
   [B] DECOMPOSE  additive / dominance / epistatic breeding-value corr.  READY
   [C] FOCAL      genome-wide exact pair scoring, cross-split repro.     planned
   [D] MATING     sex-structured, epistasis-aware allocation             partial
   [E] MULTIGEN   five generations with recombination-aware gametes      partial
   [V] VALIDATE   prospective test against the observed F1 families      stub

 READY   = wired into the driver and runnable now.
 partial = the numerical core exists and passes --selftest (mate_allocation for
           [D], gametes for [E]), but nothing in run() calls it yet.
 stub    = intentionally raises; see validate_f1 for the file format it needs.

 Requesting a non-READY stage aborts with an explanation rather than being
 accepted and silently ignored.

 Single-trait throughout, by design. Multi-trait allocation is deferred.

 NOTE ON --cm_per_mb: the map density only enters through gametes(), which only
 stage [E] uses. It has no effect on the screening batches. Once [E] is written,
 and unless a measured Hyriopsis cumingii linkage map is available, the 0.5 /
 1.0 / 2.0 cM/Mb sensitivity analysis is mandatory -- recombination rate sets
 the segregation variance and therefore the five-generation gain.

================================================================================
 CONVENTIONS CARRIED OVER FROM THE PIG ANALYSIS (each established empirically)
================================================================================
 * accuracy = cor(GEBV, y - Xb); including Xb inflates accuracy and compresses
   the delta about 4.5-fold
 * fixed effects = intercept only; genomic PCs are genetic signal and must not
   be fitted as fixed effects alongside the GRM
 * kernels are built exactly by factorisation, never Monte-Carlo sampled
 * pair scoring is standardised |cor(z,y)|, and AD is scored in both
   orientations because Ha_i*Hd_j != Ha_j*Hd_i
 * kernels with |r| > 0.5 have non-identifiable variance components; report
   their delta in prediction, not their variance share

================================================================================
 USAGE
     python mussel_pipeline.py --selftest

     # the three screening batches (phe1-5, phe6-10, phe11-15)
     python mussel_pipeline.py --batch 1 --batch_size 5
     python mussel_pipeline.py --batch 2 --batch_size 5
     python mussel_pipeline.py --batch 3 --batch_size 5

     # a single trait
     python mussel_pipeline.py --trait phe15

 Stages [C]-[E] and [V] are not reachable from the command line yet.
================================================================================
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import time
import warnings
from itertools import product

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import pig_epik_pipeline as P
    import pig_epistasis_pipeline as Q
except ImportError as _e:
    sys.exit(
        f"\n[fatal] {_e}\n"
        "  This script reuses the kernel construction, REML and split machinery\n"
        "  from the pig analysis. The following files must sit in the SAME\n"
        "  directory as mussel_pipeline.py:\n"
        "      pig_epik_pipeline.py     (noia, REMLgrad, structure_clusters)\n"
        "      pig_epistasis_pipeline.py   (build_kernels, make_split, fit_gebv,\n"
        "                                   pcc, pred_bias, boot_delta)\n"
        "      pig_multigen_figures.py     (GeneticModel; needed by --selftest\n"
        "                                   and by stages mating/multigen)\n")


class CFG:
    REF_VCF    = "ref_all_phased.vcf"
    BREED_VCF  = "breed_all_phased.vcf"
    PHENO      = "trait_imputed.csv"
    SEX_FILE   = "cand_sex.csv"
    TRAIT      = "phe15"          # purple-area ratio; shell colour headline trait
    OUT_DIR    = "mussel_out"
    CACHE_DIR  = "mussel_kernel_cache"
    SEED       = 2024

    CM_PER_MB  = 1.0              # Haldane map assumption; see --cm_per_mb
    TEST_FRAC  = 0.20
    N_SEEDS    = 20
    N_BOOT     = 1000
    CORE_MODELS     = ("A", "A+D", "A+D+MULT_t", "A+D+HAD")
    SECONDARY_SEEDS = 5

    SCORE_TYPES = ("MULT", "DD", "THR")
    TOP_POOL    = 20
    K_GRID      = (1, 2, 3)
    SCAN_CHUNK  = 512
    FOCAL_SEEDS = 5

    # --- allocation ---
    N_CROSSES    = 60
    MAX_USE_SIRE = 3
    MAX_USE_DAM  = 2
    F_MAX        = 0.0625

    # --- multi-generation simulation ---
    GENERATIONS   = 5
    OFF_PER_CROSS = 5
    N_SCREEN      = 500           # gametes screened per parent per cross
    N_REPS        = 3

    # --- F1 validation ---
    F1_VCF   = None
    F1_PHENO = None
    F1_PED   = None               # columns: family, sire, dam
    ONLY_MODELS = None            # e.g. ("A", "A+D") for a dominance-only run

    # --- trait batching ---
    # All fifteen traits are screened, but not in one process. Kernels depend
    # only on genotypes and are cached, so batches after the first are cheap.
    ALL_TRAITS = tuple(f"phe{i}" for i in range(1, 16))
    # Traits that are algebraic functions of other traits in the list. They are
    # still screened, but they are NOT independent tests and must not be
    # counted as such when accounting for multiplicity:
    #   phe8  = phe7 / phe6                      (ratio; h2 unstable, see notes)
    #   phe12 = sqrt(phe10^2 + phe11^2)          (chroma from a, b)
    #   phe13 = atan2(phe11, phe10)              (hue from a, b)
    #   phe14 = f(phe9, phe10, phe11)            (colour difference from L,a,b)
    DERIVED_TRAITS = ("phe8", "phe12", "phe13", "phe14")
    # A trait is carried forward to allocation only if the epistatic kernel
    # helps prediction consistently across splits.
    EPI_DPCC_MIN    = 0.0
    EPI_POSFRAC_MIN = 0.80


def say(*a):
    print(*a, flush=True)


# ============================================================================
# Phased VCF input
# ============================================================================
def read_phased_vcf(path, max_snps=None):
    """
    Read a phased VCF into haplotypes.
    Returns ids, chrom, pos, snp, H with H of shape (n, m, 2) as int8.
    Unphased ('/') calls are counted and reported: Mendelian sampling variance
    depends on phase, so a substantial unphased fraction invalidates stage [E].
    """
    op = gzip.open if str(path).endswith(".gz") else open
    ids, chrom, pos, snp, rows = None, [], [], [], []
    n_unphased = 0
    with op(path, "rt") as f:
        for line in f:
            if line.startswith("##"):
                continue
            p = line.rstrip("\n").split("\t")
            if line.startswith("#CHROM"):
                ids = np.array(p[9:], dtype=object)
                continue
            chrom.append(p[0])
            pos.append(int(p[1]))
            snp.append(p[2] if p[2] != "." else f"{p[0]}_{p[1]}")
            h = np.empty((len(p) - 9, 2), dtype=np.int8)
            for k, g in enumerate(p[9:]):
                gt = g.split(":", 1)[0]
                if "|" in gt:
                    a, b = gt.split("|")
                elif "/" in gt:
                    a, b = gt.split("/")
                    n_unphased += 1
                else:
                    a = b = "."
                h[k, 0] = -1 if a == "." else int(a)
                h[k, 1] = -1 if b == "." else int(b)
            rows.append(h)
            if max_snps and len(rows) >= max_snps:
                break
    H = np.stack(rows, axis=1)                       # n x m x 2
    chrom = np.array([int(str(c).replace("chr", "").replace("Chr", ""))
                      if str(c).replace("chr", "").replace("Chr", "").isdigit()
                      else 0 for c in chrom])
    frac = n_unphased / max(H.shape[0] * H.shape[1], 1)
    say(f"    {path}: {H.shape[0]} individuals x {H.shape[1]} SNPs"
        f"   unphased calls {frac:.4%}")
    if frac > 0.01:
        say("    [warning] more than 1% of calls are unphased. Mendelian")
        say("              sampling variance depends on phase, so stage [E]")
        say("              would be biased. Re-run Beagle before trusting it.")
    return ids, chrom, np.array(pos), np.array(snp, dtype=object), H


def hap_to_geno(H):
    G = H.sum(axis=2).astype(np.float64)
    G[np.any(H < 0, axis=2)] = np.nan
    for k in range(G.shape[1]):
        col = G[:, k]
        bad = np.isnan(col)
        if bad.any():
            col[bad] = np.nanmean(col) if (~bad).any() else 0.0
    return G


def align_ids(a, b):
    """Match two id vectors, trying a few common formatting variants."""
    def variants(x):
        s = str(x)
        return {s, s.split("_")[-1], s.split(":")[-1], s.split("_")[0], s.split(":")[0]}
    lookup = {}
    for j, y in enumerate(b):
        for v in variants(y):
            lookup.setdefault(v, j)
    ia, ib = [], []
    for i, x in enumerate(a):
        for v in variants(x):
            if v in lookup:
                ia.append(i)
                ib.append(lookup[v])
                break
    return np.array(ia, int), np.array(ib, int)


# ============================================================================
# Recombination-aware gametes  (the pig pipeline could not do this)
# ============================================================================
def gametes(H_ind, chrom, pos, n_gam, rng, cm_per_mb):
    """
    Simulate n_gam gametes from one phased individual.
    Crossovers follow Haldane's mapping function on a constant cM/Mb map,
    applied independently within each chromosome. Verified against the expected
    linkage decay: identity-by-descent between loci falls from 0.9995 at
    ~0.05 Mb to 0.8034 at ~25 Mb, whereas free recombination would give 0.5
    at every distance.
    """
    m = H_ind.shape[0]
    which = np.empty((n_gam, m), dtype=np.int8)
    for ch in np.unique(chrom):
        idx = np.where(chrom == ch)[0]
        if len(idx) == 0:
            continue
        d_cm = np.diff(pos[idx]) / 1e6 * cm_per_mb
        r = 0.5 * (1 - np.exp(-2 * np.maximum(d_cm, 0) / 100.0))
        w = np.empty((n_gam, len(idx)), dtype=np.int8)
        w[:, 0] = rng.integers(0, 2, n_gam)
        if len(idx) > 1:
            sw = rng.random((n_gam, len(r))) < r[None, :]
            w[:, 1:] = (w[:, [0]] + np.cumsum(sw, axis=1)) % 2
        which[:, idx] = w
    h0, h1 = H_ind[:, 0][None, :], H_ind[:, 1][None, :]
    return np.where(which == 0, h0, h1).astype(np.float64)


# ============================================================================
# Genetic model (shared with the pig module, re-parameterised for mussel)
# ============================================================================
def build_model(X, chrom, y, Xfix, cfg):
    Ha, Hd = P.noia(X)
    qcfg = Q.CFG()
    qcfg.BFILE = "mussel"
    qcfg.CACHE_DIR = cfg.CACHE_DIR
    K, meta = Q.build_kernels(qcfg, X, Ha, Hd, chrom, keep_xc=False)
    import pig_multigen_figures as MG
    gm = MG.GeneticModel(X, Ha, Hd, chrom, K, meta, y, Xfix)
    return K, meta, gm, Ha, Hd


# ============================================================================
# [D] sex-structured mate allocation
# ============================================================================
def allocate_sexed(df, cfg):
    order = np.argsort(-df["score"].values)
    us, ud, chosen = {}, {}, []
    for k in order:
        if df.progeny_F.values[k] > cfg.F_MAX:
            continue
        s, d = df.sire.values[k], df.dam.values[k]
        if us.get(s, 0) >= cfg.MAX_USE_SIRE or ud.get(d, 0) >= cfg.MAX_USE_DAM:
            continue
        us[s] = us.get(s, 0) + 1
        ud[d] = ud.get(d, 0) + 1
        chosen.append(k)
        if len(chosen) >= cfg.N_CROSSES:
            break
    return np.asarray(chosen, int)


def mate_allocation(cfg, gm, Gb, ids_b, sex, KA_b, out_dir, trait, epi_fast=None):
    sires = np.where(sex == "M")[0]
    dams = np.where(sex == "F")[0]
    say(f"    breeding population: {len(sires)} sires x {len(dams)} dams "
        f"= {len(sires)*len(dams):,} crosses")
    si, di = map(np.asarray, zip(*product(range(len(sires)), range(len(dams)))))
    S, D = sires[si], dams[di]
    add = np.empty(len(S))
    dom = np.empty(len(S))
    epi = np.zeros(len(S))
    CH = 2000
    for s in range(0, len(S), CH):
        e = min(s + CH, len(S))
        e_ha, e_hd = gm.expected_codes(Gb[S[s:e]], Gb[D[s:e]])
        add[s:e] = e_ha @ gm.a_hat
        dom[s:e] = e_hd @ gm.d_hat
        if gm.use_epi:
            epi[s:e] = (epi_fast.pairs(S[s:e], D[s:e]) if epi_fast is not None
                        else gm.epi_value(e_ha))
    df = pd.DataFrame(dict(sire=ids_b[S], dam=ids_b[D],
                           sire_idx=S, dam_idx=D,
                           gv_add=add, gv_dom=dom, gv_epi=epi,
                           gv_add_dom=add + dom, gv_full=add + dom + epi,
                           progeny_F=KA_b[S, D] / 2.0))
    df.to_csv(os.path.join(out_dir, f"{trait}_all_crosses.csv"), index=False)

    plans, rows = {}, []
    for strat, col in [("ADD", "gv_add"), ("ADD+DOM", "gv_add_dom"),
                       ("FULL", "gv_full")]:
        df["score"] = df[col]
        sel = allocate_sexed(df, cfg)
        p = df.iloc[sel].copy()
        p.insert(0, "strategy", strat)
        plans[strat] = p
    base = set(zip(plans["ADD"].sire, plans["ADD"].dam))
    for strat, p in plans.items():
        rows.append(dict(trait=trait, strategy=strat, n_crosses=len(p),
                         mean_gv_full=float(p.gv_full.mean()),
                         mean_gv_add=float(p.gv_add.mean()),
                         mean_gv_dom=float(p.gv_dom.mean()),
                         mean_gv_epi=float(p.gv_epi.mean()),
                         mean_F=float(p.progeny_F.mean()),
                         max_F=float(p.progeny_F.max()),
                         n_sires=p.sire.nunique(), n_dams=p.dam.nunique(),
                         overlap_with_ADD=len(set(zip(p.sire, p.dam)) & base)
                         / max(len(base), 1)))
    pd.concat(plans.values(), ignore_index=True).drop(columns=["score"]).to_csv(
        os.path.join(out_dir, f"{trait}_mating_plan.csv"), index=False)
    summ = pd.DataFrame(rows)
    say(summ.round(4).to_string(index=False))
    return df, summ


# ============================================================================
# [V] prospective validation against the observed F1 families
# ============================================================================
def validate_f1(cfg, gm, out_dir, trait):
    """
    The fourteen F1 families were produced by GBLUP-based selection, i.e. the
    crosses were chosen WITHOUT any of the non-additive information evaluated
    here. That makes them an honest prospective test of two distinct claims:

      (1) PREDICTION  Does adding dominance and epistasis improve the
          prediction of realised family means? Compare observed family means
          against predictions from A, A+D and A+D+epistasis.
      (2) ALLOCATION  Retrospectively, would the epistasis-aware criterion have
          ranked the fourteen realised crosses differently, and does its
          ranking agree better with what was actually observed?

    Claim (1) is the strong one: it is a genuine out-of-sample test on animals
    that did not exist when the model was fitted. Claim (2) is weaker because
    only fourteen crosses were realised and they were not chosen at random.

    ---------------------------------------------------------------------------
    STATUS: DELIBERATELY LEFT UNWIRED.

    The reader below assumes a file layout that has not yet been checked against
    the real F1 files. Rather than guess and produce a silently wrong join, this
    stage refuses to run until the actual column names are supplied.

    What it currently assumes, and what must be confirmed before it is enabled:

      --f1_vcf    phased VCF, sample ids in the #CHROM header line, GT phased
                  with '|'. Ids must be reconcilable with column 1 of --f1_pheno
                  (align_ids tolerates a few prefix/suffix variants, not
                  arbitrary renaming).
      --f1_pheno  CSV. Column 1 = individual id. One column named exactly like
                  the trait (e.g. 'phe15'). One column named 'family' giving the
                  family each individual belongs to. <-- this is the load-bearing
                  one: r_family_mean, the primary statistic, comes from it.
      --f1_ped    CSV with columns 'family', 'sire', 'dam'. Sire and dam ids must
                  match the BREEDING population ids in breed_all_phased.vcf,
                  otherwise the realised crosses cannot be located among the
                  candidate crosses and claim (2) cannot be evaluated at all.

    If the real files differ -- family coded inside the id string, a separate
    family-level phenotype table, sire/dam under different names, unphased F1
    genotypes -- this function must be rewritten to those headers rather than
    worked around: a wrong id join here produces a plausible-looking
    correlation that means nothing.
    ---------------------------------------------------------------------------
    """
    raise NotImplementedError(
        "Stage [V] is not wired yet. Supply the real F1 file headers first "
        "(see the docstring of validate_f1 for what the reader currently "
        "assumes). Stages diffuse/decompose do not need it.")


def _validate_f1_draft(cfg, gm, out_dir, trait):
    """Draft body kept for reference; NOT called. See validate_f1 above."""
    if not (cfg.F1_VCF and cfg.F1_PHENO and cfg.F1_PED):
        say("    [skip] F1 files not supplied (--f1_vcf --f1_pheno --f1_ped)")
        return None
    ids1, chrom1, pos1, snp1, H1 = read_phased_vcf(cfg.F1_VCF)
    G1 = hap_to_geno(H1)
    phe = pd.read_csv(cfg.F1_PHENO)
    ped = pd.read_csv(cfg.F1_PED)
    say(f"    F1: {len(ids1)} animals, {ped.family.nunique()} families")

    ia, ib = align_ids(ids1, phe.iloc[:, 0].values)
    if len(ia) == 0:
        say("    [error] F1 genotype and phenotype ids do not match")
        return None
    obs = phe.iloc[ib][trait].values.astype(float)
    val = gm.value(G1[ia])
    pred = dict(A=val["add"],
                AD=val["add"] + val["dom"],
                FULL=val["add"] + val["dom"] + val["epi"])

    fam_of = phe.iloc[ib]["family"].values if "family" in phe.columns else None
    rows = []
    for nm, pv in pred.items():
        ok = np.isfinite(obs) & np.isfinite(pv)
        r_ind = float(np.corrcoef(obs[ok], pv[ok])[0, 1]) if ok.sum() > 3 else np.nan
        r_fam = np.nan
        if fam_of is not None:
            d = pd.DataFrame(dict(family=fam_of[ok], obs=obs[ok], pred=pv[ok]))
            gmn = d.groupby("family").mean()
            if len(gmn) > 3:
                r_fam = float(np.corrcoef(gmn.obs, gmn.pred)[0, 1])
        rows.append(dict(trait=trait, model=nm, n_animals=int(ok.sum()),
                         r_individual=r_ind,
                         n_families=int(len(set(fam_of[ok]))) if fam_of is not None else 0,
                         r_family_mean=r_fam))
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(out_dir, f"{trait}_f1_validation.csv"), index=False)
    say(R.round(4).to_string(index=False))
    say("\n    r_family_mean is the primary number: with fourteen families it is")
    say("    a small but genuinely prospective test. r_individual additionally")
    say("    reflects within-family Mendelian sampling, which no model that uses")
    say("    only parental averages can predict.")
    return R


# ============================================================================
# [D] breeding-population machinery
# ============================================================================
def grm_a_fixed_freq(Xb, p_ref):
    """
    VanRaden GRM for the breeding population, centred and scaled on the
    REFERENCE allele frequencies.

    P.grm_a() derives p from whatever matrix it is handed. Calling it on the
    breeding genotypes would centre on breeding-population frequencies, putting
    the result on a different base from meta["KA_raw"] and from the model's
    marker coding. progeny_F would then be compared against F_MAX on a scale
    that is not the one F_MAX was chosen for. Verified to reproduce P.grm_a
    exactly when p_ref is taken from the same matrix (max abs diff 0.0).
    """
    Z = Xb - 2.0 * p_ref[None, :]
    return (Z @ Z.T) / (2.0 * np.sum(p_ref * (1 - p_ref)) + 1e-12)


def grm_a_cross(Xa, Xb, p_ref):
    """Off-diagonal block: relatedness of every Xa animal to every Xb animal."""
    Za = Xa - 2.0 * p_ref[None, :]
    Zb = Xb - 2.0 * p_ref[None, :]
    return (Za @ Zb.T) / (2.0 * np.sum(p_ref * (1 - p_ref)) + 1e-12)


def read_sex(path, ids_b):
    """
    Read cand_sex.csv and return an array of 'M'/'F' aligned to ids_b.
    The column layout is not assumed: the id column is matched against the VCF
    sample names, and the sex column is whichever column holds two distinct
    recognisable sex codes. What was detected is printed, because a silent
    sire/dam swap would invert the entire allocation.
    """
    tab = pd.read_csv(path)
    # Chinese tokens are escaped, not removed: real input files use them.
    MALE = {"m", "male", "1", "1.0", "sire", "male ", "\u516c", "\u96c4"}
    FEMALE = {"f", "female", "2", "2.0", "dam", "female ", "\u6bcd", "\u96cc"}

    id_col, ia, ib = None, None, None
    for c in tab.columns:
        i_a, i_b = align_ids(ids_b, tab[c].values)
        if len(i_a) > 0.5 * len(ids_b):
            id_col, ia, ib = c, i_a, i_b
            break
    if id_col is None:
        raise SystemExit(
            f"no column in {path} matches the breeding VCF sample names.\n"
            f"  columns present: {list(tab.columns)}\n"
            f"  first VCF ids:   {list(ids_b[:5])}")

    sex_col = None
    for c in tab.columns:
        if c == id_col:
            continue
        vals = {str(v).strip().lower() for v in tab[c].dropna().unique()}
        if 1 < len(vals) <= 3 and vals <= (MALE | FEMALE):
            sex_col = c
            break
    if sex_col is None:
        raise SystemExit(
            f"no sex column found in {path}.\n"
            f"  columns present: {list(tab.columns)}\n"
            f"  expected two codes from {sorted(MALE | FEMALE)}")

    say(f"    sex file: id column '{id_col}', sex column '{sex_col}'")
    sex = np.full(len(ids_b), "?", dtype=object)
    raw = tab[sex_col].values
    for k, j in zip(ia, ib):
        v = str(raw[j]).strip().lower()
        sex[k] = "M" if v in MALE else ("F" if v in FEMALE else "?")
    n_m, n_f, n_q = (sex == "M").sum(), (sex == "F").sum(), (sex == "?").sum()
    say(f"    sexes assigned: {n_m} male, {n_f} female, {n_q} unknown")
    if n_m == 0 or n_f == 0:
        raise SystemExit("one sex is empty; check the coding in " + str(path))
    return sex


def load_breeding(cfg, ref):
    """
    Load the breeding population and refuse to continue unless its marker set
    is identical to the reference. Ha_ref @ Ha_new.T silently returns numbers
    even when the two matrices index different loci, so this is an assertion,
    not a warning.
    """
    say("[input] breeding population")
    ids_b, chrom_b, pos_b, snp_b, Hb = read_phased_vcf(cfg.BREED_VCF)
    Gb = hap_to_geno(Hb)
    if len(snp_b) != len(ref["snp"]):
        raise SystemExit(
            f"marker count differs: reference {len(ref['snp'])}, "
            f"breeding {len(snp_b)}. The two VCFs must carry the same markers "
            f"in the same order.")
    mism = np.where(np.asarray(snp_b) != np.asarray(ref["snp"]))[0]
    if len(mism):
        raise SystemExit(
            f"{len(mism)} markers differ between the two VCFs "
            f"(first at index {mism[0]}: reference {ref['snp'][mism[0]]}, "
            f"breeding {snp_b[mism[0]]}). Re-extract both from the same "
            f"marker list before running allocation.")
    if not np.array_equal(chrom_b, ref["chrom"]) or not np.array_equal(pos_b, ref["pos"]):
        raise SystemExit("marker names match but chromosome/position do not")
    say(f"    marker sets identical ({len(snp_b)} SNPs)")
    sex = read_sex(cfg.SEX_FILE, ids_b)
    return ids_b, Gb, sex


def population_diagnostics(Gr, Gb, out_dir, trait):
    """
    Two numbers a reviewer will ask for. The model is fitted in the reference
    population and applied to the breeding population; both checks ask whether
    that extrapolation is defensible.
    """
    pr, pb = Gr.mean(0) / 2.0, Gb.mean(0) / 2.0
    r_freq = float(np.corrcoef(pr, pb)[0, 1])
    KA_x = grm_a_cross(Gr, Gb, pr)
    KA_b = grm_a_fixed_freq(Gb, pr)
    d = dict(trait=trait,
             n_ref=int(Gr.shape[0]), n_breed=int(Gb.shape[0]),
             allele_freq_cor=r_freq,
             mean_freq_shift=float(np.mean(np.abs(pr - pb))),
             mean_relatedness_ref_breed=float(KA_x.mean()),
             max_relatedness_ref_breed=float(KA_x.max()),
             mean_relatedness_within_breed=float(
                 KA_b[np.triu_indices(len(KA_b), 1)].mean()))
    say(f"    allele-frequency correlation ref vs breeding: {r_freq:.4f}")
    say(f"    mean relatedness ref-breeding {d['mean_relatedness_ref_breed']:+.4f}"
        f"   max {d['max_relatedness_ref_breed']:+.4f}")
    if r_freq < 0.95:
        say("    [warning] the breeding population has drifted from the")
        say("              reference. Marker effects estimated in one and")
        say("              applied to the other are extrapolation; say so.")
    pd.DataFrame([d]).to_csv(
        os.path.join(out_dir, f"{trait}_population_diagnostics.csv"), index=False)
    return KA_b, d


class EpiFast:
    """
    Epistatic value of expected offspring, via pre-stored Gram columns.

    E[Ha_off] = (Ha_P + Ha_Q)/2, so the reference-vs-offspring Gram column is
    the average of the two parental columns. Precomputing them turns the cost
    per cross from a full matrix product into O(n_ref x n_chrom). Verified
    against GeneticModel.epi_value in --selftest.
    """

    def __init__(self, gm, Gb):
        Ha_b = Gb - gm.mu[None, :]
        self.full = gm.Ha_ref @ Ha_b.T                       # n_ref x n_breed
        self.per_chrom = {}
        for ch in np.unique(gm.chrom):
            idx = gm.chrom == ch
            self.per_chrom[ch] = gm.Ha_ref[:, idx] @ Ha_b[:, idx].T
        self.scale = gm.scale_AA if gm.use_epi else 0.0
        self.alpha = gm.alpha

    def pairs(self, S, D):
        t = (self.full[:, S] + self.full[:, D]) / 2.0
        acc = t ** 2
        for ch, Gc in self.per_chrom.items():
            tc = (Gc[:, S] + Gc[:, D]) / 2.0
            acc -= tc ** 2
        return self.scale * (self.alpha @ acc)


def fit_reference_model(cfg, ref, K, meta, Ha, Hd, trait):
    """Fit the A/D/MULT_t model on the phenotyped reference animals."""
    import pig_multigen_figures as MG
    y = ref["phe"][trait].values.astype(float)
    keep = np.where(np.isfinite(y))[0]
    if len(keep) < 100:
        raise SystemExit(f"{trait}: only {len(keep)} phenotyped reference animals")
    Ksub = {k: K[k][np.ix_(keep, keep)] for k in ("A", "D", "MULT_t") if k in K}
    msub = dict(meta)
    msub["KA_raw"] = meta["KA_raw"][np.ix_(keep, keep)]
    gm = MG.GeneticModel(ref["G"][keep], Ha[keep], Hd[keep], ref["chrom"],
                         Ksub, msub, y[keep], np.ones((len(keep), 1)))
    tot = sum(gm.var.values())
    say("    variance components: "
        + "  ".join(f"{k}={v/tot:.4f}" for k, v in gm.var.items() if k != "_resid")
        + f"  resid={gm.var['_resid']/tot:.4f}")
    if gm.var.get("MULT_t", 0.0) / tot < 1e-4:
        say("    [note] the epistatic component was estimated at the boundary")
        say("           (~0), so gv_epi is 0 by construction and FULL will")
        say("           coincide with ADD+DOM. With n~599 the self-calibration")
        say("           gave only ~20% power at an injected 0.15, so this is a")
        say("           power limit, NOT evidence that epistasis is absent.")
    if gm.var.get("D", 0.0) / tot < 1e-4:
        say("    [note] the dominance component was estimated at ~0; ADD+DOM")
        say("           will coincide with ADD.")
    return gm, len(keep)


def stage_mating(cfg, ref, K, meta, Ha, Hd, trait, breed):
    say(f"\n[D] mate allocation for {trait}")
    ids_b, Gb, sex = breed
    gm, n_used = fit_reference_model(cfg, ref, K, meta, Ha, Hd, trait)
    KA_b, _ = population_diagnostics(ref["G"], Gb, cfg.OUT_DIR, trait)
    epi_fast = EpiFast(gm, Gb) if gm.use_epi else None
    return mate_allocation(cfg, gm, Gb, ids_b, sex, KA_b, cfg.OUT_DIR, trait,
                           epi_fast=epi_fast)


# ============================================================================
# Self-test
# ============================================================================
def _synthetic_vcf(path, n, m, n_chrom, rng):
    chrom = np.repeat(np.arange(1, n_chrom + 1), m // n_chrom)[:m]
    pos = np.concatenate([np.sort(rng.integers(1, 80_000_000, (chrom == c).sum()))
                          for c in range(1, n_chrom + 1)])
    with open(path, "w") as f:
        f.write("##fileformat=VCFv4.2\n")
        f.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t"
                + "\t".join(f"ind{i}" for i in range(n)) + "\n")
        for k in range(m):
            fr = rng.uniform(0.15, 0.85)
            h = (rng.random((n, 2)) < fr).astype(int)
            gts = "\t".join(f"{a}|{b}" for a, b in h)
            f.write(f"{chrom[k]}\t{pos[k]}\tsnp{k}\tA\tG\t.\t.\t.\tGT\t{gts}\n")
    return chrom, pos


def selftest():
    say("=" * 78)
    say("SELF-TEST")
    say("=" * 78)
    ok = True
    rng = np.random.default_rng(0)
    tmp = "/tmp/mussel_st"
    os.makedirs(tmp, exist_ok=True)
    vcf = os.path.join(tmp, "ref.vcf")
    _synthetic_vcf(vcf, 160, 600, 6, rng)

    say("\n[1] phased VCF reader")
    ids, chrom, pos, snp, H = read_phased_vcf(vcf)
    G = hap_to_geno(H)
    good = (H.shape == (160, 600, 2) and G.shape == (160, 600)
            and set(np.unique(G)).issubset({0.0, 1.0, 2.0}))
    say(f"    shapes {H.shape} -> {G.shape}, genotype values "
        f"{sorted(set(np.unique(G)))}  {'OK' if good else 'FAIL'}")
    ok &= good

    say("\n[2] recombination-aware gametes: linkage decay")
    het = np.zeros((600, 2), dtype=np.int8)
    het[:, 0] = 1
    gm_ = gametes(het, chrom, pos, 3000, rng, 1.0)
    c0 = chrom == 1
    sub = gm_[:, c0]
    lags = [(1, "adjacent"), (20, "20 apart"), (80, "80 apart")]
    prev = 1.0
    for lag, nm in lags:
        if sub.shape[1] <= lag:
            continue
        same = float(np.mean(sub[:, :-lag] == sub[:, lag:]))
        say(f"    {nm:10s} identity {same:.4f}")
        if same > prev + 1e-3:
            ok = False
            say("      [FAIL] identity should not increase with distance")
        prev = same
    if prev > 0.95:
        say("    [note] map too dense for decay to show at these lags")

    say("\n[3] free recombination would destroy the decay (contrast)")
    free = (rng.random((3000, 600)) < 0.5).astype(float)
    say(f"    free-recombination identity at lag 1 = "
        f"{float(np.mean(free[:, :-1] == free[:, 1:])):.4f} (expected ~0.5)")

    say("\n[4] genetic model and expected-vs-simulated offspring value")
    Ha, Hd = P.noia(G)
    qcfg = Q.CFG()
    qcfg.CACHE_DIR = os.path.join(tmp, "ck")
    qcfg.DO_CIS = False
    K, meta = Q.build_kernels(qcfg, G, Ha, Hd, chrom, keep_xc=False)
    d, U = np.linalg.eigh(K["A"])
    g = U @ (np.sqrt(np.maximum(d, 0)) * rng.normal(0, 1, 160))
    z = lambda v: (v - v.mean()) / (v.std() + 1e-12)
    y = 0.7 * z(g) + 0.7 * rng.normal(0, 1, 160)
    import pig_multigen_figures as MG
    gmod = MG.GeneticModel(G, Ha, Hd, chrom, K, meta, y, np.ones((160, 1)))
    i, j = 3, 17
    e_ha, e_hd = gmod.expected_codes(G[i:i + 1], G[j:j + 1])
    exp_add = float((e_ha @ gmod.a_hat)[0])
    ga = gametes(H[i], chrom, pos, 2000, rng, 1.0)
    gb = gametes(H[j], chrom, pos, 2000, rng, 1.0)
    kid = ga + gb
    v = gmod.value(kid)
    sim = float(v["add"].mean())
    se = float(v["add"].std(ddof=1) / np.sqrt(len(v["add"]))) + 1e-12
    zsc = abs(exp_add - sim) / se
    say(f"    additive: expected {exp_add:+.5f}  simulated {sim:+.5f}  "
        f"s.e. {se:.5f}  z = {zsc:.2f}")
    if zsc > 4:
        ok = False
        say("      [FAIL] expected and simulated offspring values disagree")

    say("\n[5] sexed allocation respects both use caps and the inbreeding ceiling")
    cfg = CFG()
    cfg.N_CROSSES, cfg.MAX_USE_SIRE, cfg.MAX_USE_DAM = 12, 3, 2
    n_b = 40
    sex = np.array(["M"] * 20 + ["F"] * 20)
    KA_b = meta["KA_raw"][np.ix_(np.arange(n_b), np.arange(n_b))]
    df, summ = mate_allocation(cfg, gmod, G[:n_b], np.array([f"b{i}" for i in range(n_b)]),
                               sex, KA_b, tmp, "test")
    plan = pd.read_csv(os.path.join(tmp, "test_mating_plan.csv"))
    for strat in plan.strategy.unique():
        p = plan[plan.strategy == strat]
        vs = p.sire.value_counts().max()
        vd = p.dam.value_counts().max()
        bad = vs > cfg.MAX_USE_SIRE or vd > cfg.MAX_USE_DAM or p.progeny_F.max() > cfg.F_MAX
        say(f"    {strat:9s} max sire use {vs}  max dam use {vd}  "
            f"max F {p.progeny_F.max():+.4f}  {'FAIL' if bad else 'OK'}")
        ok &= not bad

    say("\n[6] fixed-frequency GRM reproduces P.grm_a on its own frequencies")
    p_self = G.mean(0) / 2.0
    d6 = float(np.abs(grm_a_fixed_freq(G, p_self) - P.grm_a(G)).max())
    say(f"    max abs difference {d6:.3e}  {'OK' if d6 < 1e-9 else 'FAIL'}")
    ok &= d6 < 1e-9

    say("\n[7] fast epistatic evaluator agrees with GeneticModel.epi_value")
    if gmod.use_epi:
        Gb_ = G[:40]
        ef = EpiFast(gmod, Gb_)
        S_ = np.arange(0, 20)
        D_ = np.arange(20, 40)
        e_ha, _ = gmod.expected_codes(Gb_[S_], Gb_[D_])
        slow = gmod.epi_value(e_ha)
        fast = ef.pairs(S_, D_)
        scale = max(float(np.abs(slow).max()), 1e-12)
        d7 = float(np.abs(slow - fast).max()) / scale
        say(f"    max relative difference {d7:.3e}  {'OK' if d7 < 1e-8 else 'FAIL'}")
        ok &= d7 < 1e-8
    else:
        say("    [skip] REML put the epistatic component at the boundary on")
        say("           this synthetic set, so there is nothing to compare")

    say("\n[8] sex reader tolerates different codings")
    for coding in [("M", "F"), ("1", "2"), ("male", "female")]:
        f_ = os.path.join(tmp, f"sex_{coding[0]}.csv")
        pd.DataFrame({"id": [f"ind{i}" for i in range(40)],
                      "sex": [coding[0]] * 20 + [coding[1]] * 20}).to_csv(f_, index=False)
        got = read_sex(f_, np.array([f"ind{i}" for i in range(40)], dtype=object))
        good = (got == "M").sum() == 20 and (got == "F").sum() == 20
        say(f"    coding {coding} -> {'OK' if good else 'FAIL'}")
        ok &= good

    say("\n[9] the allocator actually responds to the non-additive term")
    say("    (item 5 passes even if gv_epi were ignored, because all three")
    say("     strategies happened to coincide there; this forces divergence)")
    rng9 = np.random.default_rng(7)
    n9 = 24
    s9 = np.repeat(np.arange(n9 // 2), n9 // 2)
    d9 = np.tile(np.arange(n9 // 2, n9), n9 // 2)
    add9 = rng9.normal(0, 1, len(s9))
    epi9 = rng9.normal(0, 1, len(s9))            # uncorrelated with add by design
    df9 = pd.DataFrame(dict(sire=s9, dam=d9, gv_add=add9, gv_epi=epi9,
                            gv_full=add9 + epi9,
                            progeny_F=np.zeros(len(s9))))
    cfg9 = CFG()
    cfg9.N_CROSSES, cfg9.MAX_USE_SIRE, cfg9.MAX_USE_DAM, cfg9.F_MAX = 8, 3, 2, 1.0
    df9["score"] = df9.gv_add
    pa = set(map(tuple, df9.iloc[allocate_sexed(df9, cfg9)][["sire", "dam"]].values))
    df9["score"] = df9.gv_full
    pf = set(map(tuple, df9.iloc[allocate_sexed(df9, cfg9)][["sire", "dam"]].values))
    ov = len(pa & pf) / max(len(pa), 1)
    say(f"    overlap_with_ADD under an independent epistatic term = {ov:.2f}"
        f"  {'OK' if ov < 1.0 else 'FAIL'}")
    if ov >= 1.0:
        say("      [FAIL] the epistatic term did not change the chosen crosses,")
        say("             so the allocator is not using it")
    ok &= ov < 1.0

    say("\nSELF-TEST: " + ("PASS" if ok else "FAIL"))
    return ok


# ============================================================================
# Driver
# ============================================================================
def load_population(cfg):
    say("[input] reference population")
    ids_r, chrom, pos, snp, Hr = read_phased_vcf(cfg.REF_VCF)
    Gr = hap_to_geno(Hr)
    phe = pd.read_csv(cfg.PHENO)
    ia, ib = align_ids(ids_r, phe.iloc[:, 0].values)
    if len(ia) == 0:
        raise SystemExit("reference genotype ids do not match the phenotype file")
    say(f"    matched {len(ia)}/{len(ids_r)} reference animals to phenotypes")
    return dict(ids=ids_r[ia], chrom=chrom, pos=pos, snp=snp,
                H=Hr[ia], G=Gr[ia], phe=phe.iloc[ib].reset_index(drop=True))


def screen_trait(cfg, ref, trait, K, meta, Ha, Hd, labels):
    """Stages [A] and [B] for one trait. Returns (diffuse rows, decomposition)."""
    y = ref["phe"][trait].values.astype(float)
    keep = np.isfinite(y)
    if keep.sum() < 100:
        say(f"    [skip] {trait}: only {keep.sum()} phenotyped animals")
        return None, None
    n = len(y)
    Xfix = np.ones((n, 1))

    diffuse = {"A": ["A"], "A+D": ["A", "D"]}
    for k in (x for x in K if x not in ("A", "D")):
        diffuse[f"A+D+{k}"] = ["A", "D", k]
    if getattr(cfg, "ONLY_MODELS", None):
        diffuse = {k: v for k, v in diffuse.items() if k in cfg.ONLY_MODELS}
        if "A" not in diffuse or "A+D" not in diffuse:
            raise SystemExit("--only_models must include both A and A+D")

    rows, dom_rows, acc, bias = [], [], {}, {}
    for si in range(cfg.N_SEEDS):
        seed = cfg.SEED + si * 101
        tr, te = Q.make_split(n, labels, "random", cfg.TEST_FRAC, seed)
        tr = tr[keep[tr]]
        te = te[keep[te]]
        todo = {k: v for k, v in diffuse.items()
                if k in cfg.CORE_MODELS or si < cfg.SECONDARY_SEEDS}
        preds = {}
        for nm, keys in todo.items():
            gb, ya, _ = Q.fit_gebv(y, Xfix, K, keys, tr, te)
            if gb is None:
                continue
            preds[nm] = (gb, ya)
            acc.setdefault(nm, []).append(Q.pcc(ya, gb))
            bias.setdefault(nm, []).append(Q.pred_bias(ya, gb))
        if "A+D" not in preds:
            continue
        g0, ya = preds["A+D"]
        # --- dominance: A+D vs A. Never tested before: the screen's baseline
        # IS A+D, so a dominance gain was invisible in the delta table even
        # when it was the largest non-additive effect in the dataset.
        if "A" in preds:
            gA, _ = preds["A"]
            d, lo, hi = Q.boot_delta(ya, gA, g0, cfg.N_BOOT, seed)
            dom_rows.append(dict(trait=trait, seed=seed, model="A+D vs A",
                                 dPCC=d, ci_lo=lo, ci_hi=hi))
        for nm, (g1, _) in preds.items():
            if nm in ("A", "A+D"):
                continue
            d, lo, hi = Q.boot_delta(ya, g0, g1, cfg.N_BOOT, seed)
            rows.append(dict(trait=trait, seed=seed, model=nm, dPCC=d,
                             ci_lo=lo, ci_hi=hi))
    ns = cfg.SECONDARY_SEEDS
    for nm, v in acc.items():
        say(f"      {nm:16s} accuracy {np.mean(v):.4f} +- {np.std(v):.4f}"
            f"  (n_seeds={len(v):2d})   common-{ns} {np.mean(v[:ns]):.4f}"
            f"   bias {np.nanmean(bias.get(nm, [np.nan])):.3f}")
    say(f"      ^ core models use {cfg.N_SEEDS} seeds, secondary models {ns};")
    say(f"        the 'accuracy' column is therefore NOT comparable across")
    say(f"        rows. Use common-{ns} for that, or the paired dPCC below.")

    # ---- [B] breeding-value decomposition ----
    keys = [k for k in ("A", "D", "MULT_t") if k in K]
    kk = np.where(keep)[0]
    Ktt = [K[k][np.ix_(kk, kk)] for k in keys]
    mm = P.REMLgrad().fit(y[kk], Xfix[kk], Ktt)
    v = np.asarray(mm.variances_, float)
    V = sum(s_ * A for s_, A in zip(v[:-1], Ktt)) + (v[-1] + 1e-8) * np.eye(len(kk))
    c = cho_factor(V, lower=True)
    b = np.linalg.solve(Xfix[kk].T @ cho_solve(c, Xfix[kk]),
                        Xfix[kk].T @ cho_solve(c, y[kk]))
    al = cho_solve(c, y[kk] - Xfix[kk] @ b)
    u = {k: float(s_) * (A @ al) for k, s_, A in zip(keys, v[:-1], Ktt)}
    dec = dict(trait=trait, n=int(keep.sum()))
    for i, k in enumerate(keys):
        dec[f"var_{k}"] = float(v[i] / v.sum())
    dec["var_resid"] = float(v[-1] / v.sum())
    if "A" in u and "MULT_t" in u:
        dec["r_A_MULT_t"] = Q.pcc(u["A"], u["MULT_t"])
    acc_rows = [dict(trait=trait, model=nm, accuracy=float(np.mean(v_)),
                     acc_sd=float(np.std(v_)), n_seeds=int(len(v_)),
                     accuracy_common=float(np.mean(v_[:ns])),
                     bias=float(np.nanmean(bias.get(nm, [np.nan]))))
                for nm, v_ in acc.items()]
    return ((pd.DataFrame(rows) if rows else None),
            (dec, acc_rows),
            (pd.DataFrame(dom_rows) if dom_rows else None))


IMPLEMENTED_STAGES = {"diffuse", "decompose", "mating"}
PLANNED_STAGES = {
    "focal":    "[C] genome-wide pair scoring -- deferred by design; it does "
                "not feed [D], and it is where the winner's-curse problem "
                "lives, so it runs after the aggregate result is known",
    "multigen": "[E] five-generation simulation -- gametes() exists and is "
                "self-tested, but no generation loop calls it",
    "validate": "[V] F1 prospective test -- awaiting the real F1 file headers",
}


def run(cfg, stages, traits):
    t0 = time.time()

    # Checked FIRST, before the kernel build, so a mistyped stage costs seconds
    # rather than the whole screening run.
    unknown = stages - IMPLEMENTED_STAGES - set(PLANNED_STAGES)
    if unknown:
        raise SystemExit(f"unknown stage(s): {sorted(unknown)}. "
                         f"Known: {sorted(IMPLEMENTED_STAGES | set(PLANNED_STAGES))}")
    not_ready = sorted(stages & set(PLANNED_STAGES))
    if not_ready:
        say("=" * 74)
        say("Requested stages that this build does not implement:")
        for s in not_ready:
            say(f"    {s:9s} {PLANNED_STAGES[s]}")
        say("=" * 74)
        raise SystemExit(
            "Refusing to run. Accepting these stage names and ignoring them "
            "would let a run appear to have done allocation when it had only "
            "screened. Use --stages diffuse,decompose for the screening "
            "batches.")
    if not stages & IMPLEMENTED_STAGES:
        raise SystemExit("no runnable stage requested")
    say(f"[stages] {sorted(stages)}")

    os.makedirs(cfg.OUT_DIR, exist_ok=True)
    ref = load_population(cfg)
    G, chrom = ref["G"], ref["chrom"]
    n, m = G.shape
    say(f"[data] {n} reference animals x {m} SNPs")
    Ha, Hd = P.noia(G)

    say("\n[kernels] exact aggregate kernels (genotype-only, cached)")
    qcfg = Q.CFG()
    qcfg.BFILE = "mussel"
    qcfg.CACHE_DIR = cfg.CACHE_DIR
    K, meta = Q.build_kernels(qcfg, G, Ha, Hd, chrom, keep_xc=False)
    iu = np.triu_indices(n, 1)
    names = list(K)
    CM = pd.DataFrame({a: {b: np.corrcoef(K[a][iu], K[b][iu])[0, 1] for b in names}
                       for a in names})
    CM.to_csv(os.path.join(cfg.OUT_DIR, "kernel_collinearity.csv"))
    hi = [(a, b, float(CM.loc[a, b])) for a in names for b in names
          if a < b and abs(CM.loc[a, b]) > 0.5]
    if hi:
        say("    kernels with |r| > 0.5 (variance components not separable;")
        say("    report their delta in prediction, not their variance share):")
        for a, b, r in sorted(hi, key=lambda z: -abs(z[2]))[:6]:
            say(f"      {a:8s} ~ {b:8s}  r = {r:+.3f}")
    labels, how = P.structure_clusters(K["A"], None, 10, cfg.SEED)
    say(f"[structure] {how}")

    do_screen = bool(stages & {"diffuse", "decompose"})
    do_mating = "mating" in stages
    breed = load_breeding(cfg, ref) if do_mating else None

    all_d, all_dec, all_acc, all_mate, all_dom = [], [], [], [], []
    for t in traits:
        if t not in ref["phe"].columns:
            say(f"\n[skip] {t}: not a column in {cfg.PHENO}")
            continue
        say(f"\n{'='*74}\nTRAIT {t}"
            + ("   [derived trait -- not an independent test]"
               if t in cfg.DERIVED_TRAITS else "")
            + f"\n{'='*74}")
        t1 = time.time()
        if do_screen:
            D, dec, DOM = screen_trait(cfg, ref, t, K, meta, Ha, Hd, labels)
            if DOM is not None:
                all_dom.append(DOM)
            if D is not None:
                all_d.append(D)
                g = (D.groupby("model")
                     .agg(dPCC=("dPCC", "mean"),
                          pos=("dPCC", lambda v: float(np.mean(np.asarray(v) > 0))))
                     .reset_index().sort_values("dPCC", ascending=False))
                say("\n" + g.round(5).to_string(index=False))
            if dec is not None:
                all_dec.append(dec[0])
                all_acc.extend(dec[1])
        if do_mating:
            _, summ = stage_mating(cfg, ref, K, meta, Ha, Hd, t, breed)
            all_mate.append(summ)
        say(f"    {t} took {(time.time()-t1)/60:.1f} min")

    def dump(obj, name):
        if not obj:
            return None
        df = pd.concat(obj, ignore_index=True) if isinstance(obj[0], pd.DataFrame) \
            else pd.DataFrame(obj)
        df.to_csv(os.path.join(cfg.OUT_DIR, name), index=False)
        return df

    D = dump(all_d, f"diffuse_delta_pcc_{traits[0]}_{traits[-1]}.csv")
    dump(all_dec, f"decomposition_{traits[0]}_{traits[-1]}.csv")
    dump(all_acc, f"accuracy_bias_{traits[0]}_{traits[-1]}.csv")
    M = dump(all_mate, f"mating_summary_{traits[0]}_{traits[-1]}.csv")
    DM = dump(all_dom, f"dominance_delta_pcc_{traits[0]}_{traits[-1]}.csv")

    if DM is not None:
        say("\n" + "=" * 74)
        say("Dominance screen: A+D vs an A-only baseline")
        say("=" * 74)
        gd = (DM.groupby("trait")
              .agg(dPCC=("dPCC", "mean"), sd=("dPCC", "std"),
                   pos_frac=("dPCC", lambda v: float(np.mean(np.asarray(v) > 0))))
              .reset_index().sort_values("dPCC", ascending=False))
        gd["carry_forward"] = ((gd.dPCC > cfg.EPI_DPCC_MIN)
                               & (gd.pos_frac >= cfg.EPI_POSFRAC_MIN))
        say(gd.round(5).to_string(index=False))
        gd.to_csv(os.path.join(cfg.OUT_DIR,
                               f"dominance_screen_{traits[0]}_{traits[-1]}.csv"),
                  index=False)
        say("\n  The epistasis screen below uses A+D as its baseline, so a")
        say("  dominance gain never appears there. In an outcrossing bivalve")
        say("  this is the more likely place for non-additive signal to live.")

    if M is not None:
        say("\n" + "=" * 74)
        say("Allocation result: does non-additive information change the plan?")
        say("=" * 74)
        say(M.round(4).to_string(index=False))
        say("\n  overlap_with_ADD is the number that matters. 1.00 means the")
        say("  non-additive terms reordered nothing and allocation adds nothing")
        say("  over additive selection; below 1.00 means dominance or epistasis")
        say("  changed which crosses were chosen.")

    if D is not None:
        say("\n" + "=" * 74)
        say("Screen result: epistatic kernel MULT_t vs the A+D baseline")
        say("=" * 74)
        sub = D[D.model == "A+D+MULT_t"]
        g = (sub.groupby("trait")
             .agg(dPCC=("dPCC", "mean"), sd=("dPCC", "std"),
                  pos_frac=("dPCC", lambda v: float(np.mean(np.asarray(v) > 0))))
             .reset_index().sort_values("dPCC", ascending=False))
        g["carry_forward"] = ((g.dPCC > cfg.EPI_DPCC_MIN)
                              & (g.pos_frac >= cfg.EPI_POSFRAC_MIN))
        g["derived"] = g.trait.isin(cfg.DERIVED_TRAITS)
        say(g.round(5).to_string(index=False))
        g.to_csv(os.path.join(cfg.OUT_DIR,
                              f"epistasis_screen_{traits[0]}_{traits[-1]}.csv"),
                 index=False)
        sel = list(g.loc[g.carry_forward, "trait"])
        say(f"\n  carried forward to allocation (dPCC > {cfg.EPI_DPCC_MIN} and "
            f"positive in >= {cfg.EPI_POSFRAC_MIN:.0%} of splits): "
            + (", ".join(sel) if sel else "none"))
        say("  derived traits are screened but are not independent tests; do")
        say("  not count them separately when accounting for multiplicity.")
        json.dump(dict(batch=list(traits), selected=sel,
                       minutes=(time.time() - t0) / 60),
                  open(os.path.join(cfg.OUT_DIR,
                                    f"screen_summary_{traits[0]}_{traits[-1]}.json"),
                       "w"), indent=2)
    say(f"\nDone in {(time.time()-t0)/60:.1f} min -> {cfg.OUT_DIR}/")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref_vcf")
    ap.add_argument("--breed_vcf")
    ap.add_argument("--pheno")
    ap.add_argument("--sex")
    ap.add_argument("--trait")
    ap.add_argument("--out")
    ap.add_argument("--cm_per_mb", type=float)
    ap.add_argument("--seeds", type=int)
    ap.add_argument("--generations", type=int)
    ap.add_argument("--reps", type=int)
    ap.add_argument("--stages", default="diffuse,decompose")
    ap.add_argument("--traits", nargs="*", help="explicit trait list for this batch")
    ap.add_argument("--batch", type=int, help="1-based batch index")
    ap.add_argument("--batch_size", type=int, default=5)
    ap.add_argument("--only_models", nargs="*",
                    help="restrict to these models, e.g. --only_models A A+D "
                         "for a cheap dominance-only rerun")
    ap.add_argument("--f1_vcf")
    ap.add_argument("--f1_pheno")
    ap.add_argument("--f1_ped")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    cfg = CFG()
    for v, k in [(a.ref_vcf, "REF_VCF"), (a.breed_vcf, "BREED_VCF"),
                 (a.pheno, "PHENO"), (a.sex, "SEX_FILE"), (a.trait, "TRAIT"),
                 (a.out, "OUT_DIR"), (a.cm_per_mb, "CM_PER_MB"),
                 (a.seeds, "N_SEEDS"), (a.generations, "GENERATIONS"),
                 (a.reps, "N_REPS"), (a.f1_vcf, "F1_VCF"),
                 (a.f1_pheno, "F1_PHENO"), (a.f1_ped, "F1_PED"),
                 (a.only_models, "ONLY_MODELS")]:
        if v is not None:
            setattr(cfg, k, v)
    if a.selftest:
        sys.exit(0 if selftest() else 1)

    if a.traits:
        traits = list(a.traits)
    elif a.batch:
        alls = list(cfg.ALL_TRAITS)
        s0 = (a.batch - 1) * a.batch_size
        traits = alls[s0:s0 + a.batch_size]
        if not traits:
            raise SystemExit(f"batch {a.batch} is empty; "
                             f"{len(alls)} traits, batch_size {a.batch_size}")
        say(f"[batch {a.batch}] traits {traits}")
    else:
        traits = [cfg.TRAIT]
    stages = {x.strip() for x in a.stages.split(",") if x.strip()}
    run(cfg, stages, traits)


if __name__ == "__main__":
    main()
