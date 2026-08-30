#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 trait_names.py -- internal trait codes -> the abbreviations used in the paper
================================================================================

 Every figure- and table-producing script imports this module, so a code such
 as `phe4` or `lmp` appears in output exactly once per project, in the form the
 manuscript uses. Keep this file beside the other scripts; a script that cannot
 import it falls back to the raw code and prints a notice rather than failing.

 Mussel (Hyriopsis cumingii), full names as published in
 Lv et al. (2026) Aquaculture 614:743604 and Aquaculture 612:743155:
   phe1  SL     shell length
   phe2  TSH    total shell height
   phe3  AARRL  anal-angle radial rib length
   phe4  SH     shell height
   phe5  ShW    shell width
   phe6  BW     body weight
   phe7  sW     shell weight            (shell, not soft tissue)
   phe8  SWR    shell-to-body weight ratio        [derived: sW / BW]
   phe14 dE     total colour difference           [derived from L*, a*, b*]
   phe15 %PP    purple pixel percentage           [image-derived]

 phe9-phe13 (L*, a*, b*, C*, h) are not analysed in the ADEPT paper and are
 excluded by default in the figure scripts.

 Pig (Duroc), traits as defined in Yang et al. (2021) Gigascience 10:giab048:
   bf    BF     backfat thickness
   lmd   LMD    loin muscle depth
   lmp   ELMP   estimated lean meat percentage   <- note: NOT "LMP"
   rtn   RTN    right teat number (count trait)

 Rice (Oryza sativa, RDP1) phenotype columns are already written out in full
 and pass through unchanged.
================================================================================
"""

MUSSEL = {
    "phe1": "SL",
    "phe2": "TSH",
    "phe3": "AARRL",
    "phe4": "SH",
    "phe5": "ShW",
    "phe6": "BW",
    "phe7": "sW",
    "phe8": "SWR",
    "phe14": "dE",
    "phe15": "%PP",
}

# The pig codes on disk are lower case and one of them ("lmp") is NOT the
# abbreviation used in the paper, so pig needs a mapping too. Without this,
# Table 3a said ELMP while Table 4a said lmp for the same trait.
PIG = {
    "bf": "BF",
    "lmd": "LMD",
    "lmp": "ELMP",
    "rtn": "RTN",
    # traits screened earlier but not carried into the paper; mapped so that
    # any legacy output file still resolves to a readable label
    "ltn": "LTN",
    "ttn": "TTN",
    "tpd": "TPD",
}

RICE = {}   # phenotype columns are already full names

FULL_NAME = {
    "SL": "shell length", "TSH": "total shell height",
    "AARRL": "anal-angle radial rib length", "SH": "shell height",
    "ShW": "shell width", "BW": "body weight", "sW": "shell weight",
    "SWR": "shell-to-body weight ratio", "dE": "total colour difference",
    "%PP": "purple pixel percentage",
    "BF": "backfat thickness", "LMD": "loin muscle depth",
    "ELMP": "estimated lean meat percentage", "RTN": "right teat number",
}

# Derived quantities. A non-linear function of additive components can produce
# apparent statistical epistasis, so these are flagged wherever they appear
# (Table 1b footnote in the manuscript).
DERIVED = {"SWR", "dE", "%PP"}

# Mussel traits present in the data but outside the scope of the paper.
EXCLUDED_BY_DEFAULT = ["phe9", "phe10", "phe11", "phe12", "phe13"]


def _table(species):
    s = "" if species is None else str(species).lower()
    if s.startswith("mus") or s.startswith("hcu") or s.startswith("\u868c"):
        return MUSSEL
    if s.startswith("pig") or s.startswith("dur"):
        return PIG
    if s.startswith("ric") or s.startswith("ory"):
        return RICE
    return None


def pretty(trait, species=None):
    """Internal code -> paper abbreviation. Unknown codes pass through."""
    t = str(trait).strip()
    tab = _table(species)
    if tab is not None:
        return tab.get(t, tab.get(t.lower(), t))
    # species unknown: try every table, mussel first (its codes are unique)
    for tab in (MUSSEL, PIG):
        if t in tab:
            return tab[t]
        if t.lower() in tab:
            return tab[t.lower()]
    return t


def full_name(trait, species=None):
    return FULL_NAME.get(pretty(trait, species), "")


def label(trait, species=None, with_species=True):
    """Axis label, e.g. 'mus:SL' / 'pig:BF'."""
    p = pretty(trait, species)
    if not with_species or species is None:
        return p
    s = str(species).lower()
    tag = ("mus" if s.startswith("mus") or s.startswith("hcu")
           else "pig" if s.startswith("pig") or s.startswith("dur")
           else "rice" if s.startswith("ric") or s.startswith("ory")
           else str(species)[:4])
    return f"{tag}:{p}"


def is_derived(trait, species=None):
    return pretty(trait, species) in DERIVED


# Plotting order: mussel grouped biologically (size -> weight -> ratio ->
# colour), then pig, then anything else alphabetically.
MUSSEL_ORDER = ["SL", "TSH", "AARRL", "SH", "ShW", "BW", "sW", "SWR",
                "dE", "%PP"]
PIG_ORDER = ["BF", "LMD", "ELMP", "RTN"]


def sort_key(trait, species=None):
    p = pretty(trait, species)
    if p in PIG_ORDER:
        return (0, PIG_ORDER.index(p))
    if p in MUSSEL_ORDER:
        return (1, MUSSEL_ORDER.index(p))
    return (2, p)


if __name__ == "__main__":
    for sp, codes in (("pig", ["bf", "lmd", "lmp", "rtn"]),
                      ("mussel", list(MUSSEL))):
        for c in codes:
            p = pretty(c, sp)
            print(f"{sp:7s} {c:6s} -> {p:6s} "
                  f"{'[derived]' if is_derived(c, sp) else ''} "
                  f"{full_name(c, sp)}")
