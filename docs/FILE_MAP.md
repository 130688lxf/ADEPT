# Which script produces which result

Every figure and table in the article, and the script that writes it. Paths are
relative to the repository root. Run order within a material is top to bottom.

## The agent, end to end

`adept/mateagent.py` is the entry point. One command takes genotypes and
phenotypes in and writes a full decision report out:

```
python adept/mateagent.py \
    --geno <prefix|vcf> --pheno <wide.csv|prefix> --traits <t1> <t2> ... \
    --out run1
```

Stages, in order, each writing to `run1/`:

| Stage | What it decides | Output |
|---|---|---|
| 0  INGEST | which modules the material can support at all | `agent_report.json`, `capability.csv` |
| 0b KERNELS | builds A, D and the epistatic kernel; flags collinearity | `kernel_collinearity.csv` |
| 1  CALIBRATE | false-positive rate and detection limit per module | `calibration.csv` |
| 2  SCREEN | per-trait gain in predictive correlation | `screen.csv` |
| 3  DECIDE | routes each trait to ADD, ADD+DOM or FULL | `decide.csv` |
| 4  ALLOCATE | mating plan under the routed model | `<trait>_mating_plan.csv` |
| 5  REPORT | the whole run in one readable file | `agent_report.txt` |

Calibration runs BEFORE screening on purpose. A negative screen result is not
interpretable until the detection limit is known, so stage 3 marks every
negative reached without stage 1 as `UNCALIBRATED`.

`--calib_csv <path>` reuses a finished calibration for the same material
instead of recomputing it. Calibration is a property of the population, so it
is reusable within a material and never across materials.

## Manuscript figures

| Item | Script |
|---|---|
| Fig 1 system overview | drawn by hand from the stage table above |
| Fig 2 segmentation ablation, efficiency, qualitative | `segmentation/seg_make_tables_figures.py` |
| Fig 3 three-effect screening | `adept/make_screening_figures.py` |
| Fig 4 detection-power curves | `adept/power_curve.py` (grid), `adept/pig_make_tables_figures.py` (plot) |
| Fig 5 pair validation and system decisions | `adept/make_allocation_figures.py` (`Fig_alloc_b`) |
| Fig 6 six allocation arms | `adept/make_allocation_figures.py` (`Fig_alloc_a`, `Fig_alloc_c`) |
| Fig 7 Arabidopsis positive control | `adept/make_arabidopsis_figure.py` |

## Manuscript tables

| Item | Script |
|---|---|
| Table 2 segmentation ablation | `segmentation/seg_make_tables_figures.py` |
| Table 3 screening summary | `adept/harmonise_screening.py`, then `adept/make_screening_figures.py` |
| Table 4 allocation | `adept/collect_allocation_results.py` |
| Table S1 accuracy and bias | `adept/pig_make_tables_figures.py` |
| Table S2 pair inventory | `adept/focal_pair_perfold.py`, filtered by `adept/pairval_filter.py` |
| Table S3 full allocation | `adept/collect_allocation_results.py` |
| Table S4 F1 consistency | `adept/validate_f1_families.py`, `adept/validate_f1_strategies.py` |
| Table S5 calibration | `adept/power_curve.py` |
| Table S6 kernel collinearity | `adept/capability_audit.py` |
| Table S7 variance components | `adept/mussel_step1.py` (mussel), `adept/pig_epistasis_pipeline.py` (pig) |
| Table S8 genetic correlations | `adept/mussel_step1.py` |
| Table S9 IFAM comparison | `adept/ifam_compare.py` |
| Table S10 Arabidopsis | `adept/arabidopsis_final.py` |
| Table S12 multi-generation simulation | `adept/mussel_step2.py`, `adept/pig_multigen_figures.py` |

## Per material

**Pig.** `pig_epistasis_pipeline.py` (screening, decomposition, pair scoring,
allocation) -> `dominance_from_gebv.py` (A vs A+D from stored GEBVs, no
refitting) -> `harmonise_screening.py` -> `pig_make_tables_figures.py`.

**Mussel.** `mussel_pipeline.py` (screening) -> `mussel_trait_pipeline.py`
(one trait end to end) -> `mussel_step1.py` / `mussel_step2.py` (multi-trait
variance components and the multi-generation simulation) ->
`validate_f1_families.py`.

**Rice.** `rice_finalise.py`. Structure, the dominance-kernel degeneracy audit
and the injection calibration, in one run.

**Arabidopsis.** `arabidopsis_full_validation.py` -> `arabidopsis_final.py` ->
`make_arabidopsis_figure.py`. `epik_region_restricted.py` runs the same
statistics restricted to the FRI and FLC intervals.
