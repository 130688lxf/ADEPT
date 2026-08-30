# ADEPT

**A**dditive, **D**ominance and **E**pistatic **P**artitioning for **T**rait-wise
allocation.

ADEPT decides, trait by trait, which genetic effects a population can actually
support, and then builds a mating plan using only those. Three things happen in
order, and the order matters:

1. **Capability detection.** Some modules cannot run on some material. An
   inbred panel has no heterozygotes, so dominance is not estimable; unphased
   genotypes rule out gamete simulation; no recorded sex means all pairwise
   crosses. A blocked module is reported as blocked, never silently degraded to
   a number that looks like a result.
2. **Calibration before screening.** An injection grid measures, on the real
   kernels of the real population, the false-positive rate and the smallest
   effect this material could have detected. Without that, "no epistasis" is
   uninterpretable: it may mean absent, or it may mean invisible here.
3. **Trait-wise routing.** Each trait is screened for dominance and epistasis
   against a fixed criterion, then routed to ADD, ADD+DOM or FULL. A trait with
   no usable non-additive effect falls back to a purely additive plan, and that
   fallback is the point rather than a failure.

Applied to four materials in the article: Arabidopsis (positive control), rice,
pig and freshwater mussel.

## Install

```bash
git clone <repository-url>
cd ADEPT
pip install -r requirements.txt
```

Python 3.9 or later. The core needs numpy, scipy and pandas; `joblib` enables
parallel scoring; `plink` and `admixture` are optional and the code falls back
to PCA-based clustering without them. `segmentation/` additionally needs
PyTorch, `medpy` and a CUDA device, and is not exercised by the demo.

## Run the demo

No real data required. This builds a small synthetic population and takes it
through capability detection and the kernel audit:

```bash
cd examples
bash run_example.sh
```

## Run the agent end to end

```bash
python adept/mateagent.py \
    --geno  <plink_prefix | file.vcf | file.npz> \
    --pheno <wide.csv | gcta_prefix> \
    --traits bf lmd rtn \
    --out   run1
```

Everything lands in `run1/`: `agent_report.txt` and `agent_report.json` for the
narrative, plus `capability.csv`, `kernel_collinearity.csv`, `calibration.csv`,
`screen.csv`, `decide.csv` and a `<trait>_mating_plan.csv` per routed trait.

Calibration is the expensive stage and is a property of the population rather
than of the trait, so it can be reused within a material:

```bash
python adept/mateagent.py ... --calib_csv run1/calibration.csv
```

It must never be reused across materials.

Every long stage checkpoints at the smallest recoverable unit and skips
completed work on restart, so an interrupted run resumes rather than starting
again.

## Layout

All Python modules sit flat in `adept/`. They import one another directly
(`import pig_epik_pipeline as P`), so moving them into subdirectories would
break those imports.

| | |
|---|---|
| `adept/mateagent.py` | the agent: the six-stage entry point |
| `adept/pig_epik_pipeline.py` | numerical core: PLINK I/O, NOIA coding, kernels, REML |
| `adept/power_curve.py` | injection calibration, shared by every material |
| `adept/capability_audit.py` | what the material supports, and kernel degeneracy |
| `adept/pig_epistasis_pipeline.py` | pig: screening, decomposition, pair scoring, allocation |
| `adept/mussel_pipeline.py`, `mussel_step1.py`, `mussel_step2.py` | mussel |
| `adept/rice_finalise.py` | rice: structure, kernel degeneracy, calibration |
| `adept/arabidopsis_full_validation.py`, `arabidopsis_final.py` | positive control |
| `adept/allocation_*.py`, `trait_epistasis_to_allocation.py` | allocation arms |
| `adept/validate_f1_*.py` | F1 family consistency checks |
| `adept/make_*_figures.py`, `*_make_tables_figures.py` | figures and tables |
| `segmentation/` | the ten image-segmentation models and their re-evaluation |
| `docs/FILE_MAP.md` | which script produces which figure and table |
| `docs/STATUS.md` | conventions, reproducibility, known gaps |
| `examples/` | synthetic data and a runnable demo |

Most scripts have a `--selftest` mode that runs on synthetic data and prints
PASS/FAIL checks. Start there when adapting one to new data.

Before running anything in `segmentation/`, set the image and mask directories,
either through the `SEG_IMAGE_DIR` and `SEG_MASK_DIR` environment variables or
by editing `CONFIG` at the top of each script.

## What the code enforces

These are checks rather than preferences, and each exists because the
alternative produces a plausible wrong number rather than an error. Accuracy is
`cor(GEBV, y - Xb)`; fixed effects are an intercept only, since the genomic
relationship matrix already models structure; epistatic kernels are built by
exact factorisation rather than by sampling pairs; a positive-fraction
threshold belongs to a split count, and rows fitted over a different number of
splits never face it; where kernels correlate strongly the variance components
are not separately identified, so only the prediction gain is reported. The
full list, with the reasoning behind each, is in `docs/STATUS.md`.

Multi-trait allocation raises `NotImplementedError` deliberately. Different
traits route to different models and the covariances between their non-additive
components are not estimated anywhere here, so there is no common scale on
which to combine them.

## Reproducibility

Rerunning with the same seeds in the same environment reproduces every reported
value exactly. Across environments, results can differ by up to about 0.009 in
predictive-correlation gain, because REML settles at a different point of a
nearly flat likelihood ridge when the kernels are strongly correlated and the
BLAS build differs. Pin the numerical stack before comparing runs.

## Citation

See `CITATION.cff`.

## Licence

MIT. See `LICENSE`.
