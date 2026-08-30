# Repository status

## Known gaps

`segmentation/` requires PyTorch, `medpy` and a CUDA device; those networks are
not exercised by the demo in `examples/`. The image paths at the top of each
network file (`CONFIG["image_dir"]`, `CONFIG["mask_dir"]`) point at the machine
the models were trained on and must be set before use.

Multi-trait allocation raises `NotImplementedError` on purpose. Different
traits route to different models, and the covariances between the non-additive
components of different traits are not estimated anywhere in this pipeline, so
there is no common scale on which to combine them. Raising is preferred to
returning an approximation that looks like an answer.

## Conventions the code enforces

These are checks, not stylistic preferences.

- **Accuracy is `cor(GEBV, y - Xb)`**: the correlation between the genomic
  value and the phenotype corrected for fixed effects. Including `Xb` in the
  prediction and correlating against the raw phenotype inflates accuracy and
  compresses the paired difference several-fold.
- **Fixed effects are an intercept only.** Genomic principal components are
  never fitted as fixed effects: the genomic relationship matrix already models
  structure, and residualising on PCs removes genetic signal. A covariate that
  lowers the estimated heritability is absorbing genetic variance and is
  excluded, and that decision is made once and applied to every trait.
- **Epistatic kernels are built by exact factorisation, never by sampling
  pairs.** Sampling 200,000 of about 6.4e8 pairs covers 0.03% of the space and
  correlates with the exact kernel at 0.44.
- **A positive-fraction threshold belongs to a split count.** A fraction of
  0.80 is a nominal one-sided probability of 0.0059 over 20 splits and 0.19
  over 5. Rows fitted over a different number of splits are labelled
  `sensitivity` or `diagnostic` and are never judged against the threshold.
- **A module is blocked, fitted-and-null, or fitted-with-no-usable-limit**, and
  these three outcomes are reported separately throughout. Only the second is a
  bounded statement about a trait.
- **Where kernels correlate strongly, variance components are not separately
  identified.** Report the out-of-sample gain and treat variance fractions as
  descriptive. `capability_audit.py` prints the correlations.

## Reproducibility

Rerunning with the same seeds **in the same computing environment** reproduces
every reported value exactly (measured maximum absolute difference 0.0).
**Across environments** results can differ by up to about 0.009 in predictive
correlation gain: REML lands on a different point of a nearly flat likelihood
ridge when the kernels are strongly correlated and the underlying BLAS/LAPACK
build differs. All results in the article were produced in a single
environment. Pin the numerical stack (`requirements.txt`) before comparing runs.

Long-running stages checkpoint at the smallest recoverable unit -- one
(pair, fold) cell, one (module, injection level) cell -- and skip completed
cells on restart.

## Language

Source comments and console output are English throughout `adept/`. Two kinds
of Chinese string survive as escaped code points (`"\u516c"`, `"\u6bcd"` and
similar) inside identifier- and sex-parsing helpers, and in the column names of
the F1 workbook reader. They are input-data tokens, not documentation: removing
them would stop the readers from accepting files that real breeding programmes
produce.
