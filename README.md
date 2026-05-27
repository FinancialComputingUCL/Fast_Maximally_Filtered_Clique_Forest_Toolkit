# Fast Maximally Filtered Clique Forest Toolkit

Accelerated Maximally Filtered Clique Forest (MFCF) implementations for sparse precision estimation and hierarchical clustering. The core `fast_fast_mfcf` routine is ~100× faster than the original reference implementation while preserving the same outputs, enabling practical experimentation with large correlation or covariance matrices. See [this repo](https://github.com/yanh11/fast_fast_mfcf) for more details.

## Introduction
- Build MFCF backbones from dense similarity matrices in a few seconds.
- Scikit-learn-compatible graphical-model estimator (`MFCFLoGo`) that offers a faster, more accurate alternative to Graphical Lasso when the sample-to-feature ratio is small. HPO is left to the surrounding pipeline so it can be tuned to the application's leakage and scoring conventions.

## Repository layout
All modules live in `src/`:
- `fast_fast_mfcf.py` — the MFCF builder (this is the ~100× speed-up).
- `mfcf_logo.py` — the `MFCFLoGo` scikit-learn-compatible estimator.
- `mutual_information.py` — KSG mutual-information matrix utilities.
- `logo_inverse_covariance_estimation.ipynb` — worked-example notebook.

## Getting Started
- Python 3.9+ recommended.
- Clone the repository and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

- Optional: enable logging for verbose traces:

```python
import logging
logging.basicConfig(level=logging.INFO)
```

## Usage

### Building a Maximally Filtered Clique Forest (`MFCF()`)
```python
import numpy as np
from fast_fast_mfcf import MFCF

X = ...  # samples x features
C = np.corrcoef(X, rowvar=False)
Cov = np.cov(X,rowvar=False)

builder = MFCF(
    threshold=0.05,
    min_clique_size=2,
    max_clique_size=5,
    coordination_number=10,
)
cliques, separators, peo, logo = builder.run(C=C, cov_matrix=Cov)
```
- `cliques`: list of maximal cliques.
- `separators`: separator multiplicities (`collections.Counter`).
- `peo`: perfect elimination order collected during growth.
- `logo`: sparse inverse assembled from clique and separator inverses.

### Precision Estimation (`MFCFLoGo()`)
```python
import numpy as np
from mfcf_logo import MFCFLoGo  # the "mfcflogo" estimator

X = ...  # samples x features
est = MFCFLoGo(
    threshold=0.05,
    min_clique_size=2,
    max_clique_size=6,
    coordination_number=8,
)
est.fit(X)
precision = est.get_precision()
covariance = est.get_covariance()
```
- Fully scikit-learn compatible: works with `Pipeline`, `GridSearchCV`, and scoring utilities.
- In many low-sample/high-dimensional regimes it runs faster and attains higher accuracy than Graphical Lasso while keeping interpretation straightforward through clique structure.

### Using Mutual Information Instead of Correlation
Pearson correlation is only the right similarity when the variables are
jointly Gaussian. Outside that regime (e.g., fat tails, regime shifts, non-linear co-movement) it can grossly under-state the dependence structure. The estimators accept `similarity="mutual_information"` to score MFCF gains with a pairwise mutual-information matrix instead. The matrix is built with the Kraskov–Stögbauer–Grassberger (KSG) k-nearest-neighbour estimator (Kraskov, Stögbauer, Grassberger, *Phys. Rev. E* 69, 066138, 2004).

```python
from mfcf_logo import MFCFLoGo

est = MFCFLoGo(
    similarity="mutual_information",
    mi_n_neighbors=3,         # KSG neighbours, robust around 3-6
    mi_normalize="linfoot",   # sqrt(1 - exp(-2 I)) in [0, 1]
    mi_n_jobs=-1,             # forwarded to the KSG numba/sklearn backend
    mi_random_state=0,
    max_clique_size=4,
)
est.fit(X)
```

The MI matrix uses Linfoot's informational correlation coefficient
`rho_I = sqrt(1 - exp(-2 I))` (Linfoot, *Inf. Control* 1(1), 1957), so
entries live in `[0, 1]` and collapse to `|Pearson rho|` exactly under joint
Gaussianity — the MI path is a strict generalisation of the correlation
path.

- The LoGo aggregation step still uses the empirical (or user-supplied)
  covariance matrix, because clique-wise inversion requires an actual
  covariance, not an MI matrix.
- The KSG matrix is the expensive call. If you are doing your own HPO,
  build the MI matrix **once** per training fold and pass it through
  `fit` via the `mi_matrix=` keyword on every trial that fits to the
  same fold — that skips the internal KSG call and amortises it across
  the search.
- Under joint Gaussianity the Linfoot-normalised KSG matrix converges
  to `|Pearson rho|` (Linfoot 1957) so the MI path is a strict
  generalisation of the correlation path, but at finite `n` the KSG
  estimator has much higher variance than the Pearson sample
  correlation for small `|rho|` — use it only when you have a real
  reason to suspect non-Gaussian or non-monotone dependence.

