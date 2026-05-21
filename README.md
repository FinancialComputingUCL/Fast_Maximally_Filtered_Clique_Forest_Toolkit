# Fast Maximally Filtered Clique Forest Toolkit

Accelerated Maximally Filtered Clique Forest (MFCF) implementations for sparse precision estimation and hierarchical clustering. The core `fast_fast_mfcf` routine is ~100× faster than the original reference implementation while preserving the same outputs, enabling practical experimentation with large correlation or covariance matrices. See [this repo](https://github.com/yanh11/fast_fast_mfcf) for more details.

## Introduction
- Build MFCF backbones from dense similarity matrices in a few seconds.
- Drop-in graphical-model estimators (`MFCFLoGO`, `MFCFLoGOCV`, `MFCFLoGOCVAll`) that follow the scikit-learn API and offer a faster, more accurate alternative to Graphical Lasso when the sample-to-feature ratio is small.
- Extend Riskfolio-Lib’s Direct Bubble Hierarchical Tree (DBHT) pipeline so it can operate over any MFCF backbone instead of only TMFG graphs (`mfcf_dbht`).

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

### Precision Estimation (`MFCFLoGO()`)
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

### Cross-validated Precision (`mfcflogocv()`)
```python
from mfcf_logo import MFCFLoGoCV

cv_est = MFCFLoGoCV(
    max_clique_size_grid=[3, 4, 5, 6],
    cv=5,
    threshold=0.0,
    min_clique_size=1,
    coordination_number=12,
)
cv_est.fit(X)
print(cv_est.best_max_clique_size_)
best_precision = cv_est.get_precision()
```
- Automatically selects `max_clique_size` via K-fold log-likelihood scoring.
- Keeps full diagnostics in `cv_results_` and per-fold records in `fold_scores_`.

### Automated Hyperparameter Search (`mfcflogocvall()`)
```python
from mfcf_logo import MFCFLoGoCVAll

tuned = MFCFLoGoCVAll(
    tunable_params=("threshold", "max_clique_size", "coordination_number"),
    n_trials=50,
    cv=5,
    shuffle=True,
    random_state=42,
)
tuned.fit(X)
best_covariance = tuned.get_covariance()
best_params = tuned.estimator_
```
- Uses Optuna-backed cross-validation to tune any combination of threshold, clique sizes, and coordination cap.
- `estimator_` stores the final `MFCFLoGo` instance fitted with the best parameters.

### Using Mutual Information Instead of Correlation
Pearson correlation is only the right similarity when the variables are
jointly Gaussian. Outside that regime — fat tails, regime shifts, non-linear
co-movement — it can grossly under-state the dependence structure. The
estimators accept `similarity="mutual_information"` to score MFCF gains with
a pairwise mutual-information matrix instead:

```python
from mfcf_logo import MFCFLoGo

est = MFCFLoGo(
    similarity="mutual_information",
    mi_n_neighbors=3,         # KSG neighbours, robust around 3-6
    mi_normalize="linfoot",   # default; sqrt(1 - exp(-2 I)) in [0, 1]
    mi_n_jobs=-1,             # forwarded to sklearn (>= 1.5)
    mi_random_state=0,
    max_clique_size=4,
)
est.fit(X)
```

- The MI matrix is computed with the Kraskov–Stögbauer–Grassberger k-NN
  estimator (Kraskov, Stögbauer, Grassberger, *Phys. Rev. E* 69, 066138,
  2004). It is fully non-parametric, asymptotically unbiased, and is the
  de-facto standard estimator for continuous MI.
- Two backends ship behind a `backend` switch on
  `mutual_information_matrix`: `"numba"` (default when available) runs a
  hand-rolled parallel KSG kernel and is typically 5–20× faster than the
  `"sklearn"` reference path (which delegates to
  `sklearn.feature_selection.mutual_info_regression`). Both produce KSG
  algorithm-1 estimates and agree to numerical tolerance; the numba kernel
  is `O(n^2)` per pair while sklearn uses an `O(n log n)` k-d tree, so the
  edge shrinks at very large `n`.
- Entries are then mapped through Linfoot's informational correlation
  coefficient `rho_I = sqrt(1 - exp(-2 I))` (Linfoot, *Inf. Control* 1(1),
  1957), so the matrix lives in `[0, 1]` and collapses to `|Pearson rho|`
  exactly under joint Gaussianity. The MI path is therefore a strict
  generalisation of the correlation path.
- The LoGo aggregation step still uses the empirical (or user-supplied)
  covariance matrix, because clique-wise inversion requires an actual
  covariance, not an MI matrix.
- `MFCFLoGoCV` and `MFCFLoGoCVAll` expose the same arguments. The MI matrix
  is computed once per training fold and shared across the
  `max_clique_size` grid.
- You can also build the MI matrix yourself and pass it through:

  ```python
  from mutual_information import mutual_information_matrix

  M = mutual_information_matrix(X, n_neighbors=3, normalize="linfoot")
  MFCFLoGo(similarity="mutual_information").fit(X, mi_matrix=M)
  ```

Note that KSG is `O(p^2 * n * log n)` and materially slower than
`np.corrcoef`; use `mi_n_jobs=-1` on large feature counts.

### Hierarchical Clustering with DBHT (`mfcf_dbht()`)
```python
import numpy as np
from scipy.spatial.distance import pdist, squareform
from mfcf_dbht import mfcf_dbhts as mfcf_dbht

X = ...  # samples x features
D = squareform(pdist(X, metric="euclidean"))
S = np.exp(-D)  # any similarity aligned with D

clusters, Rpm, Adjv, Dpm, Mv, Z = mfcf_dbht(
    D,
    S,
    threshold=0.1,
    min_clique_size=2,
    max_clique_size=6,
)
```
