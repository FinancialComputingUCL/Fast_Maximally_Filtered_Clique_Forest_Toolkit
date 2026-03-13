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
