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

#### Two kinds of input

MFCF works either on a **covariance or correlation matrix** (dependency
networks, precision estimation) or on a **general weight / similarity matrix**
(network filtering, clustering). The gain and the attachment rule are chosen
accordingly:

| Input `C` | `gain_function_type` | Gain of attaching `v` to `S` | `acceptance` |
|---|---|---|---|
| covariance / correlation | `"loglik"` (default) | exact `-1/2 log(1 - R^2_{v|S})` | `"threshold"`, `"ftest"`, `"bic"` |
| any weight matrix | `"sumsquares"` | `sum_{u in S} C[v,u]^2` | `"threshold"` |
| non-negative weight matrix (checked) | `"sum"` | `sum_{u in S} C[v,u]` | `"threshold"` |

With `"loglik"`, `S` is grown greedily inside a clique (or facet), adding at
each step the vertex with the largest squared partial correlation with `v`.
Each step is accepted by:

- `acceptance="threshold"` (default): squared partial correlation `>= threshold`
  (at the first step, the squared correlation);
- `acceptance="ftest"`: partial F-test of the added coefficient at level
  `alpha`, given `n_samples` observations;
- `acceptance="bic"`: `n_samples * (-log(1 - rho^2)) > log(n_samples)`.

The statistical rules scale with the sample size and with the number of
variables already in `S`, so they give a validated, data-driven sparsity. The
mandatory `min_clique_size - 1` picks bypass the rule. A vertex starts a new
component when no attachment is accepted.

Because the in-sample likelihood can only grow with `S`, `threshold=0` accepts
every step: every clique then grows to `max_clique_size`, and the result is the
fixed-size construction. Variable clique sizes come from `threshold > 0` or a
statistical rule.

With the weight gains, `threshold` applies to each `C[v,u]^2` ("sumsquares") or
`C[v,u]` ("sum"), exactly as in earlier versions. Weights must be non-negative:
"sum" raises an error for negative off-diagonal entries, and "sumsquares" is
"sum" applied to the non-negative weights `C**2`, so signed correlations can
be passed to it directly. `"loglik"` checks that `C` is
a valid covariance or correlation matrix and raises an error otherwise. For a
weight matrix, `run()` returns `None` for the LoGo matrix unless `cov_matrix`
is given.

```python
# dependency network with statistically validated attachments
cliques, seps, peo, J = MFCF(max_clique_size=10, acceptance="ftest",
                             alpha=0.01, n_samples=X.shape[0]).run(C=np.corrcoef(X, rowvar=False))
# filtering of a general similarity matrix
cliques, seps, peo, _ = MFCF(gain_function_type="sum", threshold=0.2,
                             max_clique_size=4).run(C=W)
```

The LoGo estimators (`MFCFLoGo`, `MFCFLoGoCV`, `MFCFLoGoCVAll`) accept the
same options and take `n_samples` from the data; `MFCFLoGoCVAll` can tune
`alpha` (`tunable_params=("alpha", ...)`).

The exact gain measures the conditional information that an attachment adds,
so it selects sparser and more accurate networks when the dependence structure
is sparse; on dense data the two covariance-based options perform similarly.
Gains are cached per separator, and for fixed clique sizes all facets of a
clique are scored from a single regression, so the exact gain runs at a cost
comparable to the approximation (faster for fixed clique sizes). To reproduce
results of earlier versions, pass `gain_function_type="sumsquares"`, the
former default.

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
