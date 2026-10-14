# Fast Maximally Filtered Clique Forest Toolkit

Accelerated Maximally Filtered Clique Forest (MFCF) implementations for sparse precision estimation and hierarchical clustering. The core `fast_fast_mfcf` routine is ~100× faster than the original reference implementation while preserving the same outputs, enabling practical experimentation with large correlation or covariance matrices. See [this repo](https://github.com/yanh11/fast_fast_mfcf) for more details.

## Introduction
- Build MFCF backbones from dense similarity matrices in a few seconds.
- Drop-in graphical-model estimators (`MFCFLoGO`, `MFCFLoGOCV`, `MFCFLoGOCVAll`) that follow the scikit-learn API and offer a faster, more accurate alternative to Graphical Lasso when the sample-to-feature ratio is small.
- Extend Riskfolio-Lib’s Direct Bubble Hierarchical Tree (DBHT) pipeline so it can operate over any MFCF backbone instead of only TMFG graphs (`mfcf_dbht`).

## Getting Started
- Python 3.9+ recommended.
- Clone the repository and install the package (numpy and scipy only), with
  optional extras: `sklearn` for the LoGo estimators, `cv` for
  `MFCFLoGoCVAll` (adds optuna), `dbht` for `mfcf_dbht` (riskfolio-lib),
  `test` for the test suite:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install ".[sklearn]"        # or ".[cv,dbht]", ".[test]"
python -m pytest -q             # from the repository root, with ".[test]"
```

  `pip install -r requirements.txt` still installs every optional dependency.
  Version history: [CHANGELOG.md](CHANGELOG.md).

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
| covariance / correlation (checked) | `"loglik"` (default) | exact `-1/2 log(1 - R^2_{v|S})` | `"threshold"`, `"ftest"`, `"bic"` |
| covariance / correlation (checked) | `"sumsquares"` | `sum_{u in S} C[v,u]^2`, the original approximation | `"threshold"` |
| non-negative weight / similarity matrix (checked) | `"sum"` | `sum_{u in S} C[v,u]` | `"threshold"` |

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

With `"sumsquares"` and `"sum"`, `threshold` applies to each `C[v,u]^2` or
`C[v,u]`, exactly as in earlier versions. `"loglik"` and `"sumsquares"` check
that `C` is a valid covariance or correlation matrix (signed entries allowed);
`"sum"` checks that the weights are non-negative. Each raises an error
otherwise. For a weight matrix, `run()` returns `None` for the LoGo matrix
unless `cov_matrix` is given.

```python
import numpy as np
from fast_fast_mfcf import MFCF

# dependency network with statistically validated attachments
cliques, seps, peo, J = MFCF(max_clique_size=10, acceptance="ftest",
                             alpha=0.01, n_samples=X.shape[0]).run(C=np.corrcoef(X, rowvar=False))
# filtering of a general similarity matrix
cliques, seps, peo, _ = MFCF(gain_function_type="sum", threshold=0.2,
                             max_clique_size=4).run(C=W)
```

#### Building directly from data (`MFCF.fit`)

For likelihood and covariance tasks the builder can take the data matrix
(observations in rows, e.g. a multivariate time series of returns):

```python
from fast_fast_mfcf import MFCF

m = MFCF(max_clique_size=10, acceptance="ftest", alpha=0.01).fit(
    X, shrinkage=0.05, n_effective="auto")
m.cliques_, m.separators_, m.peo_          # clique forest
m.precision_, m.covariance_               # on the covariance scale of X
m.n_effective_                            # observations used by the test
```

- The structure is built from the sample correlation, so it does not depend
  on the scales of the variables; `precision_` is the LoGo assembly of the
  (optionally shrunk) correlation blocks rescaled by the sample standard
  deviations, i.e. the decomposable maximum-likelihood estimate when
  `shrinkage=0`.
- `n_samples` is set from `X`. For autocorrelated series the statistical
  tests would otherwise be too permissive: `n_effective="auto"` uses
  Bartlett's correction for sample correlations,
  `n / mean_{i<j} (1 + phi_i phi_j) / (1 - phi_i phi_j)` with `phi` the lag-1
  autocorrelations; a number can also be given.
- `correlation="spearman"` or `"kendall"` replaces the Pearson correlation
  by the Gaussian-copula correlation estimated from ranks
  (`2 sin(pi rho_s / 6)` or `sin(pi tau / 2)`, projected onto the positive
  definite matrices if needed). The structure is then invariant to monotone
  transformations of the variables and robust to heavy tails and outliers
  (e.g. daily returns); the statistical tests hold approximately. Spearman
  costs about the same as Pearson and is the choice for long series; Kendall
  costs O(p^2 n log n) (about 4 s for p = 50, n = 20000; 0.03 s with
  Spearman).
- Assumptions: approximately stationary series over the sample (use returns,
  not prices). Data preparation (missing values, windows) is left to the user.
- `fit(X)` works with the `"loglik"` gain (default) or `"sumsquares"` (the
  original construction on the correlation matrix); for other weight
  matrices use `run(C)`.

The LoGo estimators (`MFCFLoGo`, `MFCFLoGoCV`, `MFCFLoGoCVAll`) are built on
`fit(X)` and accept the same options (`acceptance`, `alpha`, `new_component`,
`n_starts`, `shrinkage`, `n_effective`, `correlation`); `MFCFLoGoCVAll` can tune `alpha`
(`tunable_params=("alpha", ...)`). Their `precision_` and `covariance_` are on
the covariance scale of the data (earlier versions returned the precision of
the correlation matrix).

The exact gain measures the conditional information that an attachment adds,
so it selects sparser and more accurate networks when the dependence structure
is sparse; on dense data the two covariance-based options perform similarly.
Gains are cached per separator, and for fixed clique sizes all facets of a
clique are scored from a single regression, so the exact gain runs at a cost
comparable to the approximation (faster for fixed clique sizes). To reproduce
results of earlier versions exactly on a covariance or correlation matrix,
pass `gain_function_type="sumsquares"` (the former default) and
`new_component="first"`.

#### Starting new components (`new_component`)

When no vertex can be attached (its gain is below the threshold, or it fails
the statistical test), it starts a new component of the clique forest. With
`new_component="strongest"` (default) the vertex with the largest total
dependence on the remaining vertices is chosen, so the result does not depend
on the order of the variables. `new_component="first"` takes the vertex with
the lowest index, as earlier versions did.

#### Several starts (`n_starts`)

The expansion is greedy, so its result depends on where it starts.
`n_starts=k` (at most `max_clique_size`) runs it from the default seed and
from `k - 1` further seeds spread over the network (strong vertices weakly
related to the seeds already chosen), and keeps the clique forest with the
largest objective: the decomposable Gaussian log-likelihood for `"loglik"`,
the total (squared) edge weight for `"sum"` (`"sumsquares"`). The chosen start
is in `start_`. It costs `k` runs. In simulations of chordal, factor and
sparse models with the `"loglik"` gain, `k = max_clique_size` starts improved
the test log-likelihood (by 0.1-0.4 per observation on chordal and factor
models) and the recovery of the true network, with the F-test, BIC or a
threshold alike, capturing 60-90% of the gain of trying every vertex as a
start. With `"sumsquares"` and `"sum"` the gain was negligible. The default
(`n_starts=None`) is therefore `max_clique_size` starts for `"loglik"` and a
single start for the weight gains. Pass `n_starts=1` for the fastest run.

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

The backbone is built on a non-negative similarity `S` with the `"sum"` gain
(default); for a signed correlation matrix pass `gain_function_type="loglik"`
or `"sumsquares"`.

```python
import numpy as np
from scipy.cluster.hierarchy import fcluster
from mfcf_dbht import mfcf_dbhts

X = ...  # samples x features
R = np.corrcoef(X, rowvar=False)
D = np.sqrt(np.clip(2 * (1 - R), 0, None))       # distance between features
np.fill_diagonal(D, 0.0)
S = (1 + R) / 2                                  # non-negative similarity, 1 - D**2 / 4

T8, Rpm, Adjv, Dpm, Mv, Z = mfcf_dbhts(D, S, min_clique_size=2, max_clique_size=6)
labels = fcluster(Z, t=4, criterion="maxclust")  # e.g. four clusters
```
