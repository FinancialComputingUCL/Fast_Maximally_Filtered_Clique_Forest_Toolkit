# Changelog

## 2026.10.0

### Changed (results may differ from earlier versions)
- **Default gain is the exact Gaussian log-likelihood gain**
  (`gain_function_type="loglik"`, `-1/2 log(1 - R^2_{v|S})`), replacing the
  sum of squared correlations (`"sumsquares"`). Leaving the option unset
  issues a `FutureWarning` in this release.
- **Gains are matched to the input.** `"loglik"` and `"sumsquares"` require a
  covariance or correlation matrix; `"sum"` requires non-negative weights.
  `mfcf_dbht`, whose input is a similarity matrix, now uses `"sum"` (formerly
  `"sumsquares"`).
- **New components start from the strongest vertex**
  (`new_component="strongest"`), so the clique forest no longer depends on the
  order of the variables. `new_component="first"` restores the former
  lowest-index rule.
- `MFCFLoGo`, `MFCFLoGoCV` and `MFCFLoGoCVAll` return `precision_` and
  `covariance_` on the covariance scale of the data (formerly the precision
  of the correlation matrix).

For covariance or correlation input, the results of earlier versions are
reproduced exactly with `gain_function_type="sumsquares", new_component="first"`
(verified on 163 random configurations). Earlier runs of `"sumsquares"` on
other weight matrices are no longer accepted; use `"sum"` on non-negative
weights.

### Added
- Statistical acceptance of each attachment: partial F-test
  (`acceptance="ftest"`, level `alpha`) and BIC (`acceptance="bic"`), as
  alternatives to the fixed `threshold`.
- `"sum"` gain for non-negative weight matrices.
- `MFCF.fit(X)`: builds the clique forest and the decomposable maximum
  likelihood precision directly from data, with optional block shrinkage,
  effective sample size for autocorrelated series (`n_effective="auto"`), and
  rank-based Gaussian-copula correlations (`correlation="spearman"` or
  `"kendall"`).
- `n_starts`: runs the greedy expansion from several seeds spread over the
  network and keeps the clique forest with the largest objective. Default:
  `max_clique_size` starts for the `"loglik"` gain (better fit to new data
  and better network recovery, at K times the cost), one start for the
  weight gains.
- `alpha` can be tuned by `MFCFLoGoCVAll`.
- Installable package (`pyproject.toml`; core needs only numpy and scipy,
  optuna is imported only by `MFCFLoGoCVAll`); input validation with
  informative errors; version number;
  scaling benchmark (`benchmarks/`); about 2150 tests (validation against the
  reference implementation, properties, robustness, scikit-learn checks).

### Fixed
- Crash when `min_clique_size` exceeds the number of variables.
- `MFCFLoGoCVAll` ignored `cv` and the gain setting.
- `mfcf_dbht`: missing `squareform` import, asymmetric distance passed to
  `optimal_leaf_ordering`; a disconnected backbone now raises an error.
- `run()` returns `J=None` instead of inverting singular clique blocks.
- README examples and the final cell of the LoGo notebook.

### Performance
- Direct path for network generation (`"sumsquares"` / `"sum"` gain,
  `threshold=0`, `min_clique_size <= 1`), ported from the `mutual_info`
  branch: Numba kernels replace the priority queue, about 2x faster, same
  clique forest as the general path (exact ties may be broken differently).
  Used when numba is installed (extra `fast`).
- Positive semi-definiteness checked by a Cholesky factorisation instead of
  an eigendecomposition (about 10x faster for large p).
- The log-likelihood gain is cached per separator (bounded memory) and uses
  closed-form facet updates; it runs at a cost comparable to the
  sum-of-squares gain (p = 1600, K = 8: about 4 s).
