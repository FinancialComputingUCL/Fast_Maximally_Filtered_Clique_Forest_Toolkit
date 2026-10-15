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
(verified on random configurations without exact ties, coordination caps or
`drop_sep` combined with a threshold, where the fixes above apply). Earlier runs of `"sumsquares"` on
other weight matrices are no longer accepted; use `"sum"` on non-negative
weights.

### Added
- Merge of the `mutual_info` branch (experimental): `mutual_information` module (KSG and
  histogram estimators, Linfoot normalisation; A. Briola) and the
  `similarity` / `mi_*` options of `MFCFLoGo`, with its `fit(X, corr_matrix,
  mi_matrix, cov_matrix)` interface; `MFCF.fit(X, similarity=
  "mutual_information")`. `"sumsquares"` accepts any matrix of correlation
  coefficients (positive semi-definiteness is needed only by `"loglik"`).
- Options of the original `fast_fast_mfcf`: `drop_sep`, `seed="tmfg"`, and
  its interface `MFCF().fast_mfcf(C, mfcf_control(), gf_type)`
  (deprecated).
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
- `alpha` can be tuned by `MFCFLoGoCVAll`, which records its trials in
  `cv_results_` and the tuned values in `best_params_`.
- Reference check of the cross-validated estimators
  (`src/test_mfcf_cv_reference.py`): fold scores, selection and refit of
  `MFCFLoGoCV` and `MFCFLoGoCVAll` equal an independent re-implementation
  exactly, across gains, acceptance rules, shrinkage, rank correlations,
  `n_effective`, `n_starts`, shuffling and failing folds.
- Installable package (`pyproject.toml`; core needs only numpy and scipy,
  optuna is imported only by `MFCFLoGoCVAll`); input validation with
  informative errors; version number;
  scaling benchmark (`benchmarks/`); about 2400 tests (validation against the
  reference implementation, properties, robustness, scikit-learn checks).

### Fixed
- A facet was dropped from the queue when the separator kept for one of its
  vertices had reached `coordination_number` (or was used, with `drop_sep`),
  although other vertices could still attach to it; the result then
  depended on the order of the variables. The facet is now kept for its
  other vertices (the original implementation had the same behaviour, so
  with a cap or `drop_sep` and a positive threshold the legacy interface no
  longer reproduces it exactly).
- Seed and new-component rules on the correlation scale for `"loglik"`, so a
  covariance and its correlation give the same forest; exact ties in the
  seed rule broken by total strength (label-independent).
- With the F-test or BIC, as with the threshold rule, a vertex attaches only if
  its total gain reaches that of one accepted step (the critical value of the
  first test); with `min_clique_size = 2` pure noise is then attached only
  through pairs that pass the test. With `min_clique_size >= 3` the mandatory
  picks add up and can still exceed it, so noise can form structure; and each
  test is applied to the best of many candidates without a correction for
  that selection, so the statistical rules control sparsity only roughly.
- An exactly collinear vertex (e.g. a duplicated column) with
  `min_clique_size >= 3` and a threshold or test: the mandatory picks now
  complete, so it attaches to its copy, and a facet blocked for one vertex
  stays available to the others.
- The TMFG seed is also computed on the correlation scale for `"loglik"`.
- `run()` validates `cov_matrix` (finite, symmetric).
- `mfcf_dbht` rejects a similarity that is zero on backbone edges (the
  weighted graph would no longer be maximally planar).
- Kendall's tau uses bounded memory for large p.
- Numerically singular clique blocks (e.g. fewer observations than the clique
  size) give no precision (`None` and a RuntimeWarning in `MFCF`, an
  informative error in `MFCFLoGo`) instead of a meaningless inverse; use
  `shrinkage > 0`.
- Validation of `min_clique_size <= max_clique_size`, integer
  `coordination_number >= 1`, `threshold >= 0` (`<= 1` for `"loglik"`),
  `seed="tmfg"` with `min_clique_size <= 5`; `MFCFLoGoCVAll` no longer
  proposes `max_clique_size < min_clique_size`.
- `MFCFLoGoCV` passed `random_state` to `KFold` without shuffling, so it
  failed scikit-learn's estimator checks (all checks now pass for the three
  estimators).
- `MFCFLoGo.fit(X, corr_matrix=..., cov_matrix=...)` takes the number of
  observations from `X` for the statistical rules.
- `MFCF.fit(similarity="mutual_information")` no longer changes the gain of
  the instance for later fits; the legacy `fast_mfcf` no longer overwrites
  the configuration of the calling instance.
- Kendall's tau on scipy < 1.11; the F-test does not charge a degree of
  freedom for the mean with `assume_centered=True`; `run()` accepts any
  array-like; `"sum"` rejects non-finite or asymmetric weights; DBHT rejects
  a signed similarity.
- `MFCFLoGoCV` never selects a clique size whose folds failed (also with a
  finite `error_score`); it raises if all failed. Configuration errors are
  reported before cross-validation; the grid must contain integers.
- `MFCFLoGoCVAll`: a trial with failed folds is a failed trial, never selected
  and no longer aborting the search (it crashed when one trial failed every
  fold); tuned values in `best_params_`; `n_trials=0`/`None` refits with the fixed parameters (as
  documented); `random_state` also seeds the Optuna sampler (reproducible
  search); `error_score` is honoured; unknown `tunable_params` raise; the
  clique-size range reaches `min(32, n_features)`.
- A numeric `n_effective` in the cross-validated estimators refers to the full
  sample and is scaled to each training fold.
- `n_effective="auto"` uses autocorrelations about the sample mean (also with
  `assume_centered=True`); with shrinkage, starts are compared on the shrunk
  model.
- `mfcf_dbht` requires the backbone its DBHT steps assume (planar: 4-cliques
  joined by triangles, `max_clique_size=4`, `coordination_number=1`, now the
  default) and validates D and S (shapes, finite values, signs, p >= 4; a
  NaN made it hang); it raises otherwise; earlier claims that it
  ran on any MFCF backbone were wrong (other sizes collapsed to one cluster
  or crashed).
- Mutual information: the KSG estimate is computed on standardised columns
  (it depended on the units of the variables; the numba and sklearn backends
  agree on continuous data up to neighbour counts at exact distance ties,
  about 1e-3 nats; on discrete or tied data KSG is unreliable and the two
  backends can differ by about 0.1);
  `n_neighbors` is validated (0 crashed the interpreter); scikit-learn is
  needed only by the sklearn KSG backend; constant columns are rejected by
  both estimators; the histogram estimator needs 4 observations per bin; tied values share a histogram bin
  (independent discrete variables looked dependent) and the Miller-Madow
  correction counts the occupied bins; NaN inputs are
  rejected; the default jitter is reproducible.
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
  sum-of-squares gain (p = 1600, K = 8, one start, threshold 0: a few
  seconds; the F-test about ten times longer; the default
  `n_starts = max_clique_size` multiplies these by K).
