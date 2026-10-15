import warnings
from dataclasses import dataclass
from typing import Iterable, List, Optional, Union, Dict, Any

import numpy as np
import numpy.linalg as LA
from sklearn.covariance import EmpiricalCovariance
from sklearn.covariance import empirical_covariance, log_likelihood
from sklearn.model_selection import KFold, cross_val_score
from sklearn.utils.validation import check_array, check_is_fitted

from fast_fast_mfcf import MFCF as _MFCFBuilder


def _check_configuration(est, n_samples: int, max_clique_size: int) -> None:
    """Fail early on configuration errors, which cross-validation would
    otherwise report as failed folds."""
    if est.similarity not in ("correlation", "mutual_information"):
        raise ValueError("similarity must be 'correlation' or 'mutual_information'")
    if est.correlation not in ("pearson", "spearman", "kendall"):
        raise ValueError("correlation must be 'pearson', 'spearman' or 'kendall'")
    ne = est.n_effective
    if not (ne is None or (isinstance(ne, str) and ne == "auto")):
        if isinstance(ne, str) or not 3 <= float(ne) <= n_samples:
            raise ValueError("n_effective must be None, 'auto' or a number between 3 and n_samples")
    n_starts = est.n_starts if est.n_starts is None else min(est.n_starts, max_clique_size)
    _MFCFBuilder(threshold=est.threshold, min_clique_size=est.min_clique_size,
                 max_clique_size=max_clique_size, coordination_number=est.coordination_number,
                 gain_function_type=est.gain_function_type, acceptance=est.acceptance,
                 alpha=est.alpha, new_component=est.new_component, n_starts=n_starts)


def _fold_n_effective(n_effective, n_train: int, n_total: int):
    """A numeric n_effective refers to the full sample: scale it to a fold."""
    if n_effective is None or isinstance(n_effective, str):
        return n_effective
    return max(3.0, float(n_effective) * n_train / n_total)


class MFCFLoGo(EmpiricalCovariance):
    """
    A precision and covariance estimator with MFCF-LoGo algorithm. It constructs
    an MFCF first then use the LoGo algorithm to obtain a precision matrix.

    Precision matrix is the LoGo; estimated covariance is its inverse.

    Parameters
    ----------
    threshold : float, default=0.0
        Gain threshold controlling attachment vs. new component.

    min_clique_size : int, default=1
        The minimum size of cliques to form in the graph.

    max_clique_size : int, default=4
        The maximum size of cliques to form in the graph.

    coordination_number : float, default=np.inf
        Maximum allowed uses of a separator (multiplicity cap).

    gain_function_type : {'loglik', 'sumsquares', 'sum'}, default=None
        None means 'loglik' (with a FutureWarning in this release: the default
        was 'sumsquares' until version 2026.10).
        Gain function type to use in MFCF construction. The function is used to
        determine which vertex to add next to maximize the overall gain in the
        graph.

    acceptance : {'threshold', 'ftest', 'bic'}, default='threshold'
        Rule for accepting each attachment step. 'ftest' (partial F-test at
        level `alpha`) and 'bic' are statistical validations for the 'loglik'
        gain; the number of observations is taken from the data.

    alpha : float, default=0.05
        Level of the partial F-test (acceptance='ftest').

    new_component : {'strongest', 'first'}, default='strongest'
        Vertex that starts a new component (see `MFCF`).

    n_starts : int or None, default=None
        Number of starting cliques, at most `max_clique_size`; the clique
        forest with the largest objective is kept. None: `max_clique_size`
        for the 'loglik' gain, 1 otherwise (see `MFCF`).

    shrinkage : float, default=0.0
        Identity shrinkage of the correlation blocks of the LoGo assembly.

    n_effective : None, float or 'auto', default=None
        Number of observations for the statistical acceptance rules; 'auto'
        corrects for autocorrelation of time series (see `MFCF.fit`).

    correlation : {'pearson', 'spearman', 'kendall'}, default='pearson'
        Correlation used for the structure and the precision; the rank options
        estimate the Gaussian-copula correlation (see `MFCF.fit`).

    similarity : {'correlation', 'mutual_information'}, default='correlation'
        Dependence measure for the structure. 'mutual_information' uses the
        pairwise mutual information (Linfoot normalisation) and requires the
        'sumsquares' or 'sum' gain; the precision is assembled from the
        correlation blocks (see `MFCF.fit`).

    mi_estimator, mi_n_neighbors, mi_n_bins, mi_normalize, mi_n_jobs, mi_random_state
        Options of the mutual-information estimator ('ksg' or 'histogram').

    assume_centered : bool, default=False
        If True, data are not centered before computation.
        Useful when working with data whose mean is almost, but not exactly
        zero.
        If False, data are centered before computation.

    Attributes
    ----------
    covariance_ : ndarray of shape (n_features, n_features)
        Inverse of the LoGo precision estimate.

    precision_ : ndarray of shape (n_features, n_features)
        The LoGo estimator (sparse inverse assembled from cliques and separators).

    location_ : ndarray of shape (n_features,)
        Estimated mean (0 if assume_centered=True).

    cliques_ : list of frozenset[int]
        Maximal cliques found by MFCF.

    separators_count_ : Counter[frozenset[int], int]
        Separator multiplicities used in the constructed MFCF.

    peo_ : list[int]
        Perfect elimination order of vertices produced by the algorithm.
    """

    def __init__(
        self,
        *,
        threshold: float = 0.0,
        min_clique_size: int = 1,
        max_clique_size: int = 4,
        coordination_number: int = np.inf,
        gain_function_type: Optional[str] = None,
        acceptance: str = "threshold",
        alpha: float = 0.05,
        new_component: str = "strongest",
        n_starts: Optional[int] = None,
        shrinkage: float = 0.0,
        n_effective=None,
        correlation: str = "pearson",
        similarity: str = "correlation",
        mi_estimator: str = "ksg",
        mi_n_neighbors: int = 3,
        mi_n_bins="auto",
        mi_normalize: str = "linfoot",
        mi_n_jobs: Optional[int] = None,
        mi_random_state: Optional[int] = None,
        assume_centered: bool = False,
    ):
        super().__init__(assume_centered=assume_centered)
        self.threshold = threshold
        self.min_clique_size = min_clique_size
        self.max_clique_size = max_clique_size
        self.coordination_number = coordination_number
        self.gain_function_type = gain_function_type
        self.acceptance = acceptance
        self.alpha = alpha
        self.new_component = new_component
        self.n_starts = n_starts
        self.shrinkage = shrinkage
        self.n_effective = n_effective
        self.correlation = correlation
        self.similarity = similarity
        self.mi_estimator = mi_estimator
        self.mi_n_neighbors = mi_n_neighbors
        self.mi_n_bins = mi_n_bins
        self.mi_normalize = mi_normalize
        self.mi_n_jobs = mi_n_jobs
        self.mi_random_state = mi_random_state

    def _fit_precomputed(self, builder, X, corr_matrix, mi_matrix, cov_matrix):
        """Interface of the `mutual_info` branch: run on precomputed matrices."""
        from sklearn.covariance import empirical_covariance
        p = X.shape[1]
        for name, M in (("corr_matrix", corr_matrix), ("mi_matrix", mi_matrix), ("cov_matrix", cov_matrix)):
            if M is not None and np.shape(M) != (p, p):
                raise ValueError(f"{name} must have shape ({p}, {p}), matching X")
        if self.similarity == "correlation" and mi_matrix is not None:
            raise ValueError("mi_matrix is used only with similarity='mutual_information'")
        if self.similarity == "mutual_information" and corr_matrix is not None:
            raise ValueError("corr_matrix is used only with similarity='correlation'")
        if self.similarity == "mutual_information" and builder._gain_unset:
            builder._gf_type, builder._gain_unset = "sumsquares", False
            if builder._n_starts_auto:
                builder._n_starts = 1
        if self.similarity == "mutual_information" and builder._gf_type == "loglik":
            raise ValueError("similarity='mutual_information' needs gain_function_type='sumsquares' "
                             "or 'sum' (the information matrix is not a covariance)")
        if self.similarity == "correlation":
            C = corr_matrix if corr_matrix is not None else np.corrcoef(X, rowvar=False)
            cov = cov_matrix if cov_matrix is not None else C
        elif self.similarity == "mutual_information":
            if mi_matrix is not None:
                C = mi_matrix
            else:
                from mutual_information import mutual_information_matrix
                C = mutual_information_matrix(
                    X, estimator=self.mi_estimator, n_neighbors=self.mi_n_neighbors,
                    n_bins=self.mi_n_bins, normalize=self.mi_normalize,
                    n_jobs=self.mi_n_jobs, random_state=self.mi_random_state)
            cov = (cov_matrix if cov_matrix is not None
                   else empirical_covariance(X, assume_centered=self.assume_centered))
        else:
            raise ValueError("similarity must be 'correlation' or 'mutual_information'")
        if builder._n_samples is None:
            builder._n_samples = float(X.shape[0])     # for the statistical rules
        cliques, seps, peo, J = builder.run(C=C, cov_matrix=cov)
        self.cliques_, self.separators_count_, self.peo_ = cliques, seps, peo
        self.n_effective_ = float(X.shape[0])
        self.precision_ = J

    def fit(self, X: np.ndarray, y=None, corr_matrix: Optional[np.ndarray] = None,
            mi_matrix: Optional[np.ndarray] = None,
            cov_matrix: Optional[np.ndarray] = None) -> "MFCFLoGo":
        """Fit the estimator from data X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data from which to compute the covariance estimate.

        y : Ignored
            Not used, present for API consistency by convention.

        corr_matrix, mi_matrix, cov_matrix : np.ndarray, optional
            Precomputed matrices (interface of the `mutual_info` branch):
            the similarity for the structure (`corr_matrix` with
            similarity='correlation', `mi_matrix` with
            similarity='mutual_information') and the matrix whose blocks are
            assembled into the precision (`cov_matrix`). With `corr_matrix` or
            `cov_matrix` the precision is on the scale of that matrix (by
            default the correlation, or the empirical covariance for mutual
            information) and the options `shrinkage`, `n_effective` and
            `correlation` are not used.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        X = check_array(
            X,
            ensure_min_samples=2,
            ensure_min_features=1,
            dtype=[np.float64, np.float32],
        )
        # track feature count for sklearn API
        self.n_features_in_ = X.shape[1]
        self.location_ = (
            np.zeros(self.n_features_in_, dtype=X.dtype)
            if self.assume_centered
            else X.mean(axis=0)
        )

        builder = _MFCFBuilder(
            threshold=self.threshold,
            min_clique_size=self.min_clique_size,
            max_clique_size=self.max_clique_size,
            coordination_number=self.coordination_number,
            gain_function_type=self.gain_function_type,
            acceptance=self.acceptance,
            alpha=self.alpha,
            new_component=self.new_component,
            n_starts=self.n_starts,
        )
        if corr_matrix is not None or cov_matrix is not None:
            self._fit_precomputed(builder, X, corr_matrix, mi_matrix, cov_matrix)
        else:
            # structure from the chosen similarity; precision on the covariance
            # scale of X (LoGo of the correlation blocks rescaled by the variances)
            builder.fit(X, assume_centered=self.assume_centered,
                        shrinkage=self.shrinkage, n_effective=self.n_effective,
                        correlation=self.correlation, similarity=self.similarity,
                        mi_estimator=self.mi_estimator, mi_n_neighbors=self.mi_n_neighbors,
                        mi_n_bins=self.mi_n_bins, mi_normalize=self.mi_normalize,
                        mi_n_jobs=self.mi_n_jobs, mi_random_state=self.mi_random_state,
                        mi_matrix=mi_matrix)
            self.cliques_ = builder.cliques_
            self.separators_count_ = builder.separators_
            self.peo_ = builder.peo_
            self.n_effective_ = builder.n_effective_
            self.precision_ = builder.precision_
        if self.precision_ is None:
            raise ValueError("No precision estimate: some clique blocks are singular (e.g. fewer "
                             "observations than the clique size); set shrinkage > 0 or lower "
                             "max_clique_size")
        if self.store_precision and not np.all(np.isfinite(self.precision_)):
            raise ValueError("Computed precision_ contains non-finite values.")

        # Robust inversion (fallback ridge if needed)
        try:
            cov = LA.inv(self.precision_)
        except LA.LinAlgError:
            # minimal Tikhonov regularization
            eps = 1e-8 * np.trace(self.precision_) / self.precision_.shape[0]
            cov = LA.inv(self.precision_ + eps * np.eye(self.precision_.shape[0]))
        self.covariance_ = cov

        return self

    def get_precision(self) -> np.ndarray:
        check_is_fitted(self, attributes=("precision_",))
        return self.precision_

    def get_covariance(self) -> np.ndarray:
        check_is_fitted(self, attributes=("covariance_",))
        return self.covariance_


@dataclass(frozen=True)
class _FoldScore:
    max_clique_size: int
    fold: int
    train_n: int
    val_n: int
    score: float
    success: bool
    error: Optional[str] = None


class MFCFLoGoCV(EmpiricalCovariance):
    """
    Cross-validated MFCF-based covariance/precision estimator.

    This estimator selects `max_clique_size` via K-fold cross-validation by
    maximizing the validation Gaussian log-likelihood:
        score = logdet(Theta) - trace(S_val @ Theta)
    where Theta is the LoGo precision estimated on the training fold, and S_val
    is the empirical covariance of the validation fold.

    Parameters
    ----------
    # Hyper-parameter to tune
    max_clique_size_grid : Iterable[int] or int, default=4
        Candidate values for `max_clique_size`. When an array-like object is
        provided, it is used as the grid. When an int is provided, it defines
        the upper bound of a grid, and the grid starts from 2. For example,
        `max_clique_size_grid=6` results in the grid [2, 3, 4, 5, 6].

    # Cross-validation
    cv : int, default=5
        Number of KFold splits.

    shuffle : bool, default=False
        Whether to shuffle samples before splitting.

    random_state : Optional[int], default=None
        Random state for the KFold when shuffle=True.

    # Scoring / stability
    assume_centered : bool, default=False
        Passed to MFCFLoGo and empirical covariance computation.

    error_score : {'raise', float}, default=np.nan
        Score to assign if a fit/eval fails for a fold/setting. If 'raise',
        the exception is raised. If a float, the error is caught and this score
        is used for that fold.

    # Parameters forwarded to MFCFLoGo
    threshold : float, default=0.0
        Gain threshold controlling attachment vs. new component.

    min_clique_size : int, default=1
        The minimum size of cliques to form in the graph.

    coordination_number : float, default=np.inf
        Maximum allowed uses of a separator (multiplicity cap).

    gain_function_type : {'loglik', 'sumsquares', 'sum'}, default=None
        None means 'loglik' (with a FutureWarning in this release: the default
        was 'sumsquares' until version 2026.10).
        Gain function type to use in MFCF construction. The function is used to
        determine which vertext to add next to maximize the overall gain in the
        graph.

    acceptance : {'threshold', 'ftest', 'bic'}, default='threshold'
        Rule for accepting each attachment step. 'ftest' (partial F-test at
        level `alpha`) and 'bic' are statistical validations for the 'loglik'
        gain; the number of observations is taken from the data.

    alpha : float, default=0.05
        Level of the partial F-test (acceptance='ftest').

    new_component : {'strongest', 'first'}, default='strongest'
        Vertex that starts a new component (see `MFCF`).

    n_starts : int or None, default=None
        Number of starting cliques, at most `max_clique_size`; the clique
        forest with the largest objective is kept. None: `max_clique_size`
        for the 'loglik' gain, 1 otherwise (see `MFCF`).

    assume_centered : bool, default=False
        If True, data are not centered before computation.
        Useful when working with data whose mean is almost, but not exactly
        zero.
        If False, data are centered before computation.


    Attributes
    ----------
    best_max_clique_size_ : int
        Selected value for `max_clique_size`.

    covariance_ : ndarray of shape (n_features, n_features)
        Inverse of the LoGo precision estimate.

    precision_ : ndarray of shape (n_features, n_features)
        LoGo precision estimated by refitting on the full dataset with the best
        `max_clique_size`.

    location_ : ndarray of shape (n_features,)
        Estimated mean (0 if assume_centered=True).

    cliques_ : list of frozenset[int]
        Maximal cliques found by MFCF.

    separators_count_ : Counter[frozenset[int], int]
        Separator multiplicities used in the constructed MFCF.

    peo_ : list[int]
        Perfect elimination order of vertices produced by the algorithm.

    n_features_in_ : int
        Number of features seen during fit.

    cv_results_ : dict
        Aggregated CV diagnostics with keys:
          - 'params': list of {'max_clique_size': int}
          - 'mean_test_score': np.ndarray, shape (n_settings,)
          - 'std_test_score':  np.ndarray, shape (n_settings,)
          - 'split_test_score': list of per-split arrays
          - 'n_splits': int

    fold_scores_ : list[_FoldScore]
        Per-fold detailed records (including errors if any).

    estimator_ : MFCFLoGo
        Final refit estimator on the full data.

    Notes
    -----
    * `max_clique_size` is naturally bounded: min ≥ `min_clique_size`,
      max ≤ n_features. The grid is filtered accordingly (and deduplicated).
    """

    def __init__(
        self,
        *,
        max_clique_size_grid: Union[Iterable[int], int] = 4,
        cv: int = 5,
        shuffle: bool = False,
        random_state: Optional[int] = None,
        # passthrough to base estimator
        threshold: float = 0.0,
        min_clique_size: int = 1,
        coordination_number: int = np.inf,
        gain_function_type: Optional[str] = None,
        acceptance: str = "threshold",
        alpha: float = 0.05,
        new_component: str = "strongest",
        n_starts: Optional[int] = None,
        shrinkage: float = 0.0,
        n_effective=None,
        correlation: str = "pearson",
        similarity: str = "correlation",
        mi_estimator: str = "ksg",
        mi_n_neighbors: int = 3,
        mi_n_bins="auto",
        mi_normalize: str = "linfoot",
        mi_n_jobs: Optional[int] = None,
        mi_random_state: Optional[int] = None,
        assume_centered: bool = False,
        # scoring / robustness
        error_score: Union[str, float] = np.nan,
    ):
        super().__init__(assume_centered=assume_centered)
        self.max_clique_size_grid = max_clique_size_grid
        self.cv = cv
        self.shuffle = shuffle
        self.random_state = random_state

        self.threshold = threshold
        self.min_clique_size = min_clique_size
        self.coordination_number = coordination_number
        self.gain_function_type = gain_function_type
        self.acceptance = acceptance
        self.alpha = alpha
        self.new_component = new_component
        self.n_starts = n_starts
        self.shrinkage = shrinkage
        self.n_effective = n_effective
        self.correlation = correlation
        self.similarity = similarity
        self.mi_estimator = mi_estimator
        self.mi_n_neighbors = mi_n_neighbors
        self.mi_n_bins = mi_n_bins
        self.mi_normalize = mi_normalize
        self.mi_n_jobs = mi_n_jobs
        self.mi_random_state = mi_random_state
        self.assume_centered = assume_centered

        self.error_score = error_score

    # ------------------------------ utils ------------------------------
    def _iter_grid(self, n_features: int) -> List[int]:
        if not isinstance(self.max_clique_size_grid, (int, np.integer)):
            vals = list(self.max_clique_size_grid)
            if any(isinstance(k, bool) or float(k) != int(k) for k in vals):
                raise ValueError("max_clique_size_grid must contain integers")
            grid = sorted({int(k) for k in vals})
        else:
            lower = max(2, int(self.min_clique_size))
            upper = min(self.max_clique_size_grid, int(n_features))
            grid = list(range(lower, upper + 1))
        # filter feasibility
        grid = [k for k in grid if self.min_clique_size <= k <= n_features and k >= 1]
        if not grid:
            raise ValueError(
                "Empty `max_clique_size_grid` after feasibility filtering. "
                f"min_clique_size={self.min_clique_size}, n_features={n_features}."
            )
        grid.sort()
        return grid

    # ------------------------------ API ------------------------------

    def fit(self, X: np.ndarray, y=None):
        """Fit the estimator from data X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data from which to compute the covariance estimate.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : object
            Returns the instance itself.
        """

        X = check_array(
            X,
            ensure_min_samples=max(self.cv, 2),
            ensure_min_features=1,
            dtype=[np.float64, np.float32],
        )
        n_samples, n_features = X.shape
        self.n_features_in_ = n_features
        self.location_ = (
            np.zeros(n_features, dtype=X.dtype)
            if self.assume_centered
            else X.mean(axis=0)
        )

        grid = self._iter_grid(n_features)

        # CV splitter
        _check_configuration(self, X.shape[0], max(grid))
        kf = KFold(n_splits=self.cv, shuffle=self.shuffle,
                   random_state=self.random_state if self.shuffle else None)

        split_scores_per_param: List[List[float]] = [[] for _ in grid]
        fold_records: List[_FoldScore] = []

        for fold_idx, (tr, va) in enumerate(kf.split(X)):
            X_tr, X_va = X[tr], X[va]
            S_val = empirical_covariance(X_va, assume_centered=self.assume_centered)

            for gi, k in enumerate(grid):
                try:
                    base = MFCFLoGo(
                        threshold=self.threshold,
                        min_clique_size=self.min_clique_size,
                        max_clique_size=k,
                        coordination_number=self.coordination_number,
                        gain_function_type=self.gain_function_type,
                        acceptance=self.acceptance,
                        alpha=self.alpha,
                        new_component=self.new_component,
                        n_starts=None if self.n_starts is None else min(self.n_starts, k),
                        shrinkage=self.shrinkage,
                        n_effective=_fold_n_effective(self.n_effective, len(tr), X.shape[0]),
                        correlation=self.correlation,
                        similarity=self.similarity,
                        mi_estimator=self.mi_estimator,
                        mi_n_neighbors=self.mi_n_neighbors,
                        mi_n_bins=self.mi_n_bins,
                        mi_normalize=self.mi_normalize,
                        mi_n_jobs=self.mi_n_jobs,
                        mi_random_state=self.mi_random_state,
                        assume_centered=self.assume_centered,
                    )
                    base.fit(X_tr)
                    precision_ = base.precision_
                    # numerical safety: if precision has inf/nan, mark failure
                    if not np.all(np.isfinite(precision_)):
                        raise ValueError("Non-finite entries in precision_).")

                    score = log_likelihood(S_val, precision_)
                    split_scores_per_param[gi].append(score)
                    fold_records.append(
                        _FoldScore(k, fold_idx, len(tr), len(va), score, True)
                    )
                except Exception as e:
                    if self.error_score == "raise":
                        raise
                    # assign error_score, continue

                    es = float(self.error_score)
                    split_scores_per_param[gi].append(es)
                    fold_records.append(
                        _FoldScore(k, fold_idx, len(tr), len(va), es, False, str(e))
                    )
                    # Optional: warn for visibility
                    warnings.warn(
                        f"Fold {fold_idx}, max_clique_size={k} failed with: {e}. "
                        f"Assigned error_score={self.error_score}.",
                        RuntimeWarning,
                        stacklevel=2,
                    )

        # Aggregate
        means = np.array([np.mean(s) for s in split_scores_per_param], dtype=float)
        stds = np.array(
            [np.std(s, ddof=1) if len(s) > 1 else 0.0 for s in split_scores_per_param],
            dtype=float,
        )

        # pick the best k among the grid values whose folds all succeeded
        # (a finite error_score must not make a failed value look best)
        ok = np.array([all(r.success for r in fold_records if r.max_clique_size == k) for k in grid])
        ok &= np.isfinite(means)
        if not ok.any():
            raise ValueError("Every max_clique_size in the grid failed in cross-validation "
                             "(e.g. fewer training observations than the clique size); "
                             "use shrinkage > 0 or smaller clique sizes")
        best_idx = int(np.argmax(np.where(ok, means, -np.inf)))
        best_k = int(grid[best_idx])

        # Store CV artifacts
        self.fold_scores_ = fold_records
        self.cv_results_ = {
            "params": [{"max_clique_size": int(k)} for k in grid],
            "mean_test_score": means,
            "std_test_score": stds,
            "split_test_score": split_scores_per_param,
            "n_splits": self.cv,
        }
        self.best_max_clique_size_ = best_k

        # Refit on all data with best parameter
        final = MFCFLoGo(
            threshold=self.threshold,
            min_clique_size=self.min_clique_size,
            max_clique_size=best_k,
            coordination_number=self.coordination_number,
            gain_function_type=self.gain_function_type,
            acceptance=self.acceptance,
            alpha=self.alpha,
            new_component=self.new_component,
            n_starts=None if self.n_starts is None else min(self.n_starts, best_k),
            shrinkage=self.shrinkage,
            n_effective=self.n_effective,
            correlation=self.correlation,
            similarity=self.similarity,
            mi_estimator=self.mi_estimator,
            mi_n_neighbors=self.mi_n_neighbors,
            mi_n_bins=self.mi_n_bins,
            mi_normalize=self.mi_normalize,
            mi_n_jobs=self.mi_n_jobs,
            mi_random_state=self.mi_random_state,
            assume_centered=self.assume_centered,
        )
        final.fit(X)

        self.covariance_ = final.covariance_
        self.precision_ = final.precision_
        self.cliques_ = final.cliques_
        self.separators_count_ = final.separators_count_
        self.peo_ = final.peo_
        self.estimator_ = final
        return self

    def get_precision(self) -> np.ndarray:
        check_is_fitted(self, attributes=("precision_",))
        return self.precision_

    def get_covariance(self) -> np.ndarray:
        check_is_fitted(self, attributes=("covariance_",))
        return self.covariance_


def log_likelihood_scorer(estimator, X, y=None) -> float:
    S_val = empirical_covariance(X, assume_centered=estimator.assume_centered)
    return log_likelihood(S_val, estimator.precision_)


class MFCFLoGoCVAll(EmpiricalCovariance):
    """
    Cross-validated MFCF-based covariance/precision estimator.

    This estimator selects `max_clique_size` via K-fold cross-validation by
    maximizing the validation Gaussian log-likelihood:
        score = logdet(Theta) - trace(S_val @ Theta)
    where Theta is the LoGo precision estimated on the training fold, and S_val
    is the empirical covariance of the validation fold.

    Parameters
    ----------
    # Hyperparameter tuning
    tunable_params : Iterable[str], default=()
        Names of parameters to tune with Optuna. Any subset of
        ``{"threshold", "min_clique_size", "max_clique_size", "coordination_number"}``.
        Parameters not listed here are kept fixed at the values passed to
        :class:`MFCFLoGoCVAll`.

    n_trials : int or None, default=100
        Number of Optuna trials to run. If ``None`` or ``0``, no tuning is
        performed and the estimator is refit once with the fixed parameters.

    n_jobs : int or None, default=None
        Number of parallel jobs for cross-validation scoring. Passed to
        :func:`sklearn.model_selection.cross_val_score`. ``None`` means 1;
        ``-1`` uses all processors.

    # Cross-validation
    cv : int, default=5
        Number of KFold splits.

    shuffle : bool, default=False
        Whether to shuffle samples before splitting.

    random_state : Optional[int], default=None
        Random state for the KFold when shuffle=True and seed of the Optuna
        sampler (reproducible search).

    # Scoring / stability
    assume_centered : bool, default=False
        Passed to MFCFLoGo and empirical covariance computation.

    error_score : {'raise', float}, default=np.nan
        Score to assign if a fit/eval fails for a fold/setting. If 'raise',
        the exception is raised. If a float, the error is caught and this score
        is used for that fold.

    # Parameters forwarded to MFCFLoGo
    threshold : float, default=0.0
        Gain threshold controlling attachment vs. new component.

    min_clique_size : int, default=1
        The minimum size of cliques to form in the graph.

    coordination_number : float, default=np.inf
        Maximum allowed uses of a separator (multiplicity cap).

    gain_function_type : {'loglik', 'sumsquares', 'sum'}, default=None
        None means 'loglik' (with a FutureWarning in this release: the default
        was 'sumsquares' until version 2026.10).
        Gain function type to use in MFCF construction. The function is used to
        determine which vertext to add next to maximize the overall gain in the
        graph.

    acceptance : {'threshold', 'ftest', 'bic'}, default='threshold'
        Rule for accepting each attachment step. 'ftest' (partial F-test at
        level `alpha`) and 'bic' are statistical validations for the 'loglik'
        gain; the number of observations is taken from the data.

    alpha : float, default=0.05
        Level of the partial F-test (acceptance='ftest').

    new_component : {'strongest', 'first'}, default='strongest'
        Vertex that starts a new component (see `MFCF`).

    n_starts : int or None, default=None
        Number of starting cliques, at most `max_clique_size`; the clique
        forest with the largest objective is kept. None: `max_clique_size`
        for the 'loglik' gain, 1 otherwise (see `MFCF`).

    assume_centered : bool, default=False
        If True, data are not centered before computation.
        Useful when working with data whose mean is almost, but not exactly
        zero.
        If False, data are centered before computation.


    Attributes
    ----------
    covariance_ : ndarray of shape (n_features, n_features)
        Inverse of the LoGo precision estimate.

    precision_ : ndarray of shape (n_features, n_features)
        LoGo precision estimated by refitting on the full dataset with the best
        `max_clique_size`.

    location_ : ndarray of shape (n_features,)
        Estimated mean (0 if assume_centered=True).

    cliques_ : list of frozenset[int]
        Maximal cliques found by MFCF.

    separators_count_ : Counter[frozenset[int], int]
        Separator multiplicities used in the constructed MFCF.

    peo_ : list[int]
        Perfect elimination order of vertices produced by the algorithm.

    n_features_in_ : int
        Number of features seen during fit.

    estimator_ : MFCFLoGo
        Final refit estimator on the full data.

    Notes
    -----
    * Optuna is used to optimize the mean CV log-likelihood. The sampler
        draws:
            - ``threshold`` via a continuous uniform search on ``[0, 1]`` when
              tuned.
            - ``max_clique_size`` and/or ``min_clique_size`` via integer
              log-scaled suggestions with consistency constraints
              (``min_clique_size <= max_clique_size``).
            - ``coordination_number`` via integer log-scaled suggestions.

    * ``max_clique_size`` is naturally bounded:
        ``min_clique_size <= max_clique_size <= n_features``. The search
        space is filtered/deduplicated accordingly on each trial.

    * The validation score equals the Gaussian log-likelihood up to an
        additive constant that does not depend on ``Theta``; maximizing the
        stated objective is therefore equivalent to maximizing the likelihood.

    """

    def __init__(
        self,
        *,
        tunable_params: Iterable[str] = (),
        cv: int = 5,
        shuffle: bool = False,
        random_state: Optional[int] = None,
        n_jobs: Optional[int] = None,
        n_trials: Optional[int] = 100,
        # passthrough to base estimator
        threshold: float = 0.0,
        min_clique_size: int = 1,
        max_clique_size: int = 4,
        coordination_number: int = np.inf,
        gain_function_type: Optional[str] = None,
        acceptance: str = "threshold",
        alpha: float = 0.05,
        new_component: str = "strongest",
        n_starts: Optional[int] = None,
        shrinkage: float = 0.0,
        n_effective=None,
        correlation: str = "pearson",
        similarity: str = "correlation",
        mi_estimator: str = "ksg",
        mi_n_neighbors: int = 3,
        mi_n_bins="auto",
        mi_normalize: str = "linfoot",
        mi_n_jobs: Optional[int] = None,
        mi_random_state: Optional[int] = None,
        assume_centered: bool = False,
        # scoring / robustness
        error_score: Union[str, float] = np.nan,
    ):
        super().__init__(assume_centered=assume_centered)
        self.tunable_params = tunable_params
        self.cv = cv
        self.shuffle = shuffle
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.n_trials = n_trials

        self.threshold = threshold
        self.max_clique_size = max_clique_size
        self.min_clique_size = min_clique_size
        self.coordination_number = coordination_number
        self.gain_function_type = gain_function_type
        self.acceptance = acceptance
        self.alpha = alpha
        self.new_component = new_component
        self.n_starts = n_starts
        self.shrinkage = shrinkage
        self.n_effective = n_effective
        self.correlation = correlation
        self.similarity = similarity
        self.mi_estimator = mi_estimator
        self.mi_n_neighbors = mi_n_neighbors
        self.mi_n_bins = mi_n_bins
        self.mi_normalize = mi_normalize
        self.mi_n_jobs = mi_n_jobs
        self.mi_random_state = mi_random_state
        self.assume_centered = assume_centered

        self.error_score = error_score

    def _process_params(self, trial, n_features: int) -> Dict[str, Any]:
        params = dict(
            threshold=self.threshold,
            min_clique_size=self.min_clique_size,
            max_clique_size=self.max_clique_size,
            coordination_number=self.coordination_number,
            gain_function_type=self.gain_function_type,
            acceptance=self.acceptance,
            alpha=self.alpha,
            new_component=self.new_component,
            n_starts=self.n_starts,
            shrinkage=self.shrinkage,
            n_effective=self.n_effective,
            correlation=self.correlation,
            similarity=self.similarity,
            mi_estimator=self.mi_estimator,
            mi_n_neighbors=self.mi_n_neighbors,
            mi_n_bins=self.mi_n_bins,
            mi_normalize=self.mi_normalize,
            mi_n_jobs=self.mi_n_jobs,
            mi_random_state=self.mi_random_state,
        )
        if not self.tunable_params:
            return params
        unknown = set(self.tunable_params) - {"alpha", "threshold", "max_clique_size",
                                               "min_clique_size", "coordination_number"}
        if unknown:
            raise ValueError(f"Unknown tunable_params: {sorted(unknown)}; choose from 'alpha', "
                             "'threshold', 'max_clique_size', 'min_clique_size', 'coordination_number'")

        if "alpha" in self.tunable_params:
            params["alpha"] = trial.suggest_float("alpha", 1e-4, 0.5, log=True)

        if "threshold" in self.tunable_params:
            params["threshold"] = trial.suggest_float("threshold", 0.0, 1.0, step=None)

        if (
            "max_clique_size" in self.tunable_params
            and "min_clique_size" in self.tunable_params
        ):
            params["max_clique_size"] = trial.suggest_int(
                "max_clique_size", 2, max(2, min(32, n_features)), log=True
            )
            params["min_clique_size"] = trial.suggest_int(
                "min_clique_size", 1, params["max_clique_size"], log=True
            )
        elif "min_clique_size" in self.tunable_params:
            # Only tune min_clique_size, fixed max_clique_size
            params["min_clique_size"] = trial.suggest_int(
                "min_clique_size", 1, params["max_clique_size"], log=True
            )
        elif "max_clique_size" in self.tunable_params:
            # Only tune max_clique_size, fixed min_clique_size
            lo = max(2, int(self.min_clique_size))
            params["max_clique_size"] = trial.suggest_int(
                "max_clique_size", lo, max(lo, min(32, n_features)), log=True
            )

        if "coordination_number" in self.tunable_params:
            params["coordination_number"] = trial.suggest_int(
                "coordination_number", 1, n_features, log=True
            )  # Needs to confirm the max coordination_number

        return params

    def _cross_validate(self, X: np.ndarray):
        try:
            import optuna
        except ImportError as e:
            raise ImportError("MFCFLoGoCVAll requires optuna: pip install optuna") from e

        def objective(trial):
            n_features = X.shape[1]
            params = self._process_params(trial, n_features)
            # Log sampled parameters before evaluating the trial
            # print(
            #     f"[Optuna] Trial {trial.number} starting with params: "
            #     + ", ".join(f"{k}={v}" for k, v in params.items()),
            #     flush=True,
            # )
            if params["n_starts"] is not None:
                params["n_starts"] = min(params["n_starts"], params["max_clique_size"])
            n = X.shape[0]
            params["n_effective"] = _fold_n_effective(params["n_effective"], n - -(-n // self.cv), n)
            est = MFCFLoGo(**params, assume_centered=self.assume_centered)

            cv = KFold(n_splits=self.cv, shuffle=self.shuffle,
                       random_state=self.random_state if self.shuffle else None)
            try:
                scores = cross_val_score(
                    est, X, y=None, scoring=log_likelihood_scorer, cv=cv, n_jobs=self.n_jobs,
                    error_score="raise" if self.error_score == "raise" else np.nan,
                )
            except Exception:
                if self.error_score == "raise":
                    raise
                return float("nan")          # failed trial: never selected
            # a configuration with any failed fold is not eligible
            return float(scores.mean()) if np.all(np.isfinite(scores)) else float("nan")

        self._trials = []
        if not self.n_trials or not self.tunable_params:      # no tuning: fixed parameters
            best_params = {}
        else:
            sampler = optuna.samplers.TPESampler(seed=self.random_state)  # reproducible search
            study = optuna.create_study(direction="maximize", sampler=sampler)
            study.optimize(objective, n_trials=int(self.n_trials))
            complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
            if not complete:
                raise ValueError("Every trial failed in cross-validation (e.g. clique sizes larger "
                                 "than the training folds support); use shrinkage > 0, fewer "
                                 "tunable clique sizes or more observations")
            best_params = study.best_params
            self._trials = [dict(params=dict(t.params),
                                 mean_test_score=(float(t.value) if t.value is not None else np.nan),
                                 state=t.state.name) for t in study.trials]
        final_params = dict(
            threshold=self.threshold,
            min_clique_size=self.min_clique_size,
            max_clique_size=self.max_clique_size,
            coordination_number=self.coordination_number,
            gain_function_type=self.gain_function_type,
            acceptance=self.acceptance,
            alpha=self.alpha,
            new_component=self.new_component,
            n_starts=self.n_starts,
            shrinkage=self.shrinkage,
            n_effective=self.n_effective,
            correlation=self.correlation,
            similarity=self.similarity,
            mi_estimator=self.mi_estimator,
            mi_n_neighbors=self.mi_n_neighbors,
            mi_n_bins=self.mi_n_bins,
            mi_normalize=self.mi_normalize,
            mi_n_jobs=self.mi_n_jobs,
            mi_random_state=self.mi_random_state,
        )
        final_params.update(best_params)
        if final_params["n_starts"] is not None:
            final_params["n_starts"] = min(final_params["n_starts"], final_params["max_clique_size"])
        # print(best_params, flush=True)

        return final_params

    # ------------------------------ API ------------------------------

    def fit(self, X: np.ndarray, y=None):
        """Fit the estimator from data X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data from which to compute the covariance estimate.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : object
            Returns the instance itself.
        """

        X = check_array(
            X,
            ensure_min_samples=max(self.cv, 2),
            ensure_min_features=1,
            dtype=[np.float64, np.float32],
        )
        n_samples, n_features = X.shape
        self.n_features_in_ = n_features
        self.location_ = (
            np.zeros(n_features, dtype=X.dtype)
            if self.assume_centered
            else X.mean(axis=0)
        )

        _check_configuration(self, X.shape[0], self.max_clique_size)
        best_params = self._cross_validate(X)
        self.cv_results_ = list(self._trials)          # one record per Optuna trial
        self.best_params_ = {k: v for k, v in best_params.items()
                             if k in (self.tunable_params or ()) and self.n_trials}

        # Refit on all data with best parameter
        final = MFCFLoGo(**best_params, assume_centered=self.assume_centered)
        final.fit(X)

        self.covariance_ = final.covariance_
        self.precision_ = final.precision_
        self.cliques_ = final.cliques_
        self.separators_count_ = final.separators_count_
        self.peo_ = final.peo_
        self.estimator_ = final
        return self

    def get_precision(self) -> np.ndarray:
        check_is_fitted(self, attributes=("precision_",))
        return self.precision_

    def get_covariance(self) -> np.ndarray:
        check_is_fitted(self, attributes=("covariance_",))
        return self.covariance_
