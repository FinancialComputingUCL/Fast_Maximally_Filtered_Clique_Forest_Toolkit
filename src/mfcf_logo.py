from typing import Optional

import numpy as np
import numpy.linalg as LA
from sklearn.covariance import EmpiricalCovariance
from sklearn.covariance import empirical_covariance
from sklearn.utils.validation import check_array, check_is_fitted

from fast_fast_mfcf import MFCF as _MFCFBuilder
from mutual_information import mutual_information_matrix as _mi_matrix


_VALID_SIMILARITIES = ("correlation", "mutual_information")


def _build_similarity(
    X: np.ndarray,
    similarity: str,
    mi_estimator: str,
    mi_n_neighbors: int,
    mi_n_bins,
    mi_normalize: str,
    mi_n_jobs: Optional[int],
    mi_random_state: Optional[int],
) -> np.ndarray:
    """Return the (p, p) similarity matrix used as MFCF's gain input."""
    if similarity == "correlation":
        return np.corrcoef(X, rowvar=False)
    if similarity == "mutual_information":
        return _mi_matrix(
            X,
            estimator=mi_estimator,
            n_neighbors=mi_n_neighbors,
            n_bins=mi_n_bins,
            normalize=mi_normalize,
            n_jobs=mi_n_jobs,
            random_state=mi_random_state,
        )
    raise ValueError(
        f"similarity must be one of {_VALID_SIMILARITIES}, got {similarity!r}."
    )


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

    gain_function_type : {'sumsquares'}, default='sumsquares'
        Gain function type to use in MFCF construction. The function is used to
        determine which vertex to add next to maximize the overall gain in the
        graph.

    similarity : {'correlation', 'mutual_information'}, default='correlation'
        Similarity used to score MFCF gains.  ``'correlation'`` uses Pearson
        correlation (assumes joint Gaussianity).  ``'mutual_information'``
        builds the pairwise MI matrix with the estimator selected by
        ``mi_estimator``.

    mi_estimator : {'ksg', 'histogram'}, default='ksg'
        MI estimator used when ``similarity='mutual_information'``.  ``'ksg'``
        is the Kraskov-Stögbauer-Grassberger k-NN estimator (most informative,
        one neighbour search per pair).  ``'histogram'`` is the equal-frequency
        plug-in estimator with a Miller-Madow correction — much faster
        (correlation-like ``O(p^2 n)`` scaling) and the preferred choice when
        ``p >> n`` or when MI-matrix build time matters.  Both collapse to
        ``|Pearson rho|`` under joint Gaussianity.

    mi_n_neighbors : int, default=3
        Number of neighbours for the KSG estimator.  Only used when
        ``similarity='mutual_information'`` and ``mi_estimator='ksg'``.

    mi_n_bins : int or 'auto', default='auto'
        Number of equal-frequency bins for the histogram estimator.  ``'auto'``
        scales the bin count with the sample size (the binding constraint when
        ``p >> n``).  Only used when ``similarity='mutual_information'`` and
        ``mi_estimator='histogram'``.

    mi_normalize : {'linfoot', 'none'}, default='linfoot'
        Normalisation applied to the raw MI estimates.  Only used when
        ``similarity='mutual_information'``.

    mi_n_jobs : int or None, default=None
        Parallelism forwarded to the KSG estimator (sklearn >= 1.5).  Only
        used when ``similarity='mutual_information'``.

    mi_random_state : int or None, default=None
        Random state for the KSG estimator's tie-breaking jitter.  Only used
        when ``similarity='mutual_information'``.

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
        gain_function_type: str = "sumsquares",
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
        self.similarity = similarity
        self.mi_estimator = mi_estimator
        self.mi_n_neighbors = mi_n_neighbors
        self.mi_n_bins = mi_n_bins
        self.mi_normalize = mi_normalize
        self.mi_n_jobs = mi_n_jobs
        self.mi_random_state = mi_random_state

    def fit(self, X: np.ndarray, y=None,
            corr_matrix: Optional[np.ndarray] = None,
            mi_matrix: Optional[np.ndarray] = None,
            cov_matrix: Optional[np.ndarray] = None) -> "MFCFLoGo":
        """Fit the estimator from data X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data from which to compute the covariance estimate.

        y : Ignored
            Not used, present for API consistency by convention.

        corr_matrix : np.ndarray, optional
            Precomputed feature correlation matrix to use in the MFCF
            gain function.  Honoured only when ``similarity='correlation'``.
            When supplied the expensive ``np.corrcoef`` call is skipped.
            The matrix is *not* re-validated; pass shape ``(p, p)`` aligned
            with ``X``.

        mi_matrix : np.ndarray, optional
            Precomputed mutual-information similarity matrix to use in the
            MFCF gain function.  Honoured only when
            ``similarity='mutual_information'``.  When supplied the expensive
            KSG computation is skipped.  Pass shape ``(p, p)`` aligned with
            ``X``.

        cov_matrix : np.ndarray, optional
            Precomputed feature covariance/correlation to use in the
            LoGo aggregation step (the per-clique inversions).  If ``None``
            the empirical covariance is used when ``similarity='mutual_information'``
            (because the MI matrix is not a covariance and cannot be inverted),
            and the correlation matrix is used when
            ``similarity='correlation'`` (preserving the prior behaviour).

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

        if self.similarity not in _VALID_SIMILARITIES:
            raise ValueError(
                f"similarity must be one of {_VALID_SIMILARITIES}, "
                f"got {self.similarity!r}."
            )

        if self.similarity == "correlation":
            C = (
                corr_matrix
                if corr_matrix is not None
                else np.corrcoef(X, rowvar=False)
            )
        else:
            C = (
                mi_matrix
                if mi_matrix is not None
                else _build_similarity(
                    X,
                    self.similarity,
                    self.mi_estimator,
                    self.mi_n_neighbors,
                    self.mi_n_bins,
                    self.mi_normalize,
                    self.mi_n_jobs,
                    self.mi_random_state,
                )
            )
            # LoGo inverts clique sub-matrices, so it needs an actual covariance,
            # not the MI similarity.  Fall back to the empirical covariance when
            # the caller did not provide one.
            if cov_matrix is None:
                cov_matrix = empirical_covariance(
                    X, assume_centered=self.assume_centered
                )

        # run MFCF algorithm over a similarity/affinity matrix.
        builder = _MFCFBuilder(
            threshold=self.threshold,
            min_clique_size=self.min_clique_size,
            max_clique_size=self.max_clique_size,
            coordination_number=self.coordination_number,
            gain_function_type=self.gain_function_type,
        )

        cliques, separators_count, peo, logo = builder.run(
            C=C, cov_matrix=cov_matrix
        )

        # store internals
        self.cliques_ = cliques
        self.separators_count_ = separators_count
        self.peo_ = peo

        # precision is LoGo; covariance is inverse of LoGo
        # Add small ridge if needed to avoid singularities.
        self.precision_ = logo
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

