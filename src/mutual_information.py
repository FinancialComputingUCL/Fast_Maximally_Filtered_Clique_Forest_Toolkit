"""
Mutual-information similarity matrix for MFCF.

Pearson correlation is the natural similarity when the variables are jointly
Gaussian.  Outside that regime — fat tails, regime shifts, non-linear
co-movement — it can grossly miss the dependence structure of the data.
Mutual information ``I(X;Y)`` is the right replacement: it is zero iff ``X``
and ``Y`` are independent and is invariant under any monotone reparametrisation
of the marginals.

This module ships the Kraskov–Stögbauer–Grassberger k-nearest-neighbour
estimator (Kraskov, Stögbauer, Grassberger, "Estimating mutual information",
*Phys. Rev. E* 69, 066138, 2004) — non-parametric, asymptotically unbiased,
adaptive to local data density, and capable of capturing arbitrary non-linear
dependence including the tails.  Cost is ``O(p^2 * n^{3/2})`` (numba kernel
below); inherently more expensive than correlation because every column pair
needs its own neighbour search.

Backends
--------
The KSG estimator has two backends:

* ``"numba"`` — hand-rolled parallel kernel.  Per pair, instead of the naive
  ``O(n^2)`` two-pass scan it (a) sorts each column once up front, then for
  each query expands two pointers outward in x-sorted order, computes joint
  Chebyshev distances on the fly, and stops as soon as the next x-only
  distance exceeds the current k-th joint distance; and (b) replaces the
  marginal-count pass with two ``O(log n)`` binary searches on the sorted
  columns.  Typical scan length is ``~3 sqrt(n)`` for k=3 rather than ``n``.
* ``"sklearn"`` — reference path via
  ``sklearn.feature_selection.mutual_info_regression``; both produce KSG
  algorithm-1 estimates and agree to numerical tolerance.

Normalisation
-------------
Raw ``I(X;Y) \\in [0, \\infty)`` is not directly comparable across pairs and is
not on the same scale as the squared-correlation gain used by MFCF.  We
therefore default to Linfoot's *informational coefficient of correlation*
(Linfoot, "An informational measure of correlation", *Inf. Control* 1(1),
1957)

    rho_I = sqrt( 1 - exp(-2 * I) )    in [0, 1]

This transform collapses to ``|Pearson rho|`` exactly when (X, Y) is jointly
Gaussian, so the MI path is a strict generalisation of the correlation path
rather than a different objective.
"""

from __future__ import annotations

import inspect
from typing import Literal, Optional

import numpy as np
from sklearn.feature_selection import mutual_info_regression


_NormalizeKind = Literal["linfoot", "none"]
_BackendKind = Literal["auto", "numba", "sklearn"]


# Euler–Mascheroni constant, used to seed the integer digamma recurrence
# digamma(1) = -gamma_e, digamma(m+1) = digamma(m) + 1/m.
_GAMMA_E = 0.5772156649015329


# sklearn 1.5 added ``n_jobs`` to mutual_info_regression; tolerate older releases.
_MI_SUPPORTS_NJOBS = (
    "n_jobs" in inspect.signature(mutual_info_regression).parameters
)


try:
    import numba
    from numba import njit, prange  # type: ignore
    _HAS_NUMBA = True
except Exception:  # pragma: no cover - environments without numba
    _HAS_NUMBA = False
    numba = None  # type: ignore

    def njit(*args, **kwargs):  # type: ignore[misc]
        def _dec(fn):
            return fn
        if args and callable(args[0]):
            return args[0]
        return _dec

    def prange(n):  # type: ignore[misc]
        return range(n)


def _digamma_int_table(n: int) -> np.ndarray:
    """``psi[m] = digamma(m + 1)`` for ``m in [0, n)``.

    KSG algorithm 1 only ever queries digamma at positive integers
    (``k``, ``N``, ``n_x + 1``, ``n_y + 1``), so the closed-form integer
    recurrence is exact — no special-function call needed.
    """
    psi = np.empty(n, dtype=np.float64)
    val = -_GAMMA_E  # digamma(1)
    psi[0] = val
    for m in range(1, n):
        val += 1.0 / m  # digamma(m+1) = digamma(m) + 1/m
        psi[m] = val
    return psi


# -----------------------------------------------------------------------------
# Numba KSG backend
# -----------------------------------------------------------------------------
@njit(cache=True, inline="always", boundscheck=False)
def _bisect_right(a: np.ndarray, x: float, n: int) -> int:
    """Smallest ``i`` with ``a[i] > x`` (sorted ``a``, half-open ``[0, n)``)."""
    lo = 0
    hi = n
    while lo < hi:
        mid = (lo + hi) >> 1
        if a[mid] <= x:
            lo = mid + 1
        else:
            hi = mid
    return lo


@njit(cache=True, inline="always", boundscheck=False)
def _bisect_left(a: np.ndarray, x: float, n: int) -> int:
    """Smallest ``i`` with ``a[i] >= x`` (sorted ``a``, half-open ``[0, n)``)."""
    lo = 0
    hi = n
    while lo < hi:
        mid = (lo + hi) >> 1
        if a[mid] < x:
            lo = mid + 1
        else:
            hi = mid
    return lo


@njit(parallel=True, cache=True, boundscheck=False, fastmath=True)
def _ksg_matrix_numba(
    XT: np.ndarray,
    sorted_cols: np.ndarray,
    argsort_cols: np.ndarray,
    ranks_cols: np.ndarray,
    pair_i: np.ndarray,
    pair_j: np.ndarray,
    k: int,
    psi: np.ndarray,
    M: np.ndarray,
) -> None:
    """Fill the upper+lower triangle of ``M`` with KSG MI estimates.

    The kernel walks each query's neighbours in sorted-x order via a
    two-pointer scan, tracks value-space joint Chebyshev distances in a
    length-``k`` insertion-sort buffer, and resolves marginal counts via
    bisect on the sorted columns.  Parallelism is over a flat enumeration
    of upper-triangle pairs so every thread gets a balanced workload — the
    natural nested-loop ordering put ``O(p)`` work on the last thread and
    ``O(1)`` on the first.

    A rank-space variant (replace value Chebyshev with integer rank
    Chebyshev throughout, and use closed-form marginal counts) was tried
    and is materially faster, but its deterministic marginal counts
    inflate the small-MI estimate enough to push rho≈0 outside the
    Linfoot collapse tolerance — i.e., it breaks the headline correctness
    property of this module.

    Parameters
    ----------
    XT : np.ndarray, shape (p, n), C-contiguous, float64
        Transposed data; row ``i`` is column ``i`` of the original ``X``.
    sorted_cols : np.ndarray, shape (p, n), C-contiguous, float64
        ``sorted_cols[i]`` is column ``i`` sorted ascending.
    argsort_cols : np.ndarray, shape (p, n), C-contiguous, int64
        ``argsort_cols[i, r]`` is the original index whose value sits at
        sorted rank ``r`` in column ``i``.
    ranks_cols : np.ndarray, shape (p, n), C-contiguous, int64
        ``ranks_cols[i, ii]`` is the sorted rank of point ``ii`` in column
        ``i`` (inverse permutation of ``argsort_cols``).
    pair_i, pair_j : np.ndarray, shape (P,), int64
        Flat enumeration of every upper-triangle column pair with ``i < j``.
    k : int
        Number of neighbours (algorithm 1).
    psi : np.ndarray, shape (n,), float64
        Precomputed digamma table with ``psi[m] = digamma(m + 1)``.
    M : np.ndarray, shape (p, p), float64
        Output buffer.  Diagonal is left untouched.
    """
    n = XT.shape[1]
    P = pair_i.shape[0]
    digamma_k = psi[k - 1]
    digamma_n = psi[n - 1]

    for t in prange(P):
        i = pair_i[t]
        j = pair_j[t]

        # Length-k scratch for the top-k joint distances.  Stack-resident
        # (numba lowers small np.empty inside prange to per-iter locals).
        top_k = np.empty(k, dtype=np.float64)

        sum_psi = 0.0

        for ii in range(n):
            xi = XT[i, ii]
            yi = XT[j, ii]
            pos_x = ranks_cols[i, ii]

            for kk in range(k):
                top_k[kk] = np.inf

            left = pos_x - 1
            right = pos_x + 1
            eps_curr = np.inf  # = top_k[k - 1]
            # Cache the two halves of sorted_cols[i] for this query into
            # local pointer-like vars to give numba an unambiguous strided
            # access pattern.
            sorted_i = sorted_cols[i]
            argsort_i = argsort_cols[i]
            XT_j = XT[j]

            while True:
                # Branchless side picking: read both sides, treat
                # out-of-range as +inf so the min wins.
                if left < 0:
                    dl = np.inf
                else:
                    dl = xi - sorted_i[left]
                if right >= n:
                    dr = np.inf
                else:
                    dr = sorted_i[right] - xi
                if dl <= dr:
                    dx_next = dl
                    next_sorted_idx = left
                    left -= 1
                else:
                    dx_next = dr
                    next_sorted_idx = right
                    right += 1

                # No further point can shrink the k-th joint distance, and
                # if both sides are out of range dx_next == inf > eps_curr
                # naturally terminates the loop.
                if dx_next >= eps_curr:
                    break

                jj = argsort_i[next_sorted_idx]
                dy = yi - XT_j[jj]
                if dy < 0.0:
                    dy = -dy
                d = dx_next if dx_next > dy else dy

                if d < eps_curr:
                    # Insertion-sort into length-k scratch.
                    pos = k - 1
                    while pos > 0 and top_k[pos - 1] > d:
                        top_k[pos] = top_k[pos - 1]
                        pos -= 1
                    top_k[pos] = d
                    eps_curr = top_k[k - 1]

            eps = top_k[k - 1]

            # Marginal counts via bisect on the sorted columns.  Open
            # interval (xi - eps, xi + eps), excluding self when eps > 0.
            lo_x = _bisect_right(sorted_i, xi - eps, n)
            hi_x = _bisect_left(sorted_i, xi + eps, n)
            nx = hi_x - lo_x

            sorted_j = sorted_cols[j]
            lo_y = _bisect_right(sorted_j, yi - eps, n)
            hi_y = _bisect_left(sorted_j, yi + eps, n)
            ny = hi_y - lo_y

            if eps > 0.0:
                nx -= 1
                ny -= 1

            sum_psi += psi[nx] + psi[ny]

        mi = digamma_k + digamma_n - sum_psi / n
        if mi < 0.0:
            mi = 0.0
        M[i, j] = mi
        M[j, i] = mi


# -----------------------------------------------------------------------------
# sklearn backend (reference)
# -----------------------------------------------------------------------------
def _ksg_matrix_sklearn(
    X: np.ndarray,
    *,
    n_neighbors: int,
    n_jobs: Optional[int],
    random_state: Optional[int],
) -> np.ndarray:
    """Reference KSG MI matrix via sklearn (one ``mutual_info_regression`` per
    target column).  Kept for cross-validation and for environments without
    numba."""
    n_samples, n_features = X.shape
    M = np.zeros((n_features, n_features), dtype=np.float64)
    if n_features < 2:
        return M

    mi_kwargs = dict(
        discrete_features=False,
        n_neighbors=n_neighbors,
        copy=False,
        random_state=random_state,
    )
    if _MI_SUPPORTS_NJOBS:
        mi_kwargs["n_jobs"] = n_jobs

    for j in range(1, n_features):
        mi_row = mutual_info_regression(X[:, :j], X[:, j], **mi_kwargs)
        M[:j, j] = mi_row

    np.maximum(M, 0.0, out=M)
    M += M.T
    return M


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------
def mutual_information_matrix(
    X: np.ndarray,
    *,
    n_neighbors: int = 3,
    normalize: _NormalizeKind = "linfoot",
    n_jobs: Optional[int] = None,
    random_state: Optional[int] = None,
    backend: _BackendKind = "auto",
) -> np.ndarray:
    """Pairwise mutual-information similarity matrix (KSG estimator).

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        Sample-by-feature design matrix.
    n_neighbors : int, default=3
        Number of neighbours for the KSG estimator.  Values in ``[3, 6]`` are
        the usual recommendation; results are robust to the exact choice.
    normalize : {"linfoot", "none"}, default="linfoot"
        ``"linfoot"`` applies ``sqrt(1 - exp(-2 I))`` so entries live in
        ``[0, 1]`` and match ``|Pearson rho|`` under Gaussianity.  ``"none"``
        keeps the raw estimate (clipped to ``>= 0``).
    n_jobs : int or None, default=None
        Thread count for the KSG ``numba`` backend (or sklearn's
        ``mutual_info_regression`` when that backend is used).  ``None``
        keeps the backend default.
    random_state : int or None, default=None
        Seed for the tiny Gaussian jitter that breaks ties (standard
        Kraskov-2004 preprocessing).
    backend : {"auto", "numba", "sklearn"}, default="auto"
        KSG backend selector.  ``"auto"`` picks numba when it is importable,
        else sklearn.

    Returns
    -------
    M : np.ndarray, shape (n_features, n_features)
        Symmetric non-negative similarity matrix.  Diagonal is ``1.0`` for
        ``normalize="linfoot"`` (matching the correlation convention) and
        ``0.0`` for ``normalize="none"``.

    Notes
    -----
    KSG is asymptotically the most informative choice but inherently costs
    one neighbour search per column pair, so its wall time grows like ``p^2``
    rather than the ``p^2 n`` of a single matrix multiply.
    """
    if normalize not in ("linfoot", "none"):
        raise ValueError(
            f"normalize must be 'linfoot' or 'none', got {normalize!r}."
        )
    if backend not in ("auto", "numba", "sklearn"):
        raise ValueError(
            f"backend must be one of 'auto', 'numba', 'sklearn', got {backend!r}."
        )

    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D, got shape {X.shape}.")

    n_samples, n_features = X.shape
    if n_samples < n_neighbors + 1:
        raise ValueError(
            f"Need at least n_neighbors + 1 = {n_neighbors + 1} samples for "
            f"the KSG estimator, got {n_samples}."
        )
    if n_samples < 2:
        raise ValueError(
            f"Need at least 2 samples, got {n_samples}."
        )

    diag_value = 1.0 if normalize == "linfoot" else 0.0
    if n_features < 2:
        M_small = np.zeros((n_features, n_features), dtype=np.float64)
        np.fill_diagonal(M_small, diag_value)
        return M_small

    # Resolve the backend.
    if backend == "numba" and not _HAS_NUMBA:
        raise RuntimeError(
            "backend='numba' was requested but numba is not importable."
        )
    use_numba = backend == "numba" or (backend == "auto" and _HAS_NUMBA)

    # Kraskov-style tie-breaking jitter: scaled by the per-column mean
    # absolute magnitude so it never dominates the data signal.
    rng = np.random.RandomState(random_state)
    col_scale = np.maximum(1.0, np.mean(np.abs(X), axis=0))
    X_work = X + 1e-10 * col_scale * rng.standard_normal(X.shape)

    if use_numba:
        # Row-major transpose so each column (now a row) is contiguous.
        XT = np.ascontiguousarray(X_work.T)
        # Per-column sort: argsort gives the rank-to-index map; the inverse
        # permutation is the rank of every original index.  This is the
        # entire precomputation needed by the two-pointer kernel.
        argsort_cols = np.argsort(XT, axis=1, kind="quicksort").astype(np.int64)
        sorted_cols = np.take_along_axis(XT, argsort_cols, axis=1)
        ranks_cols = np.empty_like(argsort_cols)
        _idx = np.arange(n_samples, dtype=np.int64)
        for _p in range(n_features):
            ranks_cols[_p, argsort_cols[_p]] = _idx
        psi = _digamma_int_table(n_samples)
        M = np.zeros((n_features, n_features), dtype=np.float64)

        # Flat enumeration of upper-triangle pairs.  Driving the kernel with
        # ``prange`` over this flat index gives the thread pool perfectly
        # balanced chunks; the natural nested-loop ordering put O(p) work on
        # the last thread and O(1) on the first, leaving most threads idle for
        # the tail of the run.
        pair_i_arr, pair_j_arr = np.triu_indices(n_features, k=1)
        pair_i_arr = pair_i_arr.astype(np.int64, copy=False)
        pair_j_arr = pair_j_arr.astype(np.int64, copy=False)

        if n_jobs is not None and n_jobs > 0:
            _orig = numba.get_num_threads()
            numba.set_num_threads(int(n_jobs))
            try:
                _ksg_matrix_numba(
                    XT, sorted_cols, argsort_cols, ranks_cols,
                    pair_i_arr, pair_j_arr,
                    int(n_neighbors), psi, M,
                )
            finally:
                numba.set_num_threads(_orig)
        else:
            _ksg_matrix_numba(
                XT, sorted_cols, argsort_cols, ranks_cols,
                pair_i_arr, pair_j_arr,
                int(n_neighbors), psi, M,
            )

        # Clip to guard against tiny negative values from the digamma table
        # plus floating-point noise (the kernel already clips per pair).
        np.maximum(M, 0.0, out=M)
    else:
        M = _ksg_matrix_sklearn(
            X_work,
            n_neighbors=int(n_neighbors),
            n_jobs=n_jobs,
            random_state=random_state,
        )

    if normalize == "linfoot":
        # In-place: sqrt(1 - exp(-2 I)).
        np.exp(-2.0 * M, out=M)
        np.subtract(1.0, M, out=M)
        np.maximum(M, 0.0, out=M)  # numerical floor at 0
        np.sqrt(M, out=M)

    np.fill_diagonal(M, diag_value)
    return M
