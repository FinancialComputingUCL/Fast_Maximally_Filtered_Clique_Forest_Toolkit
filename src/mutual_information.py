"""
Mutual-information similarity matrix for MFCF.

Pearson correlation is the natural similarity when the variables are jointly
Gaussian.  Outside that regime — fat tails, regime shifts, non-linear
co-movement — it can grossly miss the dependence structure of the data.
Mutual information ``I(X;Y)`` is the right replacement: it is zero iff ``X``
and ``Y`` are independent and is invariant under any monotone reparametrisation
of the marginals.

This module ships two non-parametric estimators of ``I(X;Y)``, selected with
the ``estimator`` argument of :func:`mutual_information_matrix`:

``estimator="ksg"`` (default)
    The Kraskov–Stögbauer–Grassberger k-nearest-neighbour estimator (Kraskov,
    Stögbauer, Grassberger, "Estimating mutual information", *Phys. Rev. E* 69,
    066138, 2004) — adaptive to local data density and capable of capturing
    arbitrary non-linear dependence including the tails.  Cost is
    ``O(p^2 * n^{3/2})`` (numba kernel below); inherently more expensive than
    correlation because every column pair needs its own neighbour search.

``estimator="histogram"``
    The plug-in / histogram estimator on equal-frequency (rank) bins, with a
    Miller–Madow bias correction.  This is the *fast* path: its per-pair work
    is a single ``O(n)`` streaming pass that fills a tiny ``B x B`` joint count
    table (``B`` = bins) plus an ``O(B^2)`` entropy reduction, so the whole
    matrix costs ``O(p^2 (n + B^2))`` — the same ``p^2 n`` scaling as a single
    correlation matrix-multiply, only with a larger constant.  See
    :func:`histogram_mutual_information_matrix` for the design notes; in the
    ``p >> n`` regime MFCF is most often used in, the bin count is driven by
    ``n`` (the binding constraint) so the cells stay well populated.

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

The histogram estimator has a single hand-rolled numba kernel (with a pure
numpy fallback when numba is unavailable).

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
rather than a different objective.  Both estimators target the same population
``I(X;Y)``, so both Linfoot-normalised matrices converge to ``|Pearson rho|``
under joint Gaussianity — i.e. correlation, KSG and histogram collapse to the
same similarity matrix.  ``src/validate_mi_histogram.py`` demonstrates this and
benchmarks the histogram path against correlation in the ``p >> n`` regime.
"""

from __future__ import annotations

import inspect
from typing import Literal, Optional

import numpy as np
from sklearn.feature_selection import mutual_info_regression


_NormalizeKind = Literal["linfoot", "none"]
_BackendKind = Literal["auto", "numba", "sklearn"]
_EstimatorKind = Literal["ksg", "histogram"]


# Euler–Mascheroni constant, used to seed the integer digamma recurrence
# digamma(1) = -gamma_e, digamma(m+1) = digamma(m) + 1/m.
_GAMMA_E = 0.5772156649015329

# Upper bound on the automatically chosen number of equal-frequency bins.
# Histogram MI variance grows with the number of joint cells (``B^2``), so even
# when ``n`` is large we keep ``B`` modest; the headline Gaussian collapse is
# already tight well below this cap.
_MAX_AUTO_BINS = 12


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


@njit(parallel=True, cache=True, boundscheck=False, fastmath=True)
def _linfoot_inplace_numba(M: np.ndarray) -> None:
    """In-place Linfoot transform ``M <- sqrt(max(0, 1 - exp(-2 M)))``.

    A single fused parallel pass over the matrix.  The naive numpy version is
    four separate full-array ufunc passes (``exp``, ``subtract``, ``maximum``,
    ``sqrt``), i.e. four times the memory traffic of this one-touch kernel;
    folding them halves the wall time of the normalisation tail shared by both
    MI estimators.
    """
    p0, p1 = M.shape
    for i in prange(p0):
        for j in range(p1):
            v = 1.0 - np.exp(-2.0 * M[i, j])
            if v < 0.0:
                v = 0.0
            M[i, j] = np.sqrt(v)


def _linfoot_inplace(M: np.ndarray) -> None:
    """Linfoot transform, numba-fused when available else numpy in-place."""
    if _HAS_NUMBA:
        _linfoot_inplace_numba(M)
    else:
        np.exp(-2.0 * M, out=M)
        np.subtract(1.0, M, out=M)
        np.maximum(M, 0.0, out=M)
        np.sqrt(M, out=M)


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
# Histogram (plug-in) backend
# -----------------------------------------------------------------------------
def _log_count_table(n: int) -> np.ndarray:
    """``L[m] = log(m)`` for ``m in [1, n]`` with ``L[0] = 0``.

    The histogram kernel only ever takes ``log`` of integer counts in
    ``[1, n]``, so a single precomputed table turns every ``log`` in the inner
    loop into an O(1) lookup — no transcendental calls on the hot path.
    """
    table = np.empty(n + 1, dtype=np.float64)
    table[0] = 0.0
    table[1:] = np.log(np.arange(1, n + 1, dtype=np.float64))
    return table


def _resolve_n_bins(n_bins, n_samples: int) -> int:
    """Resolve ``n_bins`` (an int or ``"auto"``) to a concrete bin count.

    ``"auto"`` is tuned for the ``p >> n`` regime MFCF typically runs in: the
    *binding* constraint on resolution is ``n`` (samples), not ``p``.  With
    equal-frequency bins each marginal cell holds ``~n/B`` points and, under
    independence, each of the ``B^2`` joint cells expects ``~n/B^2`` counts.
    Keeping that expectation at a small constant (``~5``) gives ``B ~ sqrt(n/5)``
    — enough resolution to see structure without shattering the joint table
    into mostly-empty cells (which would inflate variance and the Miller–Madow
    correction).  The result is capped at :data:`_MAX_AUTO_BINS`.
    """
    if n_bins == "auto" or n_bins is None:
        b = int(round((n_samples / 5.0) ** 0.5))
        return max(2, min(b, _MAX_AUTO_BINS))
    b = int(n_bins)
    if b < 2:
        raise ValueError(f"n_bins must be >= 2, got {b}.")
    if b > n_samples:
        raise ValueError(
            f"n_bins={b} exceeds n_samples={n_samples}; pick fewer bins."
        )
    return b


def _equal_frequency_codes(X: np.ndarray, n_bins: int) -> np.ndarray:
    """Map each column of ``X`` to equal-frequency bin codes in ``[0, n_bins)``.

    Returns a C-contiguous ``(p, n)`` int32 array; row ``i`` is column ``i`` of
    ``X`` rank-binned so each of the ``n_bins`` bins receives ``~n/n_bins``
    samples.  Equal-frequency (quantile) binning is the right choice here for
    three reasons: it maximises the marginal entropy (uniform marginals), it is
    a pure rank transform — so the estimate is invariant to any monotone
    reparametrisation of the marginals, exactly the property that motivates MI
    over correlation — and it is robust to fat tails / outliers that would leave
    equal-*width* bins almost all empty.  Ties are broken by the stable argsort
    order, which is harmless for continuous data.
    """
    XT = np.ascontiguousarray(X.T)  # (p, n); rows are columns of X
    p, n = XT.shape
    # quicksort (not stable): equal-frequency binning is indifferent to the
    # order in which exact ties are split, so the cheaper sort is fine.
    order = np.argsort(XT, axis=1, kind="quicksort")  # ascending value -> index
    # The r-th smallest sample of every column goes to bin ``r * B // n``.
    # Scatter that fixed block-code vector straight through the sort
    # permutation — avoids materialising the full integer rank array and the
    # subsequent ``(rank * B) // n`` multiply over all (p, n) entries.
    block = ((np.arange(n) * n_bins) // n).astype(np.int8 if n_bins <= 127 else np.int16)
    codes = np.empty((p, n), dtype=block.dtype)
    np.put_along_axis(codes, order, np.broadcast_to(block, order.shape), axis=1)
    return codes


# NB: no ``cache=True`` — ``numba.get_num_threads()`` pulls in dynamic globals
# that the on-disk cache cannot serialise (it would warn and skip the cache).
@njit(parallel=True, boundscheck=False, fastmath=True)
def _histogram_mi_matrix_numba(
    codes: np.ndarray,
    n_bins: int,
    n_features: int,
    log_table: np.ndarray,
    log_n: float,
    M: np.ndarray,
) -> None:
    """Fill the upper+lower triangle of ``M`` with Miller–Madow MI estimates.

    For each column pair the kernel streams the ``n`` samples once, scattering
    them into a length-``B^2`` joint count table held in thread-local scratch,
    then reads off the marginals and accumulates

        I = (1/n) * sum_{cells>0} N_ab * (log N_ab + log n - log Nx_a - log Ny_b)

    via table lookups (no per-element ``log``).  The plug-in ``I`` is then
    Miller–Madow corrected,

        I_MM = I + (m_x + m_y - m_xy - 1) / (2 n),

    where ``m_x, m_y, m_xy`` are the counts of non-empty marginal / joint cells.
    The correction matters precisely in the ``p >> n`` regime: the raw plug-in
    estimate carries a positive ``O(B^2 / n)`` bias that would otherwise make
    independent columns look weakly dependent.

    Parallelism is over a *manual* equal-size partition of the ``P = p(p-1)/2``
    flat upper-triangle pairs: each of ``numba.get_num_threads()`` threads takes
    one contiguous chunk, recovers its first ``(i, j)`` from the flat offset,
    and then walks the triangle by simple increments — so we materialise neither
    the ``O(P)`` ``triu_indices`` arrays nor per-pair allocations (one set of
    scratch buffers per thread, reused across the whole chunk).

    Parameters
    ----------
    codes : (p, n) int8/int16, C-contiguous
        Equal-frequency bin codes; ``codes[i]`` is column ``i``.
    n_bins : int
        Number of bins ``B``.
    n_features : int
        Number of columns ``p``.
    log_table : (n + 1,) float64
        ``log_table[m] = log(m)`` for ``m >= 1`` (``log_table[0] = 0``).
    log_n : float
        ``log(n)``.
    M : (p, p) float64
        Output buffer; diagonal left untouched.
    """
    n = codes.shape[1]
    p = n_features
    P = p * (p - 1) // 2
    BB = n_bins * n_bins
    inv_n = 1.0 / n
    half_inv_n = 0.5 * inv_n
    n_threads = numba.get_num_threads()

    for t in prange(n_threads):
        # Thread-local scratch, allocated once per thread (not per pair).
        hist = np.empty(BB, dtype=np.int64)
        mx = np.empty(n_bins, dtype=np.int64)
        my = np.empty(n_bins, dtype=np.int64)

        start = t * P // n_threads
        end = (t + 1) * P // n_threads
        if start >= end:
            continue

        # Recover (i, j) for this chunk's first flat index.  ``row_start(r)``,
        # the number of pairs in rows ``0..r-1``, is monotone in r, so a single
        # forward scan (once per thread) lands on the right row; ``j`` follows.
        i = 0
        row_start = 0  # pairs before row i = i*(p-1) - i*(i-1)//2, built up
        while row_start + (p - 1 - i) <= start:
            row_start += p - 1 - i
            i += 1
        j = i + 1 + (start - row_start)

        for idx in range(start, end):
            ci = codes[i]
            cj = codes[j]

            for c in range(BB):
                hist[c] = 0
            for a in range(n_bins):
                my[a] = 0
            # Single O(n) streaming pass over the two columns.
            for s in range(n):
                hist[ci[s] * n_bins + cj[s]] += 1

            # One O(B^2) pass: marginals + joint log-sum + non-empty cells.
            # Using the algebraic split
            #   I = H(X) + H(Y) - H(X,Y)
            #     = (1/n)[ sum_cell N*logN - sum_a Nx*logNx - sum_b Ny*logNy ]
            #       + log n,
            # since sum_b N_ab = Nx_a and sum_a N_ab = Ny_b.  This needs only a
            # single sweep of the joint table plus one O(B) marginal sweep,
            # rather than a second full B^2 entropy pass.
            #
            # The loop is deliberately branchless: ``log_table[0] == 0`` so
            # empty cells contribute nothing to the sums without a guard, and
            # the only data-dependent quantity, the non-empty-cell count
            # ``m_xy``, is accumulated as ``(c > 0)`` (lowered to a setcc, not a
            # mispredicted branch).  Across ``p^2`` pairs the removed B^2
            # branches per pair are the dominant saving.
            joint_logsum = 0.0
            m_xy = 0
            for a in range(n_bins):
                base = a * n_bins
                row = 0
                for b in range(n_bins):
                    c = hist[base + b]
                    row += c
                    my[b] += c
                    joint_logsum += c * log_table[c]
                    m_xy += c > 0
                mx[a] = row

            marg_logsum = 0.0
            for a in range(n_bins):
                marg_logsum += mx[a] * log_table[mx[a]]
                marg_logsum += my[a] * log_table[my[a]]

            mi = (joint_logsum - marg_logsum) * inv_n + log_n
            # Miller–Madow bias correction.  Equal-frequency binning with
            # n >= B fills every marginal bin, so m_x = m_y = n_bins exactly;
            # only the joint occupancy m_xy varies.  The term is negative for
            # inflated independence estimates (m_xy large), which is exactly the
            # bias we need to remove in the p >> n regime.
            mi += (2 * n_bins - m_xy - 1) * half_inv_n

            if mi < 0.0:
                mi = 0.0
            M[i, j] = mi
            M[j, i] = mi

            # Walk to the next upper-triangle pair.
            j += 1
            if j == p:
                i += 1
                j = i + 1


def _histogram_mi_matrix_numpy(
    codes: np.ndarray,
    n_bins: int,
    log_table: np.ndarray,
    log_n: float,
) -> np.ndarray:
    """Pure-numpy fallback for :func:`_histogram_mi_matrix_numba`.

    Same Miller–Madow estimate; one ``np.bincount`` per pair.  Materially
    slower than the numba kernel but dependency-free and used only when numba
    is unavailable.
    """
    p, n = codes.shape
    M = np.zeros((p, p), dtype=np.float64)
    inv_n = 1.0 / n
    half_inv_n = 0.5 * inv_n
    BB = n_bins * n_bins
    for i in range(p):
        ci = codes[i].astype(np.int64)
        for j in range(i + 1, p):
            flat = ci * n_bins + codes[j]
            hist = np.bincount(flat, minlength=BB).reshape(n_bins, n_bins)
            mx = hist.sum(axis=1)
            my = hist.sum(axis=0)
            nz = hist > 0
            cnt = hist[nz]
            a_idx, b_idx = np.nonzero(nz)
            mi_sum = float(
                np.sum(
                    cnt
                    * (
                        log_table[cnt]
                        + log_n
                        - log_table[mx[a_idx]]
                        - log_table[my[b_idx]]
                    )
                )
            )
            mi = mi_sum * inv_n
            m_x = int(np.count_nonzero(mx))
            m_y = int(np.count_nonzero(my))
            m_xy = int(cnt.size)
            mi += (m_x + m_y - m_xy - 1) * half_inv_n
            if mi < 0.0:
                mi = 0.0
            M[i, j] = mi
            M[j, i] = mi
    return M


def _histogram_raw_matrix(
    X: np.ndarray,
    *,
    n_bins: int,
    n_jobs: Optional[int],
    use_numba: bool,
) -> np.ndarray:
    """Raw (un-normalised, ``>= 0``, zero-diagonal) histogram MI matrix."""
    n_samples, n_features = X.shape
    codes = _equal_frequency_codes(X, n_bins)
    log_table = _log_count_table(n_samples)
    log_n = float(np.log(n_samples))
    M = np.zeros((n_features, n_features), dtype=np.float64)

    if use_numba:
        if n_jobs is not None and n_jobs > 0:
            _orig = numba.get_num_threads()
            numba.set_num_threads(int(n_jobs))
            try:
                _histogram_mi_matrix_numba(
                    codes, int(n_bins), int(n_features), log_table, log_n, M
                )
            finally:
                numba.set_num_threads(_orig)
        else:
            _histogram_mi_matrix_numba(
                codes, int(n_bins), int(n_features), log_table, log_n, M
            )
        np.maximum(M, 0.0, out=M)
    else:
        M = _histogram_mi_matrix_numpy(codes, int(n_bins), log_table, log_n)
    return M


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------
def _ksg_raw_matrix(
    X: np.ndarray,
    *,
    n_neighbors: int,
    n_jobs: Optional[int],
    random_state: Optional[int],
    backend: _BackendKind,
) -> np.ndarray:
    """Raw (un-normalised, ``>= 0``, zero-diagonal) KSG MI matrix.

    Factored out of :func:`mutual_information_matrix` so the KSG and histogram
    estimators share one validated wrapper and one normalisation tail.
    """
    n_samples, n_features = X.shape
    if n_samples < n_neighbors + 1:
        raise ValueError(
            f"Need at least n_neighbors + 1 = {n_neighbors + 1} samples for "
            f"the KSG estimator, got {n_samples}."
        )

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
    return M


def mutual_information_matrix(
    X: np.ndarray,
    *,
    estimator: _EstimatorKind = "ksg",
    n_neighbors: int = 3,
    n_bins="auto",
    normalize: _NormalizeKind = "linfoot",
    n_jobs: Optional[int] = None,
    random_state: Optional[int] = None,
    backend: _BackendKind = "auto",
) -> np.ndarray:
    """Pairwise mutual-information similarity matrix.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        Sample-by-feature design matrix.
    estimator : {"ksg", "histogram"}, default="ksg"
        Which MI estimator to use.  ``"ksg"`` is the Kraskov–Stögbauer–
        Grassberger k-nearest-neighbour estimator (most informative, but one
        neighbour search per pair).  ``"histogram"`` is the equal-frequency
        plug-in estimator with a Miller–Madow correction — much faster
        (``O(p^2 n)``, the same scaling as a correlation matrix-multiply) and
        the right choice when speed matters or ``p >> n``.
    n_neighbors : int, default=3
        Number of neighbours for the KSG estimator.  Values in ``[3, 6]`` are
        the usual recommendation; results are robust to the exact choice.
        Ignored when ``estimator="histogram"``.
    n_bins : int or "auto", default="auto"
        Number of equal-frequency bins for the histogram estimator.  ``"auto"``
        scales the bin count with ``n`` (see :func:`_resolve_n_bins`), which is
        the binding constraint in the ``p >> n`` regime.  Ignored when
        ``estimator="ksg"``.
    normalize : {"linfoot", "none"}, default="linfoot"
        ``"linfoot"`` applies ``sqrt(1 - exp(-2 I))`` so entries live in
        ``[0, 1]`` and match ``|Pearson rho|`` under Gaussianity.  ``"none"``
        keeps the raw estimate (clipped to ``>= 0``).
    n_jobs : int or None, default=None
        Thread count for the numba kernels (or sklearn's
        ``mutual_info_regression`` for the KSG sklearn backend).  ``None``
        keeps the backend default.
    random_state : int or None, default=None
        Seed for the KSG tie-breaking jitter (standard Kraskov-2004
        preprocessing).  Unused by the histogram estimator, which is a
        deterministic rank transform.
    backend : {"auto", "numba", "sklearn"}, default="auto"
        Backend selector.  ``"auto"`` picks numba when it is importable, else
        the pure-numpy / sklearn fallback.  ``"sklearn"`` only affects the KSG
        estimator; the histogram estimator falls back to numpy in that case.

    Returns
    -------
    M : np.ndarray, shape (n_features, n_features)
        Symmetric non-negative similarity matrix.  Diagonal is ``1.0`` for
        ``normalize="linfoot"`` (matching the correlation convention) and
        ``0.0`` for ``normalize="none"``.

    Notes
    -----
    Both estimators target the same population ``I(X;Y)`` and therefore the
    same Linfoot-normalised limit, so under joint Gaussianity ``"ksg"``,
    ``"histogram"`` and Pearson ``|rho|`` collapse to the same matrix.  KSG is
    asymptotically the most informative but costs one neighbour search per pair;
    the histogram path is the fast, ``p >> n``-friendly alternative.
    """
    if normalize not in ("linfoot", "none"):
        raise ValueError(
            f"normalize must be 'linfoot' or 'none', got {normalize!r}."
        )
    if backend not in ("auto", "numba", "sklearn"):
        raise ValueError(
            f"backend must be one of 'auto', 'numba', 'sklearn', got {backend!r}."
        )
    if estimator not in ("ksg", "histogram"):
        raise ValueError(
            f"estimator must be 'ksg' or 'histogram', got {estimator!r}."
        )

    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D, got shape {X.shape}.")

    n_samples, n_features = X.shape
    if n_samples < 2:
        raise ValueError(
            f"Need at least 2 samples, got {n_samples}."
        )

    diag_value = 1.0 if normalize == "linfoot" else 0.0
    if n_features < 2:
        M_small = np.zeros((n_features, n_features), dtype=np.float64)
        np.fill_diagonal(M_small, diag_value)
        return M_small

    if estimator == "ksg":
        M = _ksg_raw_matrix(
            X,
            n_neighbors=int(n_neighbors),
            n_jobs=n_jobs,
            random_state=random_state,
            backend=backend,
        )
    else:
        n_bins_resolved = _resolve_n_bins(n_bins, n_samples)
        use_numba = backend != "sklearn" and _HAS_NUMBA
        M = _histogram_raw_matrix(
            X,
            n_bins=n_bins_resolved,
            n_jobs=n_jobs,
            use_numba=use_numba,
        )

    if normalize == "linfoot":
        _linfoot_inplace(M)  # sqrt(max(0, 1 - exp(-2 I)))

    np.fill_diagonal(M, diag_value)
    return M


def histogram_mutual_information_matrix(
    X: np.ndarray,
    *,
    n_bins="auto",
    normalize: _NormalizeKind = "linfoot",
    n_jobs: Optional[int] = None,
    backend: _BackendKind = "auto",
) -> np.ndarray:
    """Convenience wrapper for ``mutual_information_matrix(..., estimator='histogram')``.

    The fast, ``p >> n``-friendly histogram (plug-in) MI estimator on
    equal-frequency bins with a Miller–Madow bias correction.  See
    :func:`mutual_information_matrix` for the full parameter description and
    :func:`_resolve_n_bins` for the ``"auto"`` bin rule.
    """
    return mutual_information_matrix(
        X,
        estimator="histogram",
        n_bins=n_bins,
        normalize=normalize,
        n_jobs=n_jobs,
        backend=backend,
    )
