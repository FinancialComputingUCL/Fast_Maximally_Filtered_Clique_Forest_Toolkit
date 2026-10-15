"""
Maximum Filtering Clique Forest (MFCF)

This module builds a sparse graphical structure (a forest of cliques) from a
weighted matrix `C` (e.g., correlation or covariance). It greedily grows
cliques by selecting the next vertex/separator pair that maximizes a gain
function, while enforcing size and multiplicity constraints on separators.
The builder works on two kinds of input:

* a covariance or correlation matrix, with the exact Gaussian log-likelihood
  gain ("loglik", default); each greedy step can be accepted by a threshold on
  the squared partial correlation, a partial F-test or the BIC;
* a covariance or correlation(-coefficient) matrix with the original sum of
  squared correlations ("sumsquares"), or a non-negative weight (similarity)
  matrix with the sum of weights ("sum"); each edge is accepted by a threshold
  on its (squared) weight.

High level flow:
1) Seed an initial clique using above-mean edges.
2) Maintain a priority queue (PQ) of best (gain, node, separator) candidates.
3) Iteratively pop/validate candidates, add a new clique, and update PQ.
4) Record separators subject to multiplicity and size constraints.
5) Optionally compute a sparse inverse estimator ("logo"): sum of clique
   inverses minus weighted separator inverses.

Key terms
---------
Clique
    A frozenset of node indices.
Separator
    A frozenset representing an intersection/facet used to attach new nodes.
PEO
    Perfect Elimination Order accumulated as cliques are added.

Notes
-----
- `C` can be any dense weight matrix (e.g., correlations). When `cov_matrix`
  is provided to `MFCF.run`, it is used for the logo/inverse aggregation step.
- Shapes: `C` is (N, N). Masks are boolean arrays of length N.
"""

import heapq
import itertools
import logging
import warnings
from collections import Counter
from dataclasses import dataclass
from typing import FrozenSet, Iterable, List, Optional, Tuple

import numpy as np
import numpy.linalg as LA
from scipy import stats

# -----------------------------------------------------------------------------
# Numba kernels of the direct path for the weight gains with threshold 0
# (network generation). Ported from the `mutual_info` branch. Without Numba
# the direct path is not used (its Python fallback is slower than the
# general path).
# -----------------------------------------------------------------------------
try:
    from numba import njit, prange  # type: ignore
    _HAS_NUMBA = True
except Exception:  # pragma: no cover - environments without numba
    _HAS_NUMBA = False

    def njit(*args, **kwargs):  # type: ignore[misc]
        def _dec(fn):
            return fn
        if args and callable(args[0]):
            return args[0]
        return _dec

    def prange(n):  # type: ignore[misc]
        return range(n)


@njit(cache=True, boundscheck=False, fastmath=False)
def _refresh_seps_for_node(
    v: int,
    n_seps: int,
    pending_arr: np.ndarray,
    best_node_arr: np.ndarray,
    has_argsort: np.ndarray,
    ptrs: np.ndarray,
    argsort_mat: np.ndarray,
    outstanding: np.ndarray,
    gain_mat: np.ndarray,
    best_gain_arr: np.ndarray,
) -> None:
    """For every pending sep whose cached best was ``v`` recompute
    the best over outstanding nodes.

    For seps that have already paid the ``argsort`` cost
    (``has_argsort[s]``), we just advance the per-sep pointer past
    consumed nodes — ``O(1)`` amortised.  For not-yet-sorted seps,
    we fall back to an inline ``O(p)`` argmax so the caller can keep
    argsort creation lazy (and amortise it across larger parallel
    batches).
    """
    p = outstanding.shape[0]
    NEG_INF = -np.inf
    for s in range(n_seps):
        if not pending_arr[s]:
            continue
        if best_node_arr[s] != v:
            continue
        if has_argsort[s]:
            ptr = ptrs[s]
            while ptr < p and not outstanding[argsort_mat[s, ptr]]:
                ptr += 1
            ptrs[s] = ptr
            if ptr >= p:
                best_gain_arr[s] = NEG_INF
                best_node_arr[s] = -1
            else:
                idx = argsort_mat[s, ptr]
                best_node_arr[s] = idx
                best_gain_arr[s] = gain_mat[s, idx]
        else:
            bv = NEG_INF
            bi = -1
            for i in range(p):
                if outstanding[i]:
                    g = gain_mat[s, i]
                    if g > bv:
                        bv = g
                        bi = i
            best_node_arr[s] = bi
            best_gain_arr[s] = bv


@njit(cache=True, boundscheck=False, fastmath=False)
def _ensure_best_kernel(
    s: int,
    ptrs: np.ndarray,
    argsort_mat: np.ndarray,
    outstanding: np.ndarray,
    gain_mat: np.ndarray,
    best_gain_arr: np.ndarray,
    best_node_arr: np.ndarray,
) -> None:
    """Advance the argsort pointer for a single sep until it lands on
    an outstanding node, then refresh the cached best.  Per-sep
    sibling of :func:`_refresh_seps_for_node`.
    """
    p = outstanding.shape[0]
    ptr = ptrs[s]
    while ptr < p and not outstanding[argsort_mat[s, ptr]]:
        ptr += 1
    ptrs[s] = ptr
    if ptr >= p:
        best_gain_arr[s] = -np.inf
        best_node_arr[s] = -1
    else:
        idx = argsort_mat[s, ptr]
        best_node_arr[s] = idx
        best_gain_arr[s] = gain_mat[s, idx]


@njit(cache=True, boundscheck=False, fastmath=False, parallel=True)
def _batch_argsort_descending(
    gain_mat: np.ndarray,
    argsort_mat: np.ndarray,
    ids: np.ndarray,
) -> None:
    """Argsort by descending value for a batch of rows in parallel.

    The sequential per-row ``np.argsort`` in numpy is ~70μs and
    has ~30μs of Python wrapper overhead per call.  Folding ``N``
    of them into one Numba kernel with ``prange`` exploits the
    multiple cores available on macOS / Linux dev machines and
    gives a ~6× wall-clock speed-up at N≈1500.  Ties are resolved
    in undefined order, but that does not affect floating-point
    parity in the simple regime because gain ties are statistically
    impossible on real correlation data.
    """
    p = gain_mat.shape[1]
    n = ids.shape[0]
    for i in prange(n):
        s = ids[i]
        order = np.argsort(-gain_mat[s])
        for j in range(p):
            argsort_mat[s, j] = order[j]


@njit(cache=True, boundscheck=False, fastmath=False)
def _initial_argmax(
    s: int,
    outstanding: np.ndarray,
    gain_mat: np.ndarray,
    best_gain_arr: np.ndarray,
    best_node_arr: np.ndarray,
    is_empty: bool,
) -> None:
    """O(p) argmax over outstanding nodes — used to seed the cached
    best when a sep is first created (no argsort precomputed yet).
    """
    p = outstanding.shape[0]
    if is_empty:
        for i in range(p):
            if outstanding[i]:
                best_gain_arr[s] = 0.0
                best_node_arr[s] = i
                return
        best_gain_arr[s] = -np.inf
        best_node_arr[s] = -1
        return
    bv = -np.inf
    bi = -1
    for i in range(p):
        if outstanding[i]:
            g = gain_mat[s, i]
            if g > bv:
                bv = g
                bi = i
    best_gain_arr[s] = bv
    best_node_arr[s] = bi


@njit(cache=True, boundscheck=False, fastmath=False)
def _ensure_best_lazy_kernel(
    s: int,
    has_argsort: np.ndarray,
    ptrs: np.ndarray,
    argsort_mat: np.ndarray,
    outstanding: np.ndarray,
    gain_mat: np.ndarray,
    best_gain_arr: np.ndarray,
    best_node_arr: np.ndarray,
    is_empty: bool,
) -> None:
    """Refresh a single sep's cached best.

    Uses the argsort+ptr trick when ``has_argsort[s]`` is set;
    otherwise falls back to an O(p) argmax over outstanding nodes
    (so the caller can keep argsort-creation lazy).
    """
    p = outstanding.shape[0]
    if has_argsort[s]:
        ptr = ptrs[s]
        while ptr < p and not outstanding[argsort_mat[s, ptr]]:
            ptr += 1
        ptrs[s] = ptr
        if ptr >= p:
            best_gain_arr[s] = -np.inf
            best_node_arr[s] = -1
        else:
            idx = argsort_mat[s, ptr]
            best_node_arr[s] = idx
            best_gain_arr[s] = gain_mat[s, idx]
        return
    # No argsort yet — do an O(p) argmax over outstanding rows.
    if is_empty:
        for i in range(p):
            if outstanding[i]:
                best_gain_arr[s] = 0.0
                best_node_arr[s] = i
                return
        best_gain_arr[s] = -np.inf
        best_node_arr[s] = -1
        return
    bv = -np.inf
    bi = -1
    for i in range(p):
        if outstanding[i]:
            g = gain_mat[s, i]
            if g > bv:
                bv = g
                bi = i
    best_gain_arr[s] = bv
    best_node_arr[s] = bi


@njit(cache=True, boundscheck=False, fastmath=False)
def _argmax_pending(
    n_seps: int,
    pending_arr: np.ndarray,
    best_gain_arr: np.ndarray,
    best_node_arr: np.ndarray,
) -> int:
    """Return the sep_id with the maximum ``(best_gain, -best_node)``
    among pending entries.  Returns -1 if no candidate is valid.
    """
    best_id = -1
    best_g = -np.inf
    best_n = np.iinfo(np.int64).max
    for s in range(n_seps):
        if not pending_arr[s]:
            continue
        g = best_gain_arr[s]
        if g > best_g:
            best_g = g
            best_n = best_node_arr[s]
            best_id = s
        elif g == best_g:
            n = best_node_arr[s]
            if n < best_n:
                best_n = n
                best_id = s
    if best_id == -1 or not np.isfinite(best_g):
        return -1
    return best_id

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logger = logging.getLogger(__name__)
# Example usage from caller:
# logging.basicConfig(level=logging.INFO)  # or DEBUG


# -----------------------------------------------------------------------------
# Type aliases
# -----------------------------------------------------------------------------
Node = int
Clique = FrozenSet[int]
Separator = FrozenSet[int]


# -----------------------------------------------------------------------------
# Separator wrapper for comparison in priority queue
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class SeparatorWrapper:
    """
    Wrap a separator with an additional "prior-threshold" separator for
     membership checks in the PQ.

    Parameters
    ----------
    separator
        The (possibly reduced) separator actually used to score a candidate.
    separator_prior_threshold
        The separator as it existed prior to threshold enforcement; used as the
        identity in `_pq_separators` to avoid duplicate PQ entries.
    """

    separator: FrozenSet[int]
    separator_prior_threshold: FrozenSet[int]

    def __eq__(self, other: object) -> bool:
        if isinstance(other, SeparatorWrapper):
            return self.separator == other.separator
        return NotImplemented

    def __le__(self, other: object) -> bool:  # subset-or-equal comparison
        if isinstance(other, SeparatorWrapper):
            return (
                self.separator <= other.separator
                or self.separator_prior_threshold <= other.separator_prior_threshold
            )
        return NotImplemented

    def __ge__(self, other: object) -> bool:  # superset-or-equal comparison
        if isinstance(other, SeparatorWrapper):
            return (
                self.separator >= other.separator
                or self.separator_prior_threshold >= other.separator_prior_threshold
            )
        return NotImplemented

    def __lt__(self, other):
        if isinstance(other, SeparatorWrapper):
            return (
                self.separator < other.separator
                or self.separator_prior_threshold < other.separator_prior_threshold
            )
        return NotImplemented

    def __gt__(self, other):
        if isinstance(other, SeparatorWrapper):
            return (
                self.separator > other.separator
                or self.separator_prior_threshold > other.separator_prior_threshold
            )
        return NotImplemented


# -----------------------------------------------------------------------------
# Helper formatting (debug/pretty-print utilities)
# -----------------------------------------------------------------------------
def format_frozenset(fs: Iterable[int]) -> str:
    """
    Convert an iterable of node ids to a readable, sorted list string.

    Parameters
    ----------
    fs
        Iterable of items (ideally ints) to display.

    Returns
    -------
    str
        A string like ``"[0, 2, 5]"``; returns ``"[]"`` for empty iterables.

    Notes
    -----
    If casting to `int` fails, this falls back to sorting by default order.
    """
    try:
        return str(sorted(int(x) for x in fs)) if fs else "[]"
    except (ValueError, TypeError):
        return str(sorted(list(fs))) if fs else "[]"


# =============================================================================
# Option checks
# =============================================================================
GAIN_TYPES = ("loglik", "sumsquares", "sum")
ACCEPTANCE_RULES = ("threshold", "ftest", "bic")


def _check_options(gf_type, acceptance, alpha, n_samples, require_n: bool = True) -> None:
    if gf_type not in GAIN_TYPES:
        raise ValueError(f"Unknown gain function type: {gf_type!r}; choose one of {GAIN_TYPES}")
    if acceptance not in ACCEPTANCE_RULES:
        raise ValueError(f"Unknown acceptance rule: {acceptance!r}; choose one of {ACCEPTANCE_RULES}")
    if acceptance != "threshold":
        if gf_type != "loglik":
            raise ValueError(
                f"acceptance={acceptance!r} is a statistical test and requires "
                "gain_function_type='loglik'; weight gains use acceptance='threshold'"
            )
        if require_n and (n_samples is None or float(n_samples) < 3):
            raise ValueError(f"acceptance={acceptance!r} requires n_samples (number of observations); "
                             "pass n_samples to MFCF or build from data with fit(X)")
        if acceptance == "ftest" and not (0 < alpha < 1):
            raise ValueError("alpha must be in (0, 1)")


def _check_covariance(C: np.ndarray, gain: str = "loglik") -> None:
    msg = (f"gain_function_type='{gain}' requires a covariance or correlation matrix "
           "(symmetric, positive semi-definite, positive diagonal); for a "
           "non-negative weight matrix use gain_function_type='sum'")
    if C.ndim != 2 or C.shape[0] != C.shape[1] or not np.allclose(C, C.T, atol=1e-10):
        raise ValueError(msg)
    if np.any(np.diag(C) <= 0):
        raise ValueError(msg)
    # positive semi-definite up to a small tolerance: Cholesky of C + eps I
    # (an order of magnitude faster than an eigendecomposition)
    eps = 1e-8 * max(float(np.diag(C).max()), 1.0)
    try:
        np.linalg.cholesky((C + C.T) / 2 + eps * np.eye(C.shape[0]))
    except np.linalg.LinAlgError:
        raise ValueError(msg) from None


def _check_correlation_like(C: np.ndarray) -> None:
    """For "sumsquares": a covariance, correlation or other correlation
    coefficient matrix (e.g. Linfoot's information coefficient): symmetric,
    positive diagonal and |C_ij| <= sqrt(C_ii C_jj). Positive
    semi-definiteness is not needed for this gain."""
    msg = ("gain_function_type='sumsquares' requires a covariance or correlation "
           "matrix (symmetric, positive diagonal, |C_ij| <= sqrt(C_ii C_jj)); for a "
           "non-negative weight matrix use gain_function_type='sum'")
    if C.ndim != 2 or C.shape[0] != C.shape[1] or not np.allclose(C, C.T, atol=1e-10):
        raise ValueError(msg)
    d = np.diag(C)
    if np.any(d <= 0) or not np.all(np.isfinite(C)):
        raise ValueError(msg)
    if np.any(np.abs(C) > np.sqrt(np.outer(d, d)) * (1 + 1e-8) + 1e-12):
        raise ValueError(msg)


def _check_nonnegative(W: np.ndarray) -> None:
    if W.ndim != 2 or W.shape[0] != W.shape[1]:
        raise ValueError("gain_function_type='sum' requires a square weight matrix")
    if not np.all(np.isfinite(W)) or not np.allclose(W, W.T, atol=1e-10):
        raise ValueError("gain_function_type='sum' requires a finite, symmetric weight matrix")
    off = W[~np.eye(W.shape[0], dtype=bool)]
    if np.any(off < 0):
        raise ValueError(
            "gain_function_type='sum' requires a non-negative weight matrix; "
            "transform the weights (e.g. np.abs(W)); for a covariance or "
            "correlation matrix use 'loglik' or 'sumsquares'"
        )


__version__ = "2026.10.0"

_DEFAULT_CHANGE_MSG = (
    "The default gain_function_type changed from 'sumsquares' to 'loglik' "
    "(exact Gaussian log-likelihood gain) in version 2026.10. Pass "
    "gain_function_type explicitly to silence this warning; for the results "
    "of earlier versions use gain_function_type='sumsquares', "
    "new_component='first'.")


def mfcf_control() -> dict:
    """Default control dictionary of the original implementation, for
    ``MFCF().fast_mfcf(C, ctl, gf_type)`` (deprecated interface)."""
    return {
        "min_clique_size": 1,
        "max_clique_size": 4,
        "coordination_number": np.inf,
        "cachesize": np.inf,
        "threshold": 0.00,
        "drop_sep": True,
        "method": "MFCF",
    }


def _kendall_tau_b(X: np.ndarray, block: int = 256, gram_max_n: int = 2000) -> np.ndarray:
    """Kendall's tau-b for all pairs of columns. Short series (n <= gram_max_n):
    normalised Gram matrix of pairwise sign differences (O(n^2 p^2) operations
    in BLAS, O(block n p) memory). Long series: Knight's O(n log n) algorithm
    for each pair (scipy)."""
    n, p = X.shape
    if np.any(np.ptp(X, axis=0) == 0):
        raise ValueError("X has constant columns")
    if n > gram_max_n:
        T = np.eye(p)
        for i in range(p):
            for j in range(i):
                T[i, j] = T[j, i] = stats.kendalltau(X[:, i], X[:, j])[0]
        return T
    G = np.zeros((p, p))
    for k0 in range(0, n - 1, block):
        rows = range(k0, min(k0 + block, n - 1))
        D = np.concatenate([np.sign(X[k + 1:] - X[k]) for k in rows])
        G += D.T @ D
    d = np.sqrt(np.diag(G))
    if np.any(d == 0):
        raise ValueError("X has constant columns")
    return G / np.outer(d, d)


def _rank_correlation(X: np.ndarray, method: str) -> np.ndarray:
    """Gaussian-copula correlation from Spearman's rho or Kendall's tau:
    ``2 sin(pi rho_s / 6)`` and ``sin(pi tau / 2)``, projected onto the
    positive definite correlation matrices when the mapping leaves them."""
    if method == "spearman":
        Rk = stats.rankdata(X, axis=0)
        R = np.corrcoef(Rk, rowvar=False)
        R = 2 * np.sin(np.pi * R / 6)
    else:
        R = np.sin(np.pi * _kendall_tau_b(X) / 2)
    R = (R + R.T) / 2
    np.fill_diagonal(R, 1.0)
    w, V = LA.eigh(R)
    if w.min() < 1e-8:
        R = (V * np.maximum(w, 1e-8)) @ V.T
        d = 1 / np.sqrt(np.diag(R))
        R = R * np.outer(d, d)
        R = (R + R.T) / 2
        np.fill_diagonal(R, 1.0)
    return R


def _effective_sample_size(Xc: np.ndarray) -> float:
    """Effective number of observations for sample correlations of
    autocorrelated series (Bartlett): ``n / mean_{i<j} (1 + phi_i phi_j) /
    (1 - phi_i phi_j)``, with ``phi`` the lag-1 autocorrelations, clipped to
    [-0.99, 0.99]; never larger than n."""
    n, p = Xc.shape
    Xc = Xc - Xc.mean(axis=0)          # autocorrelations about the sample mean
    num = np.sum(Xc[1:] * Xc[:-1], axis=0)
    den = np.sum(Xc * Xc, axis=0)
    phi = np.clip(num / den, -0.99, 0.99)
    pp = np.outer(phi, phi)
    infl = (1 + pp) / (1 - pp)
    iu = np.triu_indices(p, 1)
    factor = max(1.0, float(np.mean(infl[iu]))) if p > 1 else 1.0
    return max(3.0, n / factor)


# =============================================================================
# Gains
# =============================================================================
class Gains:
    """
    Gain function handler.

    Two gains are implemented for a candidate node `v` attached to a subset `S`
    of a separator (or clique) candidate:

    - "loglik" (default): the exact Gaussian log-likelihood gain per
      observation, ``g = -1/2 log(1 - R^2_{v|S})``, where ``R^2_{v|S}`` is the
      coefficient of determination of the regression of `v` on `S` computed
      from `C`. `S` is grown greedily inside the candidate separator, adding at
      each step the vertex with the largest squared partial correlation with
      `v` given the vertices already chosen; a step is kept if that squared
      partial correlation is ``>= threshold`` (the first step uses the plain
      squared correlation, as in "sumsquares").
    - "sumsquares": ``sum_{j in S} C[v, j]^2`` over the entries
      ``C[v, j]^2 >= threshold``; `C` must be a covariance or a matrix of
      correlation coefficients, ``|C_ij| <= sqrt(C_ii C_jj)`` (checked).
    - "sum": ``sum_{j in S} C[v, j]`` over the entries ``C[v, j] >= threshold``;
      `C` must be non-negative off the diagonal (checked). "sumsquares" is
      "sum" applied to the non-negative weights ``C ** 2``.

    In all cases the top ``min_clique_size - 1`` picks are mandatory.

    Acceptance of each greedy step ("loglik" only; weight gains always use the
    threshold):

    - "threshold": squared partial correlation ``>= threshold``;
    - "ftest": partial F-test of the added coefficient at level `alpha`,
      i.e. ``rho^2 >= F / (F + n - q - 1)`` with ``F`` the ``1 - alpha``
      quantile of ``F(1, n - q - 1)`` and ``q`` the number of predictors after
      the step;
    - "bic": ``n * (-log(1 - rho^2)) > log(n)``.

    Parameters
    ----------
    C : np.ndarray, shape (N, N)
        Weight matrix (e.g., correlation). Only `np.square(C)` is used by the
        gain, so `C` need not be symmetric here.
    threshold : float, default=0.0
        Edge-wise threshold; only weights `>= threshold` contribute to the base
        gain.
    min_clique_size : int, default=1
        Minimum size of a clique. When adding a node to separator `S`, we need
        at least `min_clique_size - 1` elements in `S`. If some of the top-k
        edges fall below `threshold`, they are still counted to ensure the size.
    gf_type : {"loglik", "sumsquares", "sum"}, default="loglik"
        Type of gain function.
    acceptance : {"threshold", "ftest", "bic"}, default="threshold"
        Rule for accepting each greedy step of the "loglik" gain.
    alpha : float, default=0.05
        Level of the partial F-test (acceptance="ftest").
    n_samples : int, optional
        Number of observations behind `C`; required by "ftest" and "bic".

    Notes
    -----
    "loglik" requires `C` to be a correlation or covariance matrix; it is
    evaluated for all outstanding nodes at once by sequential sweeps of the
    conditional covariance. "sumsquares" precomputes `W = C ** 2`.
    """

    def __init__(
        self,
        C: np.ndarray,
        threshold: float = 0.0,
        min_clique_size: int = 1,
        gf_type: str = "loglik",
        acceptance: str = "threshold",
        alpha: float = 0.05,
        n_samples: Optional[int] = None,
        root_chooser=None,
    ):
        _check_options(gf_type, acceptance, alpha, n_samples)
        # vertex used to start a new component (default: lowest index)
        self._root = root_chooser or (lambda mask: int(np.argmax(mask)))
        if gf_type == "sumsquares":
            _check_correlation_like(np.asarray(C, dtype=float))
            self._W = np.square(C)
        elif gf_type == "sum":
            self._W = np.asarray(C, dtype=float)
            _check_nonnegative(self._W)
        else:  # "loglik"
            self._C = np.asarray(C, dtype=float)
            _check_covariance(self._C)
            self._diag = np.diag(self._C).copy()
            self._loglik_cache = {}            # separator -> gains of all nodes
            self._loglik_kept = {}             # (separator, node) -> kept subset

        self._gf_type = gf_type
        self._threshold: float = threshold
        self._min_clique_size: int = min_clique_size
        self._acceptance = acceptance
        self._alpha = alpha
        self._n = n_samples
        # every greedy step accepted: the whole separator is used
        self._accept_all = acceptance == "threshold" and threshold <= 0
        self._crit_cache = {}

    def _critical_pc2(self, t: int) -> float:
        """Smallest squared partial correlation accepted at greedy step `t`
        (0-based), i.e. with ``q = t + 1`` predictors after the step."""
        if self._acceptance == "threshold":
            return self._threshold
        c = self._crit_cache.get(t)
        if c is None:
            n = self._n
            if self._acceptance == "ftest":
                df2 = n - (t + 1) - 1
                if df2 <= 0:
                    c = np.inf
                else:
                    F = stats.f.ppf(1 - self._alpha, 1, df2)
                    c = F / (F + df2)
            else:  # "bic"
                c = -np.expm1(-np.log(n) / n)
            self._crit_cache[t] = c
        return c

    def get_best_gain(
        self,
        outstanding_nodes_mask: np.ndarray,
        sep: "Separator",
    ) -> Tuple[float, "Node", "SeparatorWrapper"]:
        """
        Compute the best (gain, node, kept-separator) for the given separator.

        This is vectorized over all outstanding nodes, identifies the row with
        maximal gain, and reconstructs the subset of `sep` that contributes
        (i.e., passes the threshold plus mandatory top-k if needed).

        Parameters
        ----------
        outstanding_nodes_mask : np.ndarray, shape (N,)
            Boolean mask: True for nodes not yet added to any clique.
        sep : Separator
            Proposed separator to attach a new node to. May be empty.

        Returns
        -------
        best_gain : float
            The maximum gain achieved for this separator. If `sep` is empty,
            the gain is 0.0 by definition (and the first outstanding node is chosen).
        best_node : int
            Index of the node achieving `best_gain`.
        best_sep : SeparatorWrapper
            Wrapper containing the kept subset of `sep` (after threshold/top-k)
            and the original `sep` as `separator_prior_threshold`.

        Notes
        -----
        - When `sep` is empty, all gains are 0.0; we pick the first available
          node to seed a new component.
        - Mandatory picks: `k = max(0, min(min_clique_size - 1, |sep|))`.
        """
        if not sep:
            # With empty sep, all gains are 0; choose the first outstanding node
            first = self._root(outstanding_nodes_mask)
            empty = frozenset()
            return 0.0, first, SeparatorWrapper(empty, empty)

        if self._gf_type == "loglik":
            return self._best_gain_loglik(outstanding_nodes_mask, sep)

        # 1) Prepare indices/submatrix and mandatory-picks k
        rows, cols, W_sub = self._prepare_submatrix(outstanding_nodes_mask, sep)

        # number of mandatory picks from top weights
        k = max(0, min(self._min_clique_size - 1, W_sub.shape[1]))

        # 2) Compute row-wise gains and masks needed for reconstruction
        gains, T, topk_idx, topk_in_T = self._row_gains(W_sub, self._threshold, k)

        # 3) Pick best row and reconstruct the kept separator columns for that row
        best_row_pos = int(np.argmax(gains))
        best_node = int(rows[best_row_pos])
        best_gain = float(gains[best_row_pos])

        best_sep = self._separator_for_row(
            cols=cols,
            T_row=T[best_row_pos],
            row_topk_idx=topk_idx[best_row_pos],
            row_topk_in_T=topk_in_T[best_row_pos],
            k=k,
            sep=sep,
        )

        return best_gain, best_node, best_sep

    def _best_gain_loglik(
        self,
        outstanding_nodes_mask: np.ndarray,
        sep: "Separator",
    ) -> Tuple[float, "Node", "SeparatorWrapper"]:
        """
        Exact log-likelihood gain for the best outstanding node.

        The gain of a (separator, node) pair does not change during a run, so
        the gains of all nodes for a separator are computed once and cached;
        later calls only mask the nodes that are no longer outstanding. The
        kept subset is reconstructed for the selected node only.

        Returns the same triple as `get_best_gain`.
        """
        key = sep if isinstance(sep, frozenset) else frozenset(sep)
        v, gain = self._cached_best(key, outstanding_nodes_mask)
        if v is None:
            cols = np.asarray(sorted(key), dtype=int)
            rows = np.setdiff1d(np.flatnonzero(outstanding_nodes_mask), cols)
            if len(rows) == 0:                       # nothing to attach to this separator
                return -np.inf, self._root(outstanding_nodes_mask), SeparatorWrapper(frozenset(), key)
            g, _ = self._loglik_sweep(rows, cols)
            self._store(key, rows, g)
            v, gain = self._cached_best(key, outstanding_nodes_mask)
        c = len(key)
        if self._accept_all or max(0, min(self._min_clique_size - 1, c)) >= c:
            kept = key                                  # whole separator is used
        else:
            kept = self._loglik_kept.get((key, v))
            if kept is None:
                cols = np.asarray(sorted(key), dtype=int)
                _, chosen = self._loglik_sweep(np.array([v]), cols)
                kept = frozenset(int(x) for x in cols[chosen[0]])
                self._loglik_kept[(key, v)] = kept
        return float(gain), v, SeparatorWrapper(kept, key)

    _TOP = 32   # candidates kept per separator (gains do not change during a run)

    def _store(self, key, rows: np.ndarray, gains: np.ndarray) -> None:
        """Keep the best candidates of a separator, ordered by decreasing gain
        and increasing node index (the tie-break of argmax)."""
        finite = np.isfinite(gains)
        rows, gains = rows[finite], gains[finite]
        order = np.lexsort((rows, -gains))[: self._TOP]
        self._loglik_cache[key] = (rows[order], gains[order])

    def _cached_best(self, key, outstanding_nodes_mask: np.ndarray):
        """Best outstanding candidate from the cache, or (None, None) if the
        cache is missing or all its candidates have been used."""
        entry = self._loglik_cache.get(key)
        if entry is None:
            return None, None
        idx, g = entry
        pos = np.flatnonzero(outstanding_nodes_mask[idx])
        if pos.size == 0:
            return None, None
        return int(idx[pos[0]]), float(g[pos[0]])

    def _loglik_sweep(self, rows: np.ndarray, cols: np.ndarray, cap: Optional[int] = None):
        """
        Greedy subset selection inside `cols` for every node in `rows`.

        At each step the vertex `u` with the largest squared partial
        correlation ``rho^2_{vu|S}`` is added and the conditional covariance is
        updated by a sweep, so that ``1 - R^2_{v|S}`` is exact. A step is kept
        if it is one of the mandatory ``min_clique_size - 1`` picks or if
        ``rho^2_{vu|S} >= threshold``; once a step is rejected, growth stops.

        Returns the gains ``-1/2 log(1 - R^2_{v|S})`` and the boolean matrix
        of chosen columns.
        """
        C = self._C
        m, c = len(rows), len(cols)
        cap = c if cap is None else min(cap, c)
        k = max(0, min(self._min_clique_size - 1, cap))
        if c and cap == c and (self._accept_all or k >= c):
            # every step is accepted, so S is the whole separator: one regression
            fast = self._loglik_full(rows, cols)
            if fast is not None:
                return fast
        Mcc = np.broadcast_to(C[np.ix_(cols, cols)], (m, c, c)).copy()  # cov(u, w | S)
        mvc = C[np.ix_(rows, cols)].copy()                                # cov(v, u | S)
        mvv = self._diag[rows].copy()                                     # var(v | S)
        v0 = mvv.copy()
        d0 = self._diag[cols]
        tol = 1e-12
        active = np.ones((m, c), dtype=bool)
        alive = np.ones(m, dtype=bool)
        chosen = np.zeros((m, c), dtype=bool)
        ar = np.arange(m)
        for t in range(cap):
            duu = np.einsum("mii->mi", Mcc)
            valid = active & alive[:, None] & (duu > tol * d0) & (mvv[:, None] > tol * v0[:, None])
            denom = np.where(valid, duu * mvv[:, None], 1.0)
            pc2 = np.where(valid, mvc ** 2 / denom, -1.0)
            j = np.argmax(pc2, axis=1)
            best = pc2[ar, j]
            ok = alive & (best >= 0)
            if t >= k:
                ok &= best >= self._critical_pc2(t)
            alive = ok
            if not ok.any():
                break
            r = np.flatnonzero(ok)
            jj = j[r]
            chosen[r, jj] = True
            col = Mcc[r, :, jj]                       # cov(., u* | S), shape (len(r), c)
            dvar = col[np.arange(len(r)), jj]
            cv = mvc[r, jj]
            Mcc[r] -= col[:, :, None] * col[:, None, :] / dvar[:, None, None]
            mvc[r] -= cv[:, None] * col / dvar[:, None]
            mvv[r] -= cv ** 2 / dvar
            active[r, jj] = False
        ratio = np.clip(mvv / v0, tol, 1.0)
        gains = np.where(chosen.any(axis=1), -0.5 * np.log(ratio), 0.0)
        return gains, chosen

    def precompute_facets(self, clique: Iterable[int],
                          outstanding_nodes_mask: Optional[np.ndarray] = None) -> None:
        """
        Cache the gains of all facets ``clique - {w}`` of a full clique.

        - Whole facets (``threshold <= 0`` or mandatory picks fill the facet):
          one regression on the clique; dropping `w` raises the residual
          variance of `v` by ``beta_w^2 / Omega_ww``, with
          ``Omega = C_QQ^{-1}`` and ``beta = Omega C_Qv``.
        - Otherwise: one greedy sweep over the clique (at most ``|Q| - 1``
          picks). For a node whose greedy path never picked `w`, the greedy
          path inside the facet ``Q - {w}`` is identical, so only the nodes
          that picked `w` are recomputed for that facet.
        """
        if self._gf_type != "loglik":
            return
        q = np.asarray(sorted(clique), dtype=int)
        c = len(q) - 1
        if c < 1:
            return
        C = self._C
        mask = np.ones(C.shape[0], dtype=bool) if outstanding_nodes_mask is None else outstanding_nodes_mask
        rows = np.setdiff1d(np.flatnonzero(mask), q)
        if len(rows) == 0:
            return
        keys = [frozenset(int(x) for x in q if x != w) for w in q]
        if all(key in self._loglik_cache for key in keys):
            return
        whole = self._accept_all or max(0, min(self._min_clique_size - 1, c)) >= c
        if whole:
            Cqq = C[np.ix_(q, q)]
            try:
                L = np.linalg.cholesky(Cqq)
            except np.linalg.LinAlgError:
                return
            if np.min(np.diag(L)) ** 2 <= 1e-12 * np.max(np.diag(Cqq)):
                return
            Omega = np.linalg.inv(Cqq)
            B = Omega @ C[np.ix_(q, rows)]                      # regression coefficients
            resid = self._diag[rows] - np.sum(C[np.ix_(q, rows)] * B, axis=0)
            var = self._diag[rows]
            for i, key in enumerate(keys):
                if key in self._loglik_cache:
                    continue
                vf = resid + B[i] ** 2 / Omega[i, i]
                self._store(key, rows, -0.5 * np.log(np.clip(vf / var, 1e-12, 1.0)))
            return
        g_all, chosen = self._loglik_sweep(rows, q, cap=c)
        for i, key in enumerate(keys):
            if key in self._loglik_cache:
                continue
            g = g_all.copy()
            redo = chosen[:, i]
            if redo.any():
                cols = np.asarray(sorted(key), dtype=int)
                g[redo], _ = self._loglik_sweep(rows[redo], cols)
            self._store(key, rows, g)

    def _loglik_full(self, rows: np.ndarray, cols: np.ndarray):
        """
        Gain when the whole separator is used: ``R^2_{v|S}`` from one solve for
        all nodes. Returns None if the separator block is numerically singular,
        in which case the sweep handles the degenerate directions.
        """
        C = self._C
        Css = C[np.ix_(cols, cols)]
        Csv = C[np.ix_(cols, rows)]
        try:
            L = np.linalg.cholesky(Css)
        except np.linalg.LinAlgError:
            return None
        if np.min(np.diag(L)) ** 2 <= 1e-12 * np.max(np.diag(Css)):
            return None
        Y = np.linalg.solve(L, Csv)                     # L^{-1} C_{S v}
        explained = np.sum(Y * Y, axis=0)               # C_vS C_SS^{-1} C_Sv
        var = self._diag[rows]
        ratio = np.clip(1.0 - explained / var, 1e-12, 1.0)
        gains = -0.5 * np.log(ratio)
        chosen = np.ones((len(rows), len(cols)), dtype=bool)
        return gains, chosen

    # --------------------
    # Helpers
    # --------------------

    def _prepare_submatrix(
        self,
        outstanding_nodes_mask: np.ndarray,
        sep: "Separator",
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Slice squared-weight matrix to rows of outstanding nodes and columns of `sep`.

        Returns
        -------
        rows : np.ndarray
            Indices of outstanding nodes.
        cols : np.ndarray
            Sorted array of separator indices (dtype=int).
        W_sub : np.ndarray, shape (len(rows), len(cols))
            Submatrix `self._W[np.ix_(rows, cols)]`.
        """
        rows = np.flatnonzero(outstanding_nodes_mask)
        cols = np.asarray(list(sep), dtype=int)

        W_sub = self._W[np.ix_(rows, cols)]
        return rows, cols, W_sub

    def _row_gains(
        self,
        W_sub: np.ndarray,
        threshold: float,
        k: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute per-row gains and book-keeping for top-k mandatory picks.

        Parameters
        ----------
        W_sub : np.ndarray
            Squared weights over (candidate rows, separator columns).
        threshold : float
            Edge threshold applied elementwise.
        k : int
            Number of mandatory top entries per row to include even if below
            threshold (to satisfy the minimum clique size).

        Returns
        -------
        gains : np.ndarray, shape (R,)
            Row-wise gains.
        T : np.ndarray, shape (R, C)
            Boolean mask where `W_sub >= threshold`.
        topk_idx : np.ndarray, shape (R, k)
            Column indices (in `W_sub`) of the row-wise top-k values.
        topk_in_T : np.ndarray, shape (R, k)
            Whether each top-k entry is already above threshold.
        """
        T = W_sub >= threshold
        gains = np.where(T, W_sub, 0.0).sum(axis=1)

        if k:
            topk_idx = np.argpartition(W_sub, -k, axis=1)[:, -k:]
            topk_vals = np.take_along_axis(W_sub, topk_idx, axis=1)
            topk_in_T = np.take_along_axis(T, topk_idx, axis=1)
            gains += np.where(~topk_in_T, topk_vals, 0.0).sum(axis=1)
        else:
            # Keep shapes consistent
            R = W_sub.shape[0]
            topk_idx = np.empty((R, 0), dtype=int)
            topk_in_T = np.empty((R, 0), dtype=bool)

        return gains, T, topk_idx, topk_in_T

    def _separator_for_row(
        self,
        cols: np.ndarray,
        T_row: np.ndarray,
        row_topk_idx: np.ndarray,
        row_topk_in_T: np.ndarray,
        k: int,
        sep: "Separator",
    ) -> "SeparatorWrapper":
        """
        Reconstruct the kept subset of `sep` for a chosen row.

        The kept subset includes all columns above threshold plus any of the
        row's top-k indices that were below threshold.

        Returns
        -------
        SeparatorWrapper
            With `separator` = kept subset and
            `separator_prior_threshold` = original `sep`.
        """
        keep_mask = T_row.copy()
        if k:
            keep_mask[row_topk_idx[~row_topk_in_T]] = True

        kept = frozenset(cols[keep_mask])
        return SeparatorWrapper(kept, frozenset(sep))


# =============================================================================
# MFCF
# =============================================================================
class MFCF:
    """
    Maximally Filtered Clique Forest (MFCF) builder.

    Parameters
    ----------
    threshold : float, default=0.0
        Global edge threshold used in gain computation and in deciding whether
        to start a new component when a popped candidate has insufficient gain.
    min_clique_size : int, default=1
        Minimum size of any clique produced.
    max_clique_size : int, default=4
        Upper bound on clique size. When a clique reaches this size, its facets
        (size-1 subsets) are used as candidate separators.
    coordination_number : int or float, default=np.inf
        Maximum multiplicity allowed for any separator (how many times it can be
        recorded/used). Use `np.inf` to disable.
    gain_function_type : {"loglik", "sumsquares", "sum"}, default="loglik"
        "loglik": exact Gaussian log-likelihood gain, for a covariance or
        correlation matrix `C`. "sumsquares": sum of squared correlations, the
        original approximation, for a covariance or correlation matrix `C`.
        "sum": sum of weights, for a non-negative weight / similarity matrix
        `C`. Until version 2026.10 the default was "sumsquares"; leaving it
        unset issues a FutureWarning for this release.
    acceptance : {"threshold", "ftest", "bic"}, default="threshold"
        Rule for accepting each greedy attachment step. "threshold" uses
        `threshold` (squared partial correlation for "loglik", (squared) weight
        for the weight gains). "ftest" (partial F-test at level `alpha`) and
        "bic" are statistical validations, available for "loglik" only, and
        require `n_samples`.
    alpha : float, default=0.05
        Level of the partial F-test.
    n_samples : int, optional
        Number of observations used to compute `C` ("ftest", "bic").
    new_component : {"strongest", "first"}, default="strongest"
        Vertex that starts a new component when no attachment is accepted:
        "strongest" takes the outstanding vertex with the largest sum of
        squared similarities to the other outstanding vertices (independent
        of the labelling of the variables); "first" takes the lowest index,
        as in versions before 2026.10.
    n_starts : int or None, default=None
        Number of starting cliques, at most `max_clique_size`. The expansion
        is greedy and its result depends on where it starts: with
        ``n_starts=k`` it is run from the default seed and from k - 1 further
        seeds spread over the network (strong vertices weakly related to the
        seeds already chosen), and the clique forest with the largest
        objective is kept (the decomposable Gaussian log-likelihood for
        "loglik", the total edge weight, squared for "sumsquares",
        otherwise). Costs k runs. None means `max_clique_size` for "loglik",
        where it improves the fit to new data under every acceptance rule,
        and 1 for the weight gains, where it brings little. ``start_``
        records the start chosen.
    drop_sep : bool, default=False
        If True, a separator is used for at most one attachment (option of
        the original implementation).
    seed : {"strongest", "tmfg"}, default="strongest"
        First clique: the strongest vertex (or the ``min_clique_size - 1``
        strongest), or the four vertices with the largest sums of above-mean
        squared weights, as in the TMFG (option of the original
        implementation; needs ``max_clique_size >= 4``).

    Notes
    -----
    A vertex starts a new component when no attachment is accepted. With
    acceptance="threshold" this means a best total gain below `threshold`
    (weight gains) or below ``-1/2 log(1 - threshold)`` ("loglik", the gain of
    a single edge at the threshold).

    Notes
    -----
    The algorithm maintains:
      - `_cliques`: list of current maximal cliques (frozensets),
      - `_separators_count`: multiplicity of recorded separators,
      - `_peo`: perfect elimination order,
      - `_gains_pq`: min-heap on `(-gain, node, SeparatorWrapper)`.
    """

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------
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
        n_samples: Optional[int] = None,
        new_component: str = "strongest",
        n_starts: Optional[int] = None,
        drop_sep: bool = False,
        seed: str = "strongest",
    ) -> None:
        self._gain_unset = gain_function_type is None    # FutureWarning at the first run
        if gain_function_type is None:
            gain_function_type = "loglik"
        if seed not in ("strongest", "tmfg"):
            raise ValueError("seed must be 'strongest' or 'tmfg'")
        def _is_int(x):
            return not isinstance(x, bool) and isinstance(x, (int, np.integer, float, np.floating)) \
                and np.isfinite(x) and float(x) == int(x)
        if not (_is_int(max_clique_size) and max_clique_size >= 1):
            raise ValueError("max_clique_size must be an integer >= 1")
        if not (_is_int(min_clique_size) and 0 <= min_clique_size <= max_clique_size):
            raise ValueError("min_clique_size must be an integer between 0 and max_clique_size")
        if not (coordination_number == np.inf or (_is_int(coordination_number) and coordination_number >= 1)):
            raise ValueError("coordination_number must be an integer >= 1 or np.inf")
        if not (np.isfinite(threshold) and threshold >= 0):
            raise ValueError("threshold must be finite and >= 0")
        if gain_function_type == "loglik" and threshold > 1:
            raise ValueError("with the 'loglik' gain, threshold is a squared partial correlation in [0, 1]")
        max_clique_size, min_clique_size = int(max_clique_size), int(min_clique_size)
        if seed == "tmfg" and max_clique_size < 4:
            raise ValueError("seed='tmfg' needs max_clique_size >= 4")
        if seed == "tmfg" and min_clique_size > 5:
            raise ValueError("seed='tmfg' (four vertices) needs min_clique_size <= 5")
        self._seed = seed
        self._drop_sep = bool(drop_sep)
        _check_options(gain_function_type, acceptance, alpha, n_samples, require_n=False)
        if new_component not in ("strongest", "first"):
            raise ValueError("new_component must be 'strongest' or 'first'")
        self._new_component = new_component
        self._n_starts_auto = n_starts is None
        if n_starts is None:          # several starts pay off with the likelihood objective
            n_starts = int(min(max_clique_size, 1e9)) if gain_function_type == "loglik" else 1
        if (isinstance(n_starts, bool) or not isinstance(n_starts, (int, np.integer))
                or not 1 <= n_starts <= max_clique_size):
            raise ValueError("n_starts must be None or an integer between 1 and max_clique_size")
        self._n_starts = int(n_starts)
        self._seed_override = None
        self._use_fast_path = True             # testing switch for the direct path
        self._threshold = threshold
        self._min_clique_size = min_clique_size
        self._max_clique_size = max_clique_size
        self._coordination_number = coordination_number
        self._gf_type = gain_function_type
        self._acceptance = acceptance
        self._alpha = alpha
        self._n_samples = n_samples

    def fast_mfcf(self, C: np.ndarray, ctl: dict, gf_type: str = "sumsquares",
                  cov_matrix: Optional[np.ndarray] = None):
        """Interface of the original implementation (deprecated):
        ``MFCF().fast_mfcf(C, mfcf_control(), "sumsquares")``. The control
        dictionary is mapped onto the keyword options (``method="TMFG"`` to
        ``seed="tmfg"``) and the run uses the original rule for new
        components and a single start. Returns
        ``cliques, separators_count, peo, J``."""
        warnings.warn("MFCF().fast_mfcf(C, ctl, gf_type) is deprecated; use "
                      "MFCF(**options).run(C) or .fit(X)", DeprecationWarning, stacklevel=2)
        method = ctl.get("method", "MFCF")
        if method not in ("MFCF", "TMFG"):
            raise ValueError(f"Unknown method: {method}")
        m = MFCF(threshold=ctl.get("threshold", 0.0),
                 min_clique_size=ctl.get("min_clique_size", 1),
                 max_clique_size=ctl.get("max_clique_size", 4),
                 coordination_number=ctl.get("max_separator_multiplicity",
                                             ctl.get("coordination_number", np.inf)),
                 gain_function_type=gf_type, drop_sep=ctl.get("drop_sep", False),
                 seed="tmfg" if method == "TMFG" else "strongest",
                 new_component="first", n_starts=1)
        m._legacy_seed = True
        out = m.run(C, cov_matrix=cov_matrix)
        # expose the results, not the configuration, on the calling instance
        self._C, self._cliques, self._separators_count, self._peo = m._C, m._cliques, m._separators_count, m._peo
        return out

    def run(
        self,
        C: np.ndarray,
        cov_matrix: Optional[np.ndarray] = None,
    ) -> Tuple[List[Clique], Counter, List[Node], np.ndarray]:
        """
        Execute the MFCF process.

        Parameters
        ----------
        C : np.ndarray, shape (N, N)
            Weight matrix used for scoring/gains (e.g., correlation).
        cov_matrix : np.ndarray, optional, shape (N, N)
            If provided, used to compute the final logo (sparse inverse
            estimator). If omitted, `C` is used for that step.

        Returns
        -------
        cliques : list of frozenset[int]
            The maximal cliques obtained.
        separators_count : collections.Counter
            Multiplicity counts of recorded separators.
        peo : list[int]
            Perfect elimination order in which vertices were added.
        J_logo : np.ndarray, shape (N, N)
            Sparse inverse estimator: sum of clique inverses minus multiplicity-
            weighted separator inverses.

        Notes
        -----
        - This method mutates internal state; create a new `MFCF` instance if
          you need multiple independent runs in parallel.
        - Logging at INFO/DEBUG provides a step-by-step trace.
        """
        if self._gain_unset:
            warnings.warn(_DEFAULT_CHANGE_MSG, FutureWarning, stacklevel=2)
            self._gain_unset = False
        C = np.asarray(C, dtype=float)
        if C.ndim != 2 or C.shape[0] != C.shape[1] or C.shape[0] == 0:
            raise ValueError("C must be a non-empty square matrix")
        if cov_matrix is not None:
            cov_matrix = np.asarray(cov_matrix, dtype=float)
            if cov_matrix.shape != C.shape:
                raise ValueError("cov_matrix must have the shape of C")
        self._C = C
        self._Cs = None
        best = None
        seeds = self._start_seeds(min(self._n_starts, C.shape[0]))
        for k, seed in enumerate(seeds):
            self._seed_override = seed
            try:
                if self._fast_path_applies():
                    try:
                        self._build_fast()
                    except ValueError:
                        raise
                    except Exception as e:      # e.g. an unusable Numba cache
                        warnings.warn(f"Numba direct path unavailable ({type(e).__name__}: {e}); "
                                      "using the general path", RuntimeWarning, stacklevel=2)
                        self._use_fast_path = False
                        self._build()
                else:
                    self._build()
            finally:
                self._seed_override = None
            if len(seeds) == 1:
                break
            score = self._objective()
            if best is None or score > best[0] + 1e-12:
                best = (score, k, self._cliques, self._separators_count, self._peo)
        self.start_ = 0
        if best is not None:
            _, self.start_, self._cliques, self._separators_count, self._peo = best

        matrix_for_logo = np.asarray(cov_matrix if cov_matrix is not None else C, dtype=float)
        if not self._blocks_invertible(matrix_for_logo):
            if cov_matrix is not None or self._gf_type != "sum":
                warnings.warn("No LoGo precision: some clique blocks are (numerically) singular, "
                              "e.g. fewer observations than the clique size; use shrinkage > 0 "
                              "(fit) or pass a positive definite cov_matrix (run)", RuntimeWarning,
                              stacklevel=2)
            J_logo = None
        else:
            try:
                J_logo = self._logo(matrix_for_logo, self._cliques, self._separators_count)
            except LA.LinAlgError:
                J_logo = None
        return self._cliques, self._separators_count, self._peo, J_logo

    def _blocks_invertible(self, M: np.ndarray, rcond: float = 1e-12) -> bool:
        """True if every clique block of M is invertible to working precision,
        judged on the correlation-scaled block so that the units of the
        variables do not matter (separator blocks are principal sub-blocks of
        clique blocks)."""
        for q in self._cliques:
            idx = sorted(q)
            if len(idx) < 2:
                if not M[idx[0], idx[0]] > 0:
                    return False
                continue
            B = M[np.ix_(idx, idx)]
            d = np.abs(np.diag(B))
            if not np.all(d > 0):
                return False
            B = B / np.sqrt(np.outer(d, d))         # scale-free: correlation-scaled block
            ev = np.abs(np.linalg.eigvalsh(B))
            if not np.all(np.isfinite(ev)) or ev.min() <= rcond * ev.max():
                return False
        return True

    def _start_seeds(self, k: int) -> List[Optional[Clique]]:
        """Seeds of the k starts: the default seed, then vertices spread over
        the network, each strong but weakly related to the seeds already
        chosen (largest ``s_v (1 - max_u W_vu)`` with ``s`` the strength and
        ``W`` the squared correlations, or the weights for "sum"); a seed
        with more than one vertex adds its strongest neighbours."""
        if k == 1:
            return [None]
        m = max(1, self._min_clique_size - 1)
        C = np.asarray(self._C, dtype=float)
        W = np.abs(C) if self._gf_type == "sum" else C * C
        W = W - np.diag(np.diag(W))
        if self._gf_type == "sum" and W.max() > 0:
            W = W / W.max()
        elif self._gf_type != "sum":
            d = np.sqrt(np.diag(C))
            W = W / np.outer(d, d) ** 2                 # squared correlations
        strength = W.sum(axis=1)
        first = self._get_first_clique()
        chosen = sorted(first)
        seeds = [None]
        for _ in range(k - 1):
            score = strength * (1 - W[:, chosen].max(axis=1))
            score[chosen] = -np.inf
            v = int(np.argmax(score))
            nb = [int(u) for u in np.argsort(-W[v], kind="stable") if u != v][: m - 1]
            seeds.append(frozenset([v] + nb))
            chosen.append(v)
        return seeds

    def _objective(self) -> float:
        """Objective of the current clique forest, used to compare starts
        (for fit with shrinkage: the likelihood of the shrunk model)."""
        C = np.asarray(getattr(self, "_objective_matrix", None)
                       if getattr(self, "_objective_matrix", None) is not None else self._C, dtype=float)
        if self._gf_type == "loglik":
            def ld(q):
                q = sorted(q)
                sgn, val = np.linalg.slogdet(C[np.ix_(q, q)]) if q else (1.0, 0.0)
                return val if sgn > 0 else -np.inf
            with np.errstate(invalid="ignore"):
                val = -0.5 * (sum(ld(q) for q in self._cliques)
                              - sum(m * ld(sp) for sp, m in self._separators_count.items()))
            return float(val) if np.isfinite(val) else -np.inf   # singular blocks rank last
        W = np.abs(C) if self._gf_type == "sum" else C * C
        E = np.zeros(C.shape, dtype=bool)
        for q in self._cliques:
            q = sorted(q)
            E[np.ix_(q, q)] = True
        np.fill_diagonal(E, False)
        return float(W[E].sum() / 2)

    def _fast_path_applies(self) -> bool:
        """Direct path: weight gains, threshold 0, min_clique_size <= 1, Numba."""
        return (_HAS_NUMBA and self._use_fast_path and self._gf_type in ("sumsquares", "sum")
                and self._threshold == 0.0 and self._min_clique_size <= 1
                and not self._drop_sep and self._seed == "strongest")

    def _build_fast(self) -> None:
        """Direct algorithm for the weight gains with ``threshold == 0`` and
        ``min_clique_size <= 1`` (network generation), ported from the
        `mutual_info` branch: per-separator gain vectors with lazily sorted
        candidates and Numba kernels replace the priority queue. It gives
        the same clique forest as the general path."""
        C = np.asarray(self._C, dtype=np.float64)
        p = int(C.shape[0])
        if self._gf_type == "sumsquares":
            _check_correlation_like(C)
            W = np.asfortranarray(np.square(C))
        else:
            _check_nonnegative(C)
            W = np.asfortranarray(C)
        outstanding = np.ones(p, dtype=bool)
        n_outstanding = p

        # ---- Output state ---------------------------------------------------
        first_cl: Clique = self._get_first_clique()
        cliques: List[Clique] = [first_cl]
        peo: List[Node] = [v for v in first_cl]
        for v in first_cl:
            outstanding[v] = False
            n_outstanding -= 1

        separators_count: Counter = Counter()
        cliques_by_node: dict = {}
        for v in first_cl:
            cliques_by_node.setdefault(v, []).append(first_cl)

        # ---- Active separator state ----------------------------------------
        # We precompute, per separator, the *full* argsort over its
        # gain vector and track a monotonically advancing pointer
        # into it.  Because ``outstanding`` only ever shrinks, the
        # pointer is amortised O(1) per query: each sep's pointer
        # advances at most ``p`` times across the whole algorithm.
        # All scalar bookkeeping fields are kept as numpy arrays so
        # the global argmax in ``_find_best_pending`` is a single
        # vectorised pass.
        sep_to_id: dict = {}
        sep_list: List[FrozenSet[int]] = []

        cap = 1024
        gain_mat = np.zeros((cap, p), dtype=np.float64)
        # int32 is enough for p << 2**31; halves memory vs int64.
        argsort_mat = np.zeros((cap, p), dtype=np.int32)
        ptrs = np.zeros(cap, dtype=np.int32)
        best_gain_arr = np.full(cap, -np.inf, dtype=np.float64)
        best_node_arr = np.full(cap, -1, dtype=np.int64)
        pending_arr = np.zeros(cap, dtype=bool)
        # Per-sep flag: ``True`` once we've paid the argsort cost.
        # We start at ``False`` and seed the initial best with a
        # cheap O(p) argmax, deferring the O(p log p) argsort until
        # the first refresh — at which point we batch-sort a group
        # of stale seps in parallel for a much better constant.
        has_argsort = np.zeros(cap, dtype=bool)
        # Empty-sep rows are special-cased (gain == 0 everywhere),
        # so we mark them as "argsort done" with an ``arange`` row
        # in :func:`_add_or_get_sep`.
        is_empty_sep = np.zeros(cap, dtype=bool)

        max_c = int(self._max_clique_size)
        min_c = int(self._min_clique_size)
        threshold = float(self._threshold)
        coord_cap = self._coordination_number
        # Batch threshold for the lazy argsort path.  Set to ~32 so
        # we have enough work per parallel kernel launch to amortise
        # thread-spawn overhead, but not so high that the O(p)
        # fallback dominates.
        _ARGSORT_BATCH_THRESHOLD = 64

        # Local refs for speed
        _outstanding = outstanding
        _arange_p = np.arange(p, dtype=np.int32)

        def _grow(n_needed: int) -> None:
            nonlocal cap, gain_mat, argsort_mat, ptrs
            nonlocal best_gain_arr, best_node_arr, pending_arr
            nonlocal has_argsort, is_empty_sep
            if n_needed <= cap:
                return
            new_cap = cap
            while new_cap < n_needed:
                new_cap *= 2
            new_gain = np.zeros((new_cap, p), dtype=np.float64)
            new_gain[:cap] = gain_mat
            new_argsort = np.zeros((new_cap, p), dtype=np.int32)
            new_argsort[:cap] = argsort_mat
            new_ptrs = np.zeros(new_cap, dtype=np.int32)
            new_ptrs[:cap] = ptrs
            new_bg = np.full(new_cap, -np.inf, dtype=np.float64)
            new_bg[:cap] = best_gain_arr
            new_bn = np.full(new_cap, -1, dtype=np.int64)
            new_bn[:cap] = best_node_arr
            new_pd = np.zeros(new_cap, dtype=bool)
            new_pd[:cap] = pending_arr
            new_ha = np.zeros(new_cap, dtype=bool)
            new_ha[:cap] = has_argsort
            new_ie = np.zeros(new_cap, dtype=bool)
            new_ie[:cap] = is_empty_sep
            gain_mat = new_gain
            argsort_mat = new_argsort
            ptrs = new_ptrs
            best_gain_arr = new_bg
            best_node_arr = new_bn
            pending_arr = new_pd
            has_argsort = new_ha
            is_empty_sep = new_ie
            cap = new_cap

        def _ensure_best(sid: int) -> None:
            _ensure_best_lazy_kernel(
                sid, has_argsort, ptrs, argsort_mat, _outstanding,
                gain_mat, best_gain_arr, best_node_arr,
                bool(is_empty_sep[sid]),
            )

        def _add_or_get_sep(sep_fs: FrozenSet[int]) -> int:
            sid = sep_to_id.get(sep_fs)
            if sid is not None:
                return sid
            sid = len(sep_list)
            _grow(sid + 1)
            sep_list.append(sep_fs)
            sep_to_id[sep_fs] = sid
            if not sep_fs:
                # All-zero gain; argsort is ``arange(p)`` so the
                # empty-sep best is the first outstanding node.
                gain_mat[sid] = 0.0
                argsort_mat[sid] = _arange_p
                has_argsort[sid] = True
                is_empty_sep[sid] = True
            else:
                cols = np.fromiter(sep_fs, dtype=np.intp, count=len(sep_fs))
                if cols.size == 1:
                    gain_mat[sid] = W[:, cols[0]]
                else:
                    np.sum(W[:, cols], axis=1, out=gain_mat[sid])
                # Defer argsort — initial best comes from a cheap
                # O(p) argmax.  The full sort is only paid the
                # first time we have to advance past the initial
                # best, and we batch many such payments in
                # parallel inside ``_batch_argsort_descending``.
                has_argsort[sid] = False
                is_empty_sep[sid] = False
            ptrs[sid] = 0
            _initial_argmax(
                sid, _outstanding, gain_mat, best_gain_arr,
                best_node_arr, bool(is_empty_sep[sid]),
            )
            return sid

        def _push(sep_fs: FrozenSet[int]) -> None:
            # PATCH: a separator that has reached the coordination cap
            # can no longer host new vertices.
            if sep_fs and separators_count[sep_fs] >= coord_cap:
                return
            sid = sep_to_id.get(sep_fs)
            if sid is not None and pending_arr[sid]:
                return
            sid = _add_or_get_sep(sep_fs)
            bn = int(best_node_arr[sid])
            if bn >= 0 and not _outstanding[bn]:
                _ensure_best(sid)
            pending_arr[sid] = True

        def _push_facets(clq: Clique) -> None:
            # Mirror ``_process_new_clique_gains``: facets are always
            # built from ``frozenset(facet_tuple)`` where the tuple is
            # in sorted order. Matching this keeps the underlying
            # hash-table layout of the stored ``sep_fs`` identical to
            # the reference impl's first push, which downstream
            # ``np.fromiter(sep)`` then iterates in the same order.
            cs = len(clq)
            clique_t = tuple(sorted(clq))
            if cs < max_c:
                _push(frozenset(clique_t))
                return
            for facet in itertools.combinations(clique_t, cs - 1):
                _push(frozenset(facet))

        def _find_best_pending() -> int:
            # Lexicographic argmax over ``(best_gain, -best_node)``
            # restricted to pending entries.  Implemented as a Numba
            # kernel because the per-call numpy overhead dominates
            # when ``len(sep_list) < ~5000``.
            n = len(sep_list)
            if n == 0:
                return -1
            return int(_argmax_pending(
                n, pending_arr, best_gain_arr, best_node_arr,
            ))

        # Seed PQ with facets/whole-clique of the initial seed.
        _push_facets(first_cl)

        # ---- Main loop ------------------------------------------------------
        while n_outstanding > 0:
            sid = _find_best_pending()
            if sid < 0:
                # No active separator with a valid best — fall back to
                # the reference "force new clique" path.
                sep_fs = cliques[-1]
                v = self._choose_root(_outstanding)
                gain = 0.0
                prior_sep = sep_fs
            else:
                sep_fs = sep_list[sid]
                v = int(best_node_arr[sid])
                gain = float(best_gain_arr[sid])
                prior_sep = sep_fs
                pending_arr[sid] = False

            # Apply threshold. The parent clique is only consulted
            # for logging in the reference path, so we skip looking
            # it up here.
            #
            # Note on float-exact equivalence: the reference
            # ``Gains._separator_for_row`` returns
            # ``kept = frozenset(cols[keep_mask])`` where
            # ``cols = np.fromiter(sep, dtype=np.intp)``.  In the
            # simple regime ``keep_mask`` is all-True, so the kept
            # set is just ``frozenset(cols)`` — equal to ``sep`` as a
            # set but with a different internal hash-table layout
            # (and ``np.int64`` element type).  That layout flows
            # through ``new_clique = frozenset(sep | {v})`` and
            # through ``separators_count[sep]``, where the *first*
            # frozenset inserted fixes the iteration order used by
            # ``_logo``'s ``tuple(clq)`` indexing.  Mirroring this
            # rebuild keeps every downstream ``inv(C[idx, idx])``
            # batched LU pivoting bit-identical.
            if gain < threshold or not sep_fs:
                v = self._choose_root(_outstanding)
                sep_used: FrozenSet[int] = frozenset()
            else:
                cols_arr = np.fromiter(sep_fs, dtype=np.intp, count=len(sep_fs))
                sep_used = frozenset(cols_arr)

            # ---- Add new clique --------------------------------------------
            new_clique: Clique = frozenset(sep_used | {v})
            peo.append(v)
            _outstanding[v] = False
            n_outstanding -= 1

            cliques_before = list(cliques)

            if len(new_clique) > 1:
                to_remove = [c for c in cliques if c < new_clique]
                for c in to_remove:
                    cliques.remove(c)
                    for w in c:
                        b = cliques_by_node.get(w)
                        if b is not None:
                            try:
                                b.remove(c)
                            except ValueError:
                                pass
            cliques.append(new_clique)
            for w in new_clique:
                cliques_by_node.setdefault(w, []).append(new_clique)

            # ---- Eager refresh of seps whose best_node was v ---------------
            # Strategy: keep argsort lazy and batch it across many
            # node-removal events.  Whenever the count of pushed-
            # but-not-yet-sorted non-empty seps crosses a threshold,
            # we run one big parallel argsort batch.  Until then,
            # refresh uses the inline O(p) argmax fallback inside
            # the numba kernel.  The threshold trades parallel
            # amortisation against the per-refresh argmax cost.
            n_seps = len(sep_list)
            if n_seps:
                unsorted_count_total = n_seps - int(has_argsort[:n_seps].sum())
                if unsorted_count_total >= _ARGSORT_BATCH_THRESHOLD:
                    sort_ids = np.flatnonzero(~has_argsort[:n_seps]).astype(np.int64)
                    _batch_argsort_descending(
                        gain_mat, argsort_mat, sort_ids,
                    )
                    has_argsort[sort_ids] = True
                    ptrs[sort_ids] = 0
                _refresh_seps_for_node(
                    v, n_seps, pending_arr, best_node_arr,
                    has_argsort, ptrs, argsort_mat,
                    _outstanding, gain_mat, best_gain_arr,
                )

            # ---- _check_proposed_separator equivalent ----------------------
            # The reference records ``sep_wrapper.separator`` (the
            # kept / re-hashed sep) into ``separators_count``, while
            # re-pushing under ``sep_wrapper.separator_prior_threshold``
            # (the original / sorted-insertion sep).  Mirror both.
            sep_for_check = sep_used if (gain >= threshold and sep_fs) else prior_sep
            if sep_for_check and (min_c - 1) <= len(sep_for_check) < max_c:
                under_cap = separators_count[sep_for_check] < coord_cap
                if under_cap:
                    not_superset = True
                    for clq_before in cliques_before:
                        if sep_for_check >= clq_before:
                            not_superset = False
                            break
                    if not_superset:
                        separators_count[sep_for_check] += 1
                    if n_outstanding != 0 and separators_count[sep_for_check] < coord_cap:
                        _push(prior_sep)

            if n_outstanding == 0:
                break

            # Re-push prior_sep only while it is below the coordination
            # cap (PATCH: enforce coordination_number on reuse, so that the
            # recorded multiplicities equal the actual number of attachments).
            if not (sep_for_check and separators_count[sep_for_check] >= coord_cap):
                _push(prior_sep)
            _push_facets(new_clique)

        self._cliques = cliques
        self._separators_count = separators_count
        self._peo = peo

    def _build(self) -> None:
        """One run of the greedy expansion from the current seed."""
        C = self._C
        self._gains = Gains(C, self._threshold, self._min_clique_size, self._gf_type,
                            self._acceptance, self._alpha, self._n_samples,
                            root_chooser=self._choose_root)
        self._gf = self._gains.get_best_gain
        # total-gain threshold for starting a new component, on the gain's scale
        if self._gf_type != "loglik":
            self._component_threshold = self._threshold
        elif self._acceptance == "threshold":
            self._component_threshold = -0.5 * np.log1p(-min(self._threshold, 1 - 1e-12))
        else:
            # statistical acceptance: as with the threshold rule, the total gain
            # must reach that of one accepted step (the critical value of the
            # first test), so mandatory picks alone do not attach a vertex
            c0 = self._gains._critical_pc2(0)
            self._component_threshold = (np.inf if not np.isfinite(c0)
                                         else max(1e-15, -0.5 * np.log1p(-min(c0, 1 - 1e-12))))

        self._initialise()
        self._compute_mfcf()

    def fit(
        self,
        X: np.ndarray,
        *,
        assume_centered: bool = False,
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
        mi_matrix: Optional[np.ndarray] = None,
    ) -> "MFCF":
        """
        Build the clique forest and a precision estimate directly from data.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Observations in rows (e.g. a multivariate time series of returns).
        assume_centered : bool, default=False
            If False, the sample mean is removed.
        shrinkage : float in [0, 1), default=0.0
            Identity shrinkage of the correlation blocks used for the precision
            estimate, ``(1 - shrinkage) R + shrinkage I``; recommended when
            the number of observations is not much larger than the clique size.
        n_effective : None, float or "auto", default=None
            Number of observations used by the statistical acceptance rules.
            None uses ``n_samples``. For autocorrelated series, "auto" estimates
            it from the lag-1 autocorrelations ``phi`` of the series with
            Bartlett's inflation factor for sample correlations,
            ``n / mean_{i<j} (1 + phi_i phi_j) / (1 - phi_i phi_j)``.
        correlation : {"pearson", "spearman", "kendall"}, default="pearson"
            Correlation used for the structure and the precision. The rank
            options estimate the correlation of the Gaussian copula,
            ``2 sin(pi rho_s / 6)`` (Spearman) or ``sin(pi tau / 2)``
            (Kendall), projected onto the positive definite matrices if
            needed; they are invariant to monotone transformations of the
            variables and robust to heavy tails and outliers. The statistical
            acceptance rules then hold approximately. "spearman" costs about
            as much as "pearson" and is recommended for long series; "kendall"
            costs O(p^2 n log n) (a few seconds for p = 50, n = 20000).
        similarity : {"correlation", "mutual_information"}, default="correlation"
            Dependence measure on which the structure is built. "correlation"
            uses `correlation`. "mutual_information" uses the pairwise mutual
            information in Linfoot's normalisation ``sqrt(1 - exp(-2 I))``
            (see `mutual_information.mutual_information_matrix`), which
            captures non-linear and tail dependence; it requires the
            "sumsquares" or "sum" gain ("sumsquares" if the gain is unset). The precision is assembled from the
            correlation blocks in either case.
        mi_estimator, mi_n_neighbors, mi_n_bins, mi_normalize, mi_n_jobs, mi_random_state
            Options of the mutual-information estimator ("ksg" or "histogram").
        mi_matrix : np.ndarray, optional
            Precomputed mutual-information similarity (skips the estimation).

        Returns
        -------
        self
            With attributes ``cliques_``, ``separators_`` (Counter), ``peo_``,
            ``precision_`` and ``covariance_`` (covariance scale of `X`),
            ``correlation_``, ``location_``, ``n_samples_``, ``n_effective_``.

        Notes
        -----
        The structure is built from the sample correlation with the chosen
        gain ("loglik", or "sumsquares" for the original sum-of-squared-
        correlations construction; "sum" requires non-negative correlations),
        so it does not depend on the scales of the variables. The precision is the LoGo
        assembly of the (shrunk) correlation blocks, rescaled by the sample
        standard deviations: ``precision_ = D^{-1} J_R D^{-1}``. With
        ``shrinkage=0`` and clique blocks of full rank it is the maximum
        likelihood estimate for the decomposable model.
        """
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[0] < 3:
            raise ValueError("X must be an (n_samples, n_features) array with n_samples >= 3")
        if not np.all(np.isfinite(X)):
            raise ValueError("X contains NaN or infinite values; handle missing data before fit(X)")
        if not 0 <= shrinkage < 1:
            raise ValueError("shrinkage must be in [0, 1)")
        n, p = X.shape
        loc = np.zeros(p) if assume_centered else X.mean(axis=0)
        Xc = X - loc
        cov = Xc.T @ Xc / n
        sd = np.sqrt(np.diag(cov))
        if np.any(sd <= 0):
            raise ValueError("X has constant columns")
        if correlation == "pearson":
            R = cov / np.outer(sd, sd)
            R = (R + R.T) / 2
            np.fill_diagonal(R, 1.0)
        elif correlation in ("spearman", "kendall"):
            R = _rank_correlation(X, correlation)
        else:
            raise ValueError("correlation must be 'pearson', 'spearman' or 'kendall'")

        if n_effective is None:
            n_eff = float(n)
        elif isinstance(n_effective, str) and n_effective == "auto":
            n_eff = _effective_sample_size(Xc)
        else:
            n_eff = float(n_effective)
            if not 3 <= n_eff <= n:
                raise ValueError("n_effective must lie between 3 and n_samples")

        restore = None
        if similarity == "correlation":
            S = R
        elif similarity == "mutual_information":
            if self._gain_unset:      # as in the mutual_info branch: sum of squares
                restore = (self._gf_type, self._gain_unset, self._n_starts)
                self._gf_type, self._gain_unset = "sumsquares", False
                if self._n_starts_auto:
                    self._n_starts = 1
            if self._gf_type == "loglik":
                raise ValueError(
                    "similarity='mutual_information' needs gain_function_type='sumsquares' "
                    "or 'sum' (the information matrix is not a covariance)")
            if self._gf_type == "sumsquares" and mi_normalize != "linfoot":
                raise ValueError("raw mutual information (mi_normalize='none') is not a correlation "
                                 "coefficient: use gain_function_type='sum' or mi_normalize='linfoot'")
            if mi_matrix is not None:
                S = np.asarray(mi_matrix, dtype=float)
                if S.shape != (p, p):
                    raise ValueError("mi_matrix must have shape (n_features, n_features)")
            else:
                from mutual_information import mutual_information_matrix
                S = mutual_information_matrix(
                    X, estimator=mi_estimator, n_neighbors=mi_n_neighbors, n_bins=mi_n_bins,
                    normalize=mi_normalize, n_jobs=mi_n_jobs, random_state=mi_random_state)
        else:
            raise ValueError("similarity must be 'correlation' or 'mutual_information'")

        saved = self._n_samples
        if self._acceptance != "threshold":
            # the F-test spends one degree of freedom on the estimated mean
            self._n_samples = n_eff + (1.0 if assume_centered and self._acceptance == "ftest" else 0.0)
        try:
            R_shrunk = (1 - shrinkage) * R + shrinkage * np.eye(p)
            if self._gf_type == "loglik" and shrinkage > 0:
                self._objective_matrix = R_shrunk      # starts compared on the shrunk model
            cliques, seps, peo, J = self.run(C=S, cov_matrix=R_shrunk)
        finally:
            self._objective_matrix = None
            self._n_samples = saved
            if restore is not None:   # the switch applies to this fit only
                self._gf_type, self._gain_unset, self._n_starts = restore
        self.similarity_ = S

        self.cliques_, self.separators_, self.peo_ = cliques, seps, peo
        self.correlation_ = R
        self.location_ = loc
        self.n_samples_ = n
        self.n_effective_ = n_eff
        if J is None:
            self.precision_ = self.covariance_ = None
        else:
            self.precision_ = J / np.outer(sd, sd)
            Sigma = LA.inv(J) * np.outer(sd, sd)
            self.covariance_ = (Sigma + Sigma.T) / 2
        return self

    # -------------------------------------------------------------------------
    # Initialisation
    # -------------------------------------------------------------------------
    def _initialise(self) -> None:
        """
        Prepare data structures and seed the first clique.

        Side Effects
        ------------
        - Initializes the PQ, cliques, separator counts, PEO, and outstanding mask.
        - Seeds `_cliques` with `_get_first_clique()` and pushes its facets to PQ.
        """
        self._gains_pq: List[Tuple[float, Node, SeparatorWrapper]] = []
        self._iteration = 0

        first_cl = self._get_first_clique()

        self._cliques: List[Clique] = [first_cl]
        self._remaining_nodes_count = self._C.shape[0] - len(first_cl)

        self._separators_count: Counter = Counter()
        self._pq_separators: set[Separator] = set()
        self._seen_separators: set[Separator] = set()
        self._facet_excluded: dict = {}                # facet -> vertices it can no longer host

        self._peo: List[Node] = [v for v in first_cl]  # Perfect elimination order
        self._outstanding_nodes_mask = np.ones(self._C.shape[0], dtype=bool)
        self._outstanding_nodes_mask[list(first_cl)] = False

        self._log_initial_state(first_cl)

        self._process_new_clique_gains(first_cl)

    def _get_first_clique(self, first: int = 1) -> Clique:
        """
        Seed with node(s) having the largest sum of above-mean incident weights.

        Parameters
        ----------
        first : int, unused
            Present for backward compatibility; ignored.

        Returns
        -------
        Clique
            Initial clique of size `max(0, min_clique_size - 1)`.
        """
        if self._seed_override is not None:
            return self._seed_override
        if getattr(self, "_legacy_seed", False) and self._seed == "strongest":
            # original implementation: a single seed vertex for any min_clique_size
            C1 = self._C.copy()
            C1[self._C <= self._C.mean()] = 0
            return frozenset([int(np.argsort(-C1.sum(axis=0), kind="stable")[0])])
        if self._seed == "tmfg":
            W = np.square(np.asarray(self._C, dtype=float))
            v = np.sum(W * (W > W.mean()), axis=1)
            return frozenset(int(x) for x in np.argsort(-v, kind="stable")[:4])
        Cs = self._scale_free()
        C1 = Cs.copy()
        C1[Cs <= Cs.mean()] = 0
        sums = C1.sum(axis=0)
        W = Cs * Cs if self._gf_type != "sum" else np.abs(Cs)
        strength = W.sum(axis=0) - np.diag(W)
        cand = np.lexsort((-strength, -sums))        # ties broken by total strength
        # seed with at least the strongest vertex, so the output does not
        # depend on the order of the variables
        return frozenset(int(x) for x in cand[: max(1, self._min_clique_size - 1)])

    def _scale_free(self) -> np.ndarray:
        """The input on the scale used by the seed and root rules: the
        correlation matrix for "loglik" (whose gain is scale-invariant, so a
        covariance and its correlation give the same forest), else as given."""
        Cs = getattr(self, "_Cs", None)
        if Cs is None or Cs.shape != np.shape(self._C):
            Cs = np.asarray(self._C, dtype=float)
            if self._gf_type == "loglik":
                d = np.sqrt(np.diag(Cs))
                Cs = Cs / np.outer(d, d)
            self._Cs = Cs
        return Cs

    # -------------------------------------------------------------------------
    # Main algorithm loop
    # -------------------------------------------------------------------------
    def _compute_mfcf(self) -> None:
        """
        Greedy loop: pop best (gain, node, separator), validate, then attach.

        The loop:
        1) Pops from PQ if available; otherwise forces a new component using the
           last clique as the separator (or empty).
        2) Applies threshold logic to decide whether to start a new component.
        3) Adds the new clique, updates PEO/outstanding, and records separators.
        4) Updates PQ with newly available separators/facets.
        """
        while self._remaining_nodes_count > 0:
            self._iteration += 1
            if self._gains_pq:
                gain, v, sep_wrapper = self._pop_from_pq()
                sep = sep_wrapper.separator
                if self._should_skip_candidate(gain, v, sep_wrapper):
                    continue
            else:
                # No candidates left; force a new clique
                sep = self._cliques[-1]
                gain, v, sep_wrapper = (
                    0.0,
                    self._choose_root(self._outstanding_nodes_mask),
                    SeparatorWrapper(sep, sep),
                )

            v, sep, parent_clique = self._apply_threshold_and_find_parent(gain, v, sep)
            cliques_before = list(self._cliques)
            new_clique = self._add_new_clique(parent_clique, sep, v)

            self._remaining_nodes_count -= 1
            if sep:  # a new component (empty separator) records no separator
                self._check_proposed_separator(sep_wrapper, cliques_before)

            if self._remaining_nodes_count == 0:
                break

            self._update_pq_for_new_separator(sep_wrapper.separator_prior_threshold)
            self._process_new_clique_gains(new_clique)

    # -------------------------------------------------------------------------
    # Candidate checks & parent search
    # -------------------------------------------------------------------------
    def _should_skip_candidate(
        self, gain: float, v: Node, sep_wrapper: SeparatorWrapper
    ) -> bool:
        """
        Validate a popped PQ candidate against constraints and state.

        Skips a candidate if:
        - The separator has exceeded multiplicity cap.
        - `gain` is NaN.
        - The node is no longer outstanding (and triggers a recompute for the sep).
        - The separator length violates clique size bounds.
        - The separator is not a subset of any current clique.

        Returns
        -------
        bool
            True if the candidate should be skipped.
        """
        # If drop_sep is enabled, disable candidates with a seen/used separator.
        sep = sep_wrapper.separator
        # multiplicity constraint (and drop_sep) on the separator actually used
        if (self._separators_count[sep] >= self._coordination_number
                or (self._drop_sep and (sep in self._seen_separators or self._separators_count[sep] > 0))):
            facet = sep_wrapper.separator_prior_threshold
            if facet != sep:
                # only this vertex's kept subset of the facet is blocked (or the
                # candidate is stale): the kept subset depends on (v, facet)
                # alone and caps only grow, so exclude v from the facet for
                # good and requeue the facet for its other vertices
                if self._outstanding_nodes_mask[v]:
                    excl = self._facet_excluded.setdefault(
                        facet, np.zeros(len(self._outstanding_nodes_mask), bool))
                    excl[v] = True
                self._update_pq_for_new_separator(facet)
            return True
        if np.isnan(gain):
            return True
        if not self._outstanding_nodes_mask[v]:
            # v already used, recompute best gain for this sep and push back to heap
            self._update_pq_for_new_separator(sep_wrapper.separator_prior_threshold)
            return True
        # length constraint
        if not (
            len(sep) >= self._min_clique_size - 1 and len(sep) < self._max_clique_size
        ):
            return True
        # subset-of-some-current-clique constraint
        if not any(sep.issubset(clq) for clq in self._cliques):
            return True
        return False

    def _choose_root(self, mask: np.ndarray) -> Node:
        """Vertex that starts a new component."""
        out = np.flatnonzero(mask)
        if self._new_component == "first" or len(out) <= 1:
            return int(out[0])
        W = self._scale_free()[np.ix_(out, out)]
        W = W * W if self._gf_type != "sum" else np.abs(W)
        strength = W.sum(axis=1) - np.diag(W)
        return int(out[np.argmax(strength)])        # ties: lowest index

    def _apply_threshold_and_find_parent(
        self, gain: float, v: Node, sep: Separator
    ) -> Tuple[Node, Separator, Optional[Clique]]:
        """
        Decide whether to start a new component or attach to a parent clique.

        Parameters
        ----------
        gain : float
            Negative of the PQ-stored value (heap stores `-gain`).
        v : int
            Candidate node.
        sep : Separator
            Candidate separator.

        Returns
        -------
        v : int
            (Possibly replaced) node to add.
        sep : Separator
            (Possibly empty) separator used for the new clique.
        parent_clique : Clique or None
            The clique containing `sep` when attaching; None if starting anew.

        Notes
        -----
        The PQ stores `-gain` for min-heap semantics. We compare `pos_gain` with
        `self._threshold` to decide if we start a new component.
        """
        pos_gain = -gain  # negate back to positive for threshold compare
        if pos_gain < self._component_threshold:
            # start a new clique
            v = self._choose_root(self._outstanding_nodes_mask)
            sep = frozenset()
            parent_clique = frozenset()
        else:
            parent_clique = self._find_parent_clique_for_separator(sep)
        return v, sep, parent_clique

    def _find_parent_clique_for_separator(self, sep: Separator) -> Optional[Clique]:
        """
        Find a current clique that contains `sep`.

        Returns
        -------
        Clique or None
            A clique `C` such that `sep ⊆ C`, if any; otherwise None.
        """
        for clq in self._cliques:
            if sep <= clq:
                return clq
        return None

    # -------------------------------------------------------------------------
    # Clique/separator updates
    # -------------------------------------------------------------------------
    def _add_new_clique(
        self, parent_clique: Optional[Clique], sep: Separator, v: Node
    ) -> Clique:
        """
        Add a new clique, keep only maximal cliques, and update state.

        Parameters
        ----------
        parent_clique : Clique or None
            Clique to which we attach via `sep`, if any.
        sep : Separator
            The (facet) separator used with node `v`.
        v : int
            Node to add.

        Returns
        -------
        Clique
            The new maximal clique.

        Side Effects
        ------------
        - Appends `v` to PEO, marks `v` as not outstanding.
        - Drops strict-subset cliques of the new one to maintain maximality.
        """
        new_clique: Clique = frozenset(sep | {v})
        self._peo.append(v)
        self._outstanding_nodes_mask[v] = False

        self._log_added_clique(v, new_clique, parent_clique, sep)

        # keep only maximal cliques (drop strict subsets of the new one)
        if len(new_clique) > 1:
            to_remove = [c for c in self._cliques if c < new_clique]
            for c in to_remove:
                self._cliques.remove(c)

        self._cliques.append(new_clique)
        return new_clique

    def _check_proposed_separator(
        self,
        separator_wrapper: SeparatorWrapper,
        cliques_before: List[Clique],
    ) -> None:
        """
        Consider a proposed separator and record it if it passes constraints.

        Conditions
        ----------
        - Non-empty.
        - Length in [min_clique_size - 1, max_clique_size).
        - Not a superset (or equal) of any existing clique at the time proposed.
        - Under multiplicity cap.

        Also enqueues it for potential reuse if nodes remain.
        """
        sep = separator_wrapper.separator
        if not sep:
            # Empty separator, nothing to record
            return
        if self._drop_sep:
            self._seen_separators.add(sep)

        if not (self._min_clique_size - 1 <= len(sep) < self._max_clique_size):
            return

        # Must NOT be a superset (or equal) of any prior clique
        not_superset_of_any_prior = not any(sep >= clique for clique in cliques_before)

        recorded = False
        under_multiplicity_cap = self._separators_count[sep] < self._coordination_number
        if not under_multiplicity_cap:
            self._log_processed_separator(sep, recorded)
            return

        if not_superset_of_any_prior:
            self._separators_count[sep] += 1
            recorded = True

        # We might reuse the same separator; keep PQ updated
        if self._remaining_nodes_count != 0:
            self._update_pq_for_new_separator(
                separator_wrapper.separator_prior_threshold
            )

        self._log_processed_separator(sep, recorded)

    def _process_new_clique_gains(self, clq: Clique) -> None:
        """
        Push gain candidates for all facets of `clq` vs. outstanding nodes.

        If `|clq| < max_clique_size`, the whole clique is considered a separator
        candidate; otherwise, all size-1 facets are pushed.
        """
        if not self._outstanding_nodes_mask.any():
            return                      # nothing left to attach (e.g. seed covers all nodes)
        clique = tuple(sorted(clq))
        clique_size = len(clq)

        facets = (
            [clique]
            if clique_size < self._max_clique_size
            else list(itertools.combinations(clique, clique_size - 1))
        )
        if clique_size >= self._max_clique_size:
            # exact gain: all facets from one regression on the clique
            self._gains.precompute_facets(clique, self._outstanding_nodes_mask)
        for facet in facets:
            self._update_pq_for_new_separator(frozenset(facet))

    # -------------------------------------------------------------------------
    # Priority queue management
    # -------------------------------------------------------------------------
    def _update_pq_for_new_separator(self, sep: Separator) -> None:
        """
        Recompute and push the best (gain, node) candidate for a separator.

        Avoids duplicates using `_pq_separators` keyed by `separator_prior_threshold`.
        """
        if sep in self._pq_separators:
            return
        if sep and self._separators_count[sep] >= self._coordination_number:
            return  # a separator at the coordination cap cannot host new vertices
        mask = self._outstanding_nodes_mask
        excl = self._facet_excluded.get(sep)
        if excl is not None:
            mask = mask & ~excl
            if not mask.any():
                return
        gain, v, ranked_sep = self._gf(mask, sep)
        self._push_to_pq(gain, v, ranked_sep)

    def _pop_from_pq(self):
        """
        Pop the best candidate from the PQ.

        Returns
        -------
        gain : float
            Stored as negative in the heap for min-heap semantics.
        v : int
            Candidate node.
        sep_wrapper : SeparatorWrapper
            Candidate separator wrapper (kept and prior-threshold variants).

        Side Effects
        ------------
        Removes the separator's `separator_prior_threshold` from `_pq_separators`.
        """
        gain, v, sep_wrapper = heapq.heappop(self._gains_pq)
        self._pq_separators.remove(sep_wrapper.separator_prior_threshold)
        return gain, v, sep_wrapper

    def _push_to_pq(self, gain: float, v: Node, sep_wrapper: SeparatorWrapper):
        """
        Push a candidate to the PQ and mark its prior-threshold separator as seen.
        """
        heapq.heappush(self._gains_pq, (-gain, v, sep_wrapper))
        self._pq_separators.add(sep_wrapper.separator_prior_threshold)

    # -------------------------------------------------------------------------
    # Logo computation
    # -------------------------------------------------------------------------
    def _logo(
        self, C: np.ndarray, cliques: List[Clique], separators: Counter
    ) -> np.ndarray:
        """
        Compute a sparse inverse estimator as cliques minus separators.

        For each clique `Q`, add `inv(C[Q,Q])`. For each separator `S` with
        multiplicity `m`, subtract `m * inv(C[S,S])`.

        Parameters
        ----------
        C : np.ndarray, shape (N, N)
            The matrix used for inversion blocks (usually covariance).
        cliques : list of Clique
            Maximal cliques to include.
        separators : collections.Counter
            Multiplicity counts of recorded separators.

        Returns
        -------
        J : np.ndarray, shape (N, N)
            Sparse inverse estimator.
        """
        J = np.zeros(C.shape)
        # For each clique, add the inverse of the submatrix defined by the clique indices.
        for clq in cliques:
            clqt = tuple(clq)
            J[np.ix_(clqt, clqt)] += LA.inv(C[np.ix_(clqt, clqt)])

        # For each separator, subtract the inverse of the submatrix defined by the separator indices.
        for sep, mult in separators.items():
            if sep:  # non-empty
                sept = tuple(sep)
                J[np.ix_(sept, sept)] -= mult * LA.inv(C[np.ix_(sept, sept)])

        return J

    # -------------------------------------------------------------------------
    # Debug logging
    # -------------------------------------------------------------------------
    def _log_initial_state(self, first_cl: Clique) -> None:
        """Log the initial seed clique and remaining node count."""
        logger.info("  Seed clique: %s", format_frozenset(first_cl))
        logger.info("  Selected based on gain function maximization")
        logger.info("  Remaining nodes: %d", self._remaining_nodes_count)
        logger.info("---")

    def _log_added_clique(
        self,
        v: Node,
        new_clique: Clique,
        parent_clique: Optional[Clique],
        sep: Separator,
    ) -> None:
        """Log details after adding a clique at the current iteration."""
        logger.info("Iteration %d", self._iteration)
        logger.info("  Added vertex: %s", v)
        logger.info("  Proposed sub-clique: %s", format_frozenset(sep))
        logger.info(
            "  Parent clique: %s",
            format_frozenset(parent_clique) if parent_clique else None,
        )
        logger.info("  New clique: %s", format_frozenset(new_clique))

    def _log_processed_separator(
        self, proposed_separator: Separator, separator_recorded: bool
    ) -> None:
        """
        Log whether a proposed separator was recorded or skipped, and why.
        """
        minc = self._min_clique_size
        maxc = self._max_clique_size
        if len(proposed_separator) == 0:
            logger.info("  → No separator recorded (empty proposed sub-clique)")
        elif not separator_recorded:
            if len(proposed_separator) >= maxc or len(proposed_separator) < (minc - 1):
                logger.info(
                    "  → Separator NOT recorded (size constraints: %d not in [%d, %d])",
                    len(proposed_separator),
                    minc - 1,
                    maxc - 1,
                )
            else:
                logger.info(
                    "  → Separator NOT recorded (proposed sub-clique equals existing clique - not proper)"
                )
        else:
            logger.info(
                "  → Separator RECORDED: %s", format_frozenset(proposed_separator)
            )
