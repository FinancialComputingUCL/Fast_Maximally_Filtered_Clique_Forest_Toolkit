"""
Maximum Filtering Clique Forest (MFCF)

This module builds a sparse graphical structure (a forest of cliques) from a
weighted matrix `C` (e.g., correlation or covariance). It greedily grows
cliques by selecting the next vertex/separator pair that maximizes a gain
function (currently: sum of squared weights), while enforcing size and
multiplicity constraints on separators.

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
from collections import Counter
from dataclasses import dataclass
from typing import FrozenSet, Iterable, List, Optional, Tuple

import numpy as np
import numpy.linalg as LA

# -----------------------------------------------------------------------------
# Optional Numba accelerators for the fast-path inner loops.  The whole
# module still imports/runs without Numba — the fallback uses pure
# numpy, which is materially slower but correct.
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
# Gains
# =============================================================================
class Gains:
    """
    Gain function handler.

    Currently implements the "sumsquares" gain: for a candidate node `i` and
    separator `S`, the gain is the sum of squared weights `C[i, j]^2` over
    `j in S` that are above a threshold, with optional mandatory top-k picks to
    satisfy a minimum clique size.

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
    gf_type : {"sumsquares"}, default="sumsquares"
        Type of gain function. Only "sumsquares" is supported.

    Notes
    -----
    The handler precomputes `W = C ** 2` for efficient vectorized scoring.
    """

    def __init__(
        self,
        C: np.ndarray,
        threshold: float = 0.0,
        min_clique_size: int = 1,
        gf_type: str = "sumsquares",
    ):
        if gf_type == "sumsquares":
            # Fortran-order so the dominant op ``self._W[:, cols]`` reads
            # contiguous columns (cache-friendly at large p).
            self._W = np.asfortranarray(np.square(C))
        else:
            raise ValueError(f"Unknown gain function type: {gf_type}")

        self._threshold: float = threshold
        self._min_clique_size: int = min_clique_size
        # Cache for ``flatnonzero(outstanding_nodes_mask)``: the mask
        # only loses Trues between gain queries (one per attached node),
        # so the count of outstanding nodes uniquely keys a cache hit
        # within a single MFCF run.
        self._cached_outstanding_count: int = -1
        self._cached_rows: Optional[np.ndarray] = None

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
            first = int(outstanding_nodes_mask.argmax())
            empty = frozenset()
            return 0.0, first, SeparatorWrapper(empty, empty)

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

    # --------------------
    # Helpers
    # --------------------

    def invalidate_outstanding_cache(self) -> None:
        """Force the next gain query to recompute ``flatnonzero``."""
        self._cached_outstanding_count = -1

    def _prepare_submatrix(
        self,
        outstanding_nodes_mask: np.ndarray,
        sep: "Separator",
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Slice squared-weight matrix to rows of outstanding nodes and
        columns of ``sep``.

        Equivalent to ``self._W[np.ix_(rows, cols)]`` but (i) skips
        ``flatnonzero`` and the per-call ``mask.sum()`` whenever the
        outstanding set has not changed since the previous query, and
        (ii) reads the columns first against the Fortran-ordered
        ``self._W`` so each column is a contiguous strided memcpy.
        The MFCF caller is responsible for calling
        ``invalidate_outstanding_cache`` whenever a node is attached.
        """
        if self._cached_rows is None or self._cached_outstanding_count < 0:
            self._cached_rows = np.flatnonzero(outstanding_nodes_mask)
            self._cached_outstanding_count = self._cached_rows.size
        rows = self._cached_rows
        cols = np.fromiter(sep, dtype=np.intp, count=len(sep))
        W_sub = self._W[:, cols][rows]
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
    gain_function_type : {"sumsquares"}, default="sumsquares"
        Gain function type; currently only "sumsquares" is supported.

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
        gain_function_type: str = "sumsquares",
    ) -> None:

        self._threshold = threshold
        self._min_clique_size = min_clique_size
        self._max_clique_size = max_clique_size
        self._coordination_number = coordination_number
        self._gf_type = gain_function_type

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
        # Fast direct-selection path for the common simple configuration.
        # The kept_subset reduces to the full sep and mandatory-topk is empty,
        # so we can skip the per-call submatrix slicing entirely.
        if (
            self._gf_type == "sumsquares"
            and self._threshold == 0.0
            and self._min_clique_size <= 1
        ):
            return self._run_fast(C, cov_matrix)

        self._C = C
        self._gains = Gains(
            C, self._threshold, self._min_clique_size, self._gf_type
        )
        self._gf = self._gains.get_best_gain

        self._initialise()
        self._compute_mfcf()

        matrix_for_logo = cov_matrix if cov_matrix is not None else C
        J_logo = self._logo(matrix_for_logo, self._cliques, self._separators_count)
        return self._cliques, self._separators_count, self._peo, J_logo

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

        self._peo: List[Node] = [v for v in first_cl]  # Perfect elimination order
        self._outstanding_nodes_mask = np.ones(self._C.shape[0], dtype=bool)
        self._outstanding_nodes_mask[list(first_cl)] = False
        # Inverted index node -> cliques containing the node. Cuts the
        # ``any(sep.issubset(clq) for clq in self._cliques)`` scan in
        # ``_should_skip_candidate`` from O(#cliques) to
        # O(#cliques-containing-sep[0]).
        self._cliques_by_node: dict = {}
        for v in first_cl:
            self._cliques_by_node.setdefault(v, []).append(first_cl)
        # The mask cache held by the Gains handler must be flushed
        # whenever ``outstanding_nodes_mask`` changes.
        self._gains.invalidate_outstanding_cache()

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
        C1 = self._C.copy()
        r, c = np.nonzero(self._C <= self._C.mean())
        C1[r, c] = 0
        sums = C1.sum(axis=0)
        cand = np.argsort(-sums, kind="stable")
        return frozenset(cand[: (self._min_clique_size - 1)])

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
                    int(self._outstanding_nodes_mask.argmax()),
                    SeparatorWrapper(sep, sep),
                )

            v, sep, parent_clique = self._apply_threshold_and_find_parent(gain, v, sep)
            cliques_before = list(self._cliques)
            new_clique = self._add_new_clique(parent_clique, sep, v)

            self._remaining_nodes_count -= 1
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
        # multiplicity constraint
        if self._separators_count[sep] > self._coordination_number:
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
        # subset-of-some-current-clique constraint. The empty set is a
        # subset of every clique, so it always passes; for non-empty
        # ``sep`` we index by an arbitrary node so the scan only
        # touches cliques that actually contain that node.
        if not sep:
            return False
        anchor = next(iter(sep))
        candidates = self._cliques_by_node.get(anchor, ())
        if not any(sep <= clq for clq in candidates):
            return True
        return False

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
        if pos_gain < self._threshold:
            # start a new clique
            v = int(self._outstanding_nodes_mask.argmax())
            sep = frozenset()
            parent_clique = frozenset()
        else:
            parent_clique = self._find_parent_clique_for_separator(sep)
        return v, sep, parent_clique

    def _find_parent_clique_for_separator(self, sep: Separator) -> Optional[Clique]:
        """
        Find a current clique that contains ``sep``. Indexed by an
        arbitrary node of ``sep`` (same shortcut as
        ``_should_skip_candidate``).
        """
        if not sep:
            return None
        anchor = next(iter(sep))
        for clq in self._cliques_by_node.get(anchor, ()):
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
        # Mask changed -> cache held by Gains is stale.
        self._gains.invalidate_outstanding_cache()

        self._log_added_clique(v, new_clique, parent_clique, sep)

        # keep only maximal cliques (drop strict subsets of the new one)
        if len(new_clique) > 1:
            to_remove = [c for c in self._cliques if c < new_clique]
            for c in to_remove:
                self._cliques.remove(c)
                for w in c:
                    bucket = self._cliques_by_node.get(w)
                    if bucket is not None:
                        try:
                            bucket.remove(c)
                        except ValueError:
                            pass

        self._cliques.append(new_clique)
        for w in new_clique:
            self._cliques_by_node.setdefault(w, []).append(new_clique)
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
        clique = tuple(sorted(clq))
        clique_size = len(clq)

        facets = (
            [clique]
            if clique_size < self._max_clique_size
            else list(itertools.combinations(clique, clique_size - 1))
        )
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
        gain, v, ranked_sep = self._gf(self._outstanding_nodes_mask, sep)
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
    # Fast direct-selection path for the simple case
    # -------------------------------------------------------------------------
    def _run_fast(
        self,
        C: np.ndarray,
        cov_matrix: Optional[np.ndarray] = None,
    ) -> Tuple[List[Clique], Counter, List[Node], np.ndarray]:
        """
        Direct algorithm specialised for ``threshold == 0``,
        ``min_clique_size == 1`` and ``gf_type == "sumsquares"``.

        In this regime the gain reduces to a plain column-sum
        ``gain[i] = sum_{j in sep} W[i, j]`` (with ``W = C**2``), the
        kept-subset always equals the proposed separator, and every
        candidate passes the threshold check.  This lets us drop the
        heapq machinery (and its stale-on-pop O(p² × k) overhead) and
        instead:
          - cache one ``gain_array`` per unique separator,
          - maintain an "active set" of separators with their current
            best ``(gain, node)`` over the outstanding mask,
          - eagerly refresh only those separators whose ``best_node``
            was just consumed.
        Selection at each step is a single argmax over the active set.
        The cliques / separators_count / peo outputs are produced in
        the exact same order as the reference implementation, so the
        downstream ``_logo`` floats are bit-identical.
        """
        p = int(C.shape[0])
        # ``_get_first_clique`` expects ``self._C``; set it up.
        self._C = C
        # Fortran-order squared weight matrix; column reads are
        # contiguous memcpys.
        W = np.asfortranarray(np.square(C, dtype=np.float64))
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
                v = int(_outstanding.argmax())
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
                v = int(_outstanding.argmax())
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
                    if n_outstanding != 0:
                        _push(prior_sep)

            if n_outstanding == 0:
                break

            # Re-push prior_sep (matches the second update_pq call) and
            # push the new clique's facets.
            _push(prior_sep)
            _push_facets(new_clique)

        # ---- Logo --------------------------------------------------------
        matrix_for_logo = cov_matrix if cov_matrix is not None else C
        J_logo = self._logo(matrix_for_logo, cliques, separators_count)

        # Expose the internal state on ``self`` so downstream callers
        # that introspect (and to match the reference path) see them.
        self._C = C
        self._cliques = cliques
        self._separators_count = separators_count
        self._peo = peo

        return cliques, separators_count, peo, J_logo

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

        def _batched_signed_add(items_by_size, sign_scale_fn):
            # items_by_size: dict{size: list of (clique_or_sep_tuple, sign_scale)}
            for size, items in items_by_size.items():
                if not items:
                    continue
                if size == 1:
                    for tpl, mult in items:
                        i = tpl[0]
                        J[i, i] += mult * (1.0 / C[i, i])
                    continue
                idx = np.empty((len(items), size), dtype=np.intp)
                mults = np.empty(len(items), dtype=float)
                for i, (tpl, mult) in enumerate(items):
                    idx[i] = tpl
                    mults[i] = mult
                # Batched (B, size, size) inverse.
                sub = C[idx[:, :, None], idx[:, None, :]]
                inv_sub = LA.inv(sub)
                # Scale and scatter-add.
                inv_sub *= mults[:, None, None]
                for i, (tpl, _) in enumerate(items):
                    J[np.ix_(tpl, tpl)] += inv_sub[i]

        clique_groups: dict = {}
        for clq in cliques:
            tpl = tuple(clq)
            clique_groups.setdefault(len(tpl), []).append((tpl, 1.0))
        _batched_signed_add(clique_groups, sign_scale_fn=None)

        sep_groups: dict = {}
        for sep, mult in separators.items():
            if sep:
                tpl = tuple(sep)
                sep_groups.setdefault(len(tpl), []).append((tpl, -float(mult)))
        _batched_signed_add(sep_groups, sign_scale_fn=None)

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
