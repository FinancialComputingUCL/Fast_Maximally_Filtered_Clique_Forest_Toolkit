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
            self._W = np.square(C)
        else:
            raise ValueError(f"Unknown gain function type: {gf_type}")

        self._threshold: float = threshold
        self._min_clique_size: int = min_clique_size

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
        self._C = C
        self._gf = Gains(
            C, self._threshold, self._min_clique_size, self._gf_type
        ).get_best_gain

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
        # seed with at least the strongest vertex, so the output does not
        # depend on the order of the variables
        return frozenset(int(x) for x in cand[: max(1, self._min_clique_size - 1)])

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
        # multiplicity constraint
        if self._separators_count[sep] >= self._coordination_number:
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
        if sep and self._separators_count[sep] >= self._coordination_number:
            return  # a separator at the coordination cap cannot host new vertices
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
