"""
Tools for performing Direct Bubble Hierarchical Tree (DBHT) clustering
using a Maximally Filtered Clique Forest (MFCF) backbone.

This module replaces the TMFG graph-construction step in the original
riskfolio DBHT implementation with an MFCF-based builder, improving
flexibility via tunable clique sizes, a gain threshold, and an optional
cap on separator reuse.

Functions
---------
mfcf_dbhts
    Run DBHT on a similarity matrix S (and the corresponding dissimilarity
    matrix D), returning cluster assignments, the filtered graph, and the
    resulting hierarchical linkage.

_convert_cliques_to_adjacency
    Helper to convert a list of cliques into a weighted adjacency matrix
    using weights from S.
"""

from typing import Optional

import numpy as np
from scipy.cluster.hierarchy import optimal_leaf_ordering
from scipy.spatial.distance import squareform
from riskfolio import (
    distance_wei,
    CliqHierarchyTree2s,
    BubbleCluster8s,
    HierarchyConstruct4s,
)

from fast_fast_mfcf import MFCF as _MFCFBuilder


def mfcf_dbhts(
    D: np.ndarray,
    S: np.ndarray,
    leaf_order: bool = True,
    threshold: float = 0.0,
    min_clique_size: int = 1,
    max_clique_size: int = 4,
    coordination_number: int = np.inf,
    gain_function_type: str = "sum",
    acceptance: str = "threshold",
    alpha: float = 0.05,
    new_component: str = "strongest",
    n_starts: Optional[int] = None,
    n_samples: int = None,
):
    """
    Perform Direct Bubble Hierarchical Tree (DBHT) clustering with an
    MFCF-generated backbone graph.

    This function is a drop-in alternative to the TMFG-based DBHT pipeline in
    riskfolio. It constructs a filtered graph using the Maximal Filtering
    Clique Forest (MFCF) algorithm and then applies the standard DBHT steps to
    produce deterministic, parameter-light hierarchical clusters.

    Parameters
    ----------
    D : ndarray of shape (N, N)
        Dissimilarity matrix (non-negative). Typical choices include pairwise
        distances such as ``squareform(pdist(X, metric='euclidean'))``.

    S : ndarray of shape (N, N)
        Similarity matrix (non-negative), consistent with ``D``. Examples:
        a correlation-based affinity, ``S = 1 / (1 + D / D.mean())``, or
        ``S = exp(-D)``. Only the relative magnitudes matter.

    leaf_order : bool, default=True
        If ``True``, apply optimal leaf ordering to the linkage to reduce
        crossings and improve dendrogram readability.

    threshold : float, default=0.0
        Gain threshold controlling whether a vertex attaches to an existing
        component versus spawning a new component during MFCF construction.

    min_clique_size : int, default=1
        Minimum clique size allowed in the MFCF graph.

    max_clique_size : int, default=4
        Maximum clique size allowed in the MFCF graph.

    coordination_number : float, default=np.inf
        Maximum allowed uses of a separator (multiplicity cap) during MFCF.

    Returns
    -------
    T8 : ndarray of shape (N,)
        Cluster labels obtained from DBHT.

    Rpm : ndarray of shape (N, N)
        Weighted adjacency of the filtered graph (edges weighted by ``S``; zero
        on the diagonal).

    Adjv : ndarray
        Bubble-cluster adjacency as produced by ``BubbleCluster8s``.

    Dpm : ndarray of shape (N, N)
        All-pairs shortest-path distances on the filtered graph.

    Mv : ndarray of shape (N, Nb)
        Bubble membership matrix where ``Mv[n, b] = 1`` if vertex ``n`` belongs
        to bubble ``b``.

    Z : ndarray of shape (N-1, 4)
        Linkage matrix encoding the DBHT hierarchy (compatible with
        ``scipy.cluster.hierarchy`` utilities).

    Raises
    ------
    ValueError
        If ``D`` or ``S`` is not square or the shapes are inconsistent.

    Notes
    -----
    - Deterministic given ``S`` and MFCF hyperparameters.
    - The result can be used with ``scipy.cluster.hierarchy.fcluster`` to obtain
      a flat clustering at a chosen number of clusters or distance threshold.
    """

    builder = _MFCFBuilder(
        threshold=threshold,
        min_clique_size=min_clique_size,
        max_clique_size=max_clique_size,
        coordination_number=coordination_number,
        gain_function_type=gain_function_type,
        acceptance=acceptance,
        alpha=alpha,
        new_component=new_component,
        n_starts=n_starts,
        n_samples=n_samples,
    )
    cliques, _, _, _ = builder.run(C=S, cov_matrix=np.eye(S.shape[0]))
    Rpm = _convert_cliques_to_adjacency(cliques, S)
    if not _is_connected(Rpm):
        raise ValueError(
            "DBHT requires a connected MFCF graph, but the attachment rule left "
            "isolated components; lower `threshold` (or use acceptance='threshold' "
            "with threshold=0) so that every vertex is attached"
        )

    Apm = Rpm.copy()
    Apm[Apm != 0] = D[Apm != 0].copy()
    (Dpm, _) = distance_wei(Apm)
    (H1, Hb, Mb, CliqList, Sb) = CliqHierarchyTree2s(Rpm, method1="uniqueroot")
    del H1, Sb
    Mb = Mb[0 : CliqList.shape[0], :]
    Mv = np.empty((Rpm.shape[0], 0))
    for i in range(0, Mb.shape[1]):
        vec = np.zeros(Rpm.shape[0])
        vec[np.int32(np.unique(CliqList[Mb[:, i] != 0, :]))] = 1
        Mv = np.hstack((Mv, vec.reshape(-1, 1)))

    (Adjv, T8) = BubbleCluster8s(Rpm, Dpm, Hb, Mb, Mv, CliqList)
    Z = HierarchyConstruct4s(Rpm, Dpm, T8, Adjv, Mv)

    if leaf_order == True:
        # symmetrise: correlation-based distances are symmetric only up to rounding
        Dsym = (np.asarray(D, dtype=float) + np.asarray(D, dtype=float).T) / 2
        np.fill_diagonal(Dsym, 0.0)
        Z = optimal_leaf_ordering(Z, squareform(Dsym))

    return (T8, Rpm, Adjv, Dpm, Mv, Z)


def _is_connected(A) -> bool:
    """True if the graph with adjacency pattern A (nonzero entries) is connected."""
    n = A.shape[0]
    seen = np.zeros(n, dtype=bool)
    stack = [0]
    seen[0] = True
    while stack:
        i = stack.pop()
        for j in np.flatnonzero((A[i] != 0) & ~seen):
            seen[j] = True
            stack.append(j)
    return bool(seen.all())


def _convert_cliques_to_adjacency(cliques, S):
    """
    Convert a list of cliques into a weighted adjacency matrix.

    Each clique is assumed to be an iterable of vertex indices. For every clique,
    the submatrix induced by its vertices is filled with the corresponding
    weights from ``S``. The diagonal is set to zero at the end.

    Parameters
    ----------
    cliques : Iterable[Iterable[int]]
        Collection of cliques. Elements may be lists, tuples, sets, or numpy
        arrays of integer indices.

    S : ndarray of shape (N, N)
        Similarity matrix supplying the edge weights.

    Returns
    -------
    adj : ndarray of shape (N, N)
        Weighted adjacency matrix with zeros on the diagonal.
    """
    n = S.shape[0]
    adj = np.zeros((n, n), dtype=float)

    for clique in cliques:
        clique = np.array([v for v in clique])
        # Fill clique edges using weights from S
        adj[np.ix_(clique, clique)] = S[np.ix_(clique, clique)]

    np.fill_diagonal(adj, 0)

    return adj


if __name__ == "__main__":
    import numpy as np
    from scipy.spatial.distance import pdist, squareform
    from scipy.cluster.hierarchy import fcluster
    from sklearn.datasets import load_iris

    # --- Load sample data ---
    iris = load_iris()
    X = iris.data
    y_true = iris.target + 1  # match MATLAB convention: classes 1,2,3

    # --- Compute distance and similarity matrices ---
    D = squareform(pdist(X, metric="cityblock"))
    S = 1.0 / (1.0 + D / np.mean(np.abs(D)))

    # --- Run DBHT clustering ---
    T8, Rpm, Adjv, Dpm, Mv, Z = mfcf_dbhts(D, S)

    # --- Evaluate simple 3-cluster cut ---
    Tz = fcluster(Z, t=3, criterion="maxclust")

    # --- Compute misalignments (try all 6 permutations) ---
    from itertools import permutations

    best_mis = len(y_true)
    best_perm = None
    for perm in permutations([1, 2, 3]):
        mapping = {1: perm[0], 2: perm[1], 3: perm[2]}
        mapped = np.vectorize(mapping.get)(Tz)
        mis = np.sum(mapped != y_true)
        if mis < best_mis:
            best_mis = mis
            best_perm = mapping

    print(f"Found {len(np.unique(T8))} clusters.")
    print(f"Best label mapping: {best_perm}")
    print(
        f"Misalignments: {best_mis}/{len(y_true)} (accuracy={(1-best_mis/len(y_true)):.2%})"
    )
