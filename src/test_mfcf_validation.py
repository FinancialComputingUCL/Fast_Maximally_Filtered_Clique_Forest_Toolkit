"""Regression tests for the MFCF builder.

Every output is checked against an independent reference built from the
returned graph: maximal cliques, separator multiplicities of a junction forest,
the coordination cap, and the LoGo matrix J.  Run with:  pytest -q src/test_mfcf_validation.py
"""
import itertools
from collections import Counter

import networkx as nx
import numpy as np
import pytest

from fast_fast_mfcf import MFCF, Gains


def _corr(n_vars, seed, kind):
    rng = np.random.default_rng(seed)
    k = max(2, n_vars // 10)
    sec = rng.integers(0, k, n_vars)
    t = 3 * n_vars + 20
    x = 0.5 * rng.standard_normal((t, 1)) + 0.8 * rng.standard_normal((t, k))[:, sec] + rng.standard_normal((t, n_vars))
    if kind == "signed":
        x = x * rng.choice([-1, 1], n_vars)
    if kind == "ties":
        b = np.full((n_vars, n_vars), 0.3)
        b[sec[:, None] == sec[None, :]] = 0.6
        np.fill_diagonal(b, 1.0)
        return b
    return np.corrcoef(x, rowvar=False)


def _outerplanar(g):
    h = g.copy()
    h.add_edges_from((-1, v) for v in g)
    return nx.check_planarity(h)[0]


CASES = list(itertools.product(
    [5, 30],                       # N
    ["sector", "signed", "ties"],  # input
    [2, 3, 4, 6],                  # max_clique_size
    [1, 2, 3],                     # min_clique_size
    [1, 2, np.inf],                # coordination_number
    [0.0, 0.05, 0.2],              # threshold
    ["loglik", "sumsquares"],      # gain function
))
CASES = [c for c in CASES if c[3] <= c[2]]


@pytest.mark.parametrize("n,kind,kmax,kmin,c,theta,gain", CASES)
def test_mfcf_against_independent_reference(n, kind, kmax, kmin, c, theta, gain):
    C = _corr(n, 0, kind)
    cliques, seps, peo, J = MFCF(threshold=theta, min_clique_size=kmin,
                                 max_clique_size=kmax, coordination_number=c,
                                 gain_function_type=gain).run(C)
    cliques = [frozenset(int(v) for v in q) for q in cliques]
    seps = Counter({frozenset(int(v) for v in s): m for s, m in seps.items() if len(s)})

    g = nx.Graph()
    g.add_nodes_from(range(n))
    g.add_edges_from(e for q in cliques for e in itertools.combinations(sorted(q), 2))

    assert sorted(int(v) for v in peo) == list(range(n))
    assert max(len(q) for q in cliques) <= kmax
    assert nx.is_chordal(g)
    assert set(cliques) == {frozenset(q) for q in nx.find_cliques(g)}, "returned cliques are not the maximal cliques"

    # separators of an independently built junction forest
    qs = list(set(cliques))
    h = nx.Graph()
    h.add_nodes_from(range(len(qs)))
    for i, j in itertools.combinations(range(len(qs)), 2):
        if qs[i] & qs[j]:
            h.add_edge(i, j, weight=len(qs[i] & qs[j]))
    ref = Counter(qs[i] & qs[j] for i, j in nx.maximum_spanning_tree(h).edges())
    assert seps == ref, "separator multiplicities do not match the clique forest"
    if np.isfinite(c) and seps:
        assert max(seps.values()) <= c, "coordination number exceeded"

    # LoGo: J must equal the clique/separator formula and reproduce C on the graph
    if np.linalg.eigvalsh(C).min() > 1e-8:
        jref = np.zeros((n, n))
        for q in qs:
            q = sorted(q); jref[np.ix_(q, q)] += np.linalg.inv(C[np.ix_(q, q)])
        for s, m in ref.items():
            s = sorted(s); jref[np.ix_(s, s)] -= m * np.linalg.inv(C[np.ix_(s, s)])
        assert np.allclose(J, jref, atol=1e-8), "J differs from LoGo on the clique forest"
        assert np.linalg.eigvalsh((J + J.T) / 2).min() > 0
        a = nx.to_numpy_array(g, nodelist=range(n)) + np.eye(n)
        assert np.allclose(np.linalg.inv(J)[a > 0], C[a > 0], atol=1e-8)

    # network classes for theta = 0, min_clique_size <= 1
    if theta == 0 and kmin <= 1 and n >= kmax:
        assert g.number_of_edges() == (kmax - 1) * n - kmax * (kmax - 1) // 2
        deg = max(dict(g.degree()).values())
        if kmax == 2 and np.isfinite(c):
            assert deg <= c + 1
        if c == 1 and kmax == 3:
            assert _outerplanar(g)
        if c == 1 and kmax == 4:
            assert nx.check_planarity(g)[0]


def test_default_gain_is_loglik():
    assert MFCF()._gf_type == "loglik"


@pytest.mark.parametrize("seed", range(5))
def test_loglik_gain_equals_regression(seed):
    """With threshold 0 the whole separator is used and the gain must equal
    -1/2 log(1 - R^2) of the regression of the chosen node on the separator."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((300, 12)) @ rng.standard_normal((12, 12)) + rng.standard_normal((300, 12))
    C = np.corrcoef(X, rowvar=False)
    sep = frozenset(range(5))
    mask = np.zeros(12, dtype=bool)
    mask[5:] = True
    g, v, sw = Gains(C, 0.0, 1, "loglik").get_best_gain(mask, sep)
    s = sorted(sw.separator)
    assert s == list(range(5))
    r2 = C[v, s] @ np.linalg.solve(C[np.ix_(s, s)], C[s, v])
    assert np.isclose(g, -0.5 * np.log(1 - r2))
    best = max(-0.5 * np.log(1 - C[u, s] @ np.linalg.solve(C[np.ix_(s, s)], C[s, u])) for u in range(5, 12))
    assert np.isclose(g, best)


@pytest.mark.parametrize("k", [3, 4, 6])
def test_loglik_fixed_size_is_k_tree(k):
    C = _corr(30, 1, "sector")
    cliques, seps, peo, J = MFCF(threshold=0.0, min_clique_size=k, max_clique_size=k,
                                 gain_function_type="loglik").run(C)
    edges = {frozenset(e) for q in cliques for e in itertools.combinations(sorted(q), 2)}
    assert len(edges) == (k - 1) * 30 - k * (k - 1) // 2
    assert all(len(q) == k for q in cliques)
