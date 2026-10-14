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
    ["loglik", "sumsquares", "sum"],  # gain function
))
CASES = [c for c in CASES if c[3] <= c[2]]


@pytest.mark.parametrize("n,kind,kmax,kmin,c,theta,gain", CASES)
def test_mfcf_against_independent_reference(n, kind, kmax, kmin, c, theta, gain):
    C = _corr(n, 0, kind)
    if gain == "sum":
        C = np.abs(C)                          # "sum" requires non-negative weights
    out = MFCF(threshold=theta, min_clique_size=kmin, max_clique_size=kmax,
               coordination_number=c, gain_function_type=gain).run(C)
    _check_against_reference(n, C, out, kmax, kmin, c, theta)


def _check_against_reference(n, C, out, kmax, kmin, c, theta):
    cliques, seps, peo, J = out
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



# --------------------------------------------------------------------------
# statistical acceptance (loglik gain)
# --------------------------------------------------------------------------
STAT_CASES = list(itertools.product([30], ["sector", "signed"], [3, 6], [1, 3],
                                    [1, np.inf], ["ftest", "bic"]))
STAT_CASES = [c for c in STAT_CASES if c[3] <= c[2]]


@pytest.mark.parametrize("n,kind,kmax,kmin,c,rule", STAT_CASES)
def test_statistical_acceptance_invariants(n, kind, kmax, kmin, c, rule):
    C = _corr(n, 0, kind)
    out = MFCF(min_clique_size=kmin, max_clique_size=kmax, coordination_number=c,
               acceptance=rule, alpha=0.01, n_samples=3 * n + 20).run(C)
    _check_against_reference(n, C, out, kmax, kmin, c, theta=1.0)   # theta>0: no k-tree check


def _sparse_chordal_sample(p=40, n=200, seed=3):
    rng = np.random.default_rng(seed)
    L = np.eye(p)
    for v in range(1, p):                       # a random tree as precision support
        L[rng.integers(0, v), v] = rng.choice([-1, 1]) * rng.uniform(0.5, 0.9)
    X = rng.standard_normal((n, p)) @ np.linalg.inv(L @ L.T + 0 * np.eye(p)).T ** 0
    S = np.linalg.inv(L @ L.T)
    X = rng.standard_normal((n, p)) @ np.linalg.cholesky(S).T
    return np.corrcoef(X, rowvar=False), n


def test_ftest_sparsity_decreases_with_alpha():
    C, n = _sparse_chordal_sample()
    edges = []
    for a in (0.2, 0.01, 1e-4):
        cl, _, _, _ = MFCF(max_clique_size=8, acceptance="ftest", alpha=a, n_samples=n).run(C)
        edges.append(len({frozenset(e) for q in cl for e in itertools.combinations(sorted(q), 2)}))
    assert edges[0] >= edges[1] >= edges[2]


def test_ftest_rejects_independent_noise():
    rng = np.random.default_rng(0)
    C = np.corrcoef(rng.standard_normal((500, 20)), rowvar=False)
    cl, seps, _, _ = MFCF(max_clique_size=5, acceptance="ftest", alpha=1e-4, n_samples=500).run(C)
    assert sum(len(q) > 1 for q in cl) <= 1           # (almost) all singletons


def test_option_validation():
    C = _corr(10, 0, "sector")
    with pytest.raises(ValueError):
        MFCF(acceptance="ftest").run(C)                           # needs n_samples
    with pytest.raises(ValueError):
        MFCF(gain_function_type="sumsquares", acceptance="bic", n_samples=50)
    with pytest.raises(ValueError):
        MFCF(gain_function_type="foo")
    W = np.abs(C) - 0.5                                            # not a covariance
    np.fill_diagonal(W, 1.0)
    W[0, 1] = W[1, 0] = 1.5
    with pytest.raises(ValueError):
        MFCF().run(W)
    cl, _, _, _ = MFCF(gain_function_type="sumsquares").run(W)     # weight gains accept it
    assert sorted({int(v) for q in cl for v in q}) == list(range(10))
    with pytest.raises(ValueError):
        MFCF(gain_function_type="sum").run(W)                     # negative weights rejected
    cl, _, _, _ = MFCF(gain_function_type="sum").run(np.abs(W))
    assert sorted({int(v) for q in cl for v in q}) == list(range(10))


def test_weight_matrix_without_logo():
    rng = np.random.default_rng(1)
    W = rng.uniform(0, 1, (15, 15)); W = (W + W.T) / 2; np.fill_diagonal(W, 0.0)
    cl, seps, peo, J = MFCF(gain_function_type="sum", max_clique_size=4).run(W)
    assert sorted(int(v) for v in peo) == list(range(15))



# --------------------------------------------------------------------------
# building directly from data: MFCF.fit(X)
# --------------------------------------------------------------------------
def _data(n=400, p=25, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, 3)) @ rng.standard_normal((3, p)) + rng.standard_normal((n, p))


def test_fit_matches_run_on_correlation_and_is_scale_invariant():
    X = _data()
    C = np.corrcoef(X, rowvar=False)
    m = MFCF(max_clique_size=5).fit(X)
    cl, seps, peo, J = MFCF(max_clique_size=5).run(C)
    assert {frozenset(q) for q in m.cliques_} == {frozenset(q) for q in cl}
    s = np.random.default_rng(1).uniform(0.1, 10, X.shape[1])
    m2 = MFCF(max_clique_size=5).fit(X * s)
    assert {frozenset(q) for q in m2.cliques_} == {frozenset(q) for q in m.cliques_}
    assert np.allclose(m2.precision_, m.precision_ / np.outer(s, s))


def test_fit_precision_is_decomposable_mle():
    X = _data(n=500, p=20, seed=2)
    m = MFCF(max_clique_size=4).fit(X)
    S = np.cov(X, rowvar=False, bias=True)
    for q in m.cliques_:
        q = sorted(q)
        assert np.allclose(m.covariance_[np.ix_(q, q)], S[np.ix_(q, q)], atol=1e-8)
    assert np.allclose(np.linalg.inv(m.precision_), m.covariance_, atol=1e-8)


def test_fit_sets_n_for_statistical_acceptance():
    X = _data()
    m = MFCF(max_clique_size=6, acceptance="ftest", alpha=0.01).fit(X)
    assert m.n_effective_ == X.shape[0]
    cl, _, _, _ = MFCF(max_clique_size=6, acceptance="ftest", alpha=0.01,
                       n_samples=X.shape[0]).run(np.corrcoef(X, rowvar=False))
    assert {frozenset(q) for q in m.cliques_} == {frozenset(q) for q in cl}


def test_effective_sample_size():
    rng = np.random.default_rng(3)
    n, p = 4000, 10
    assert MFCF(acceptance="bic").fit(rng.standard_normal((n, p)), n_effective="auto").n_effective_ > 0.9 * n
    phi = 0.8                                        # AR(1) series: n_eff ~ n (1 - phi^2) / (1 + phi^2)
    e = rng.standard_normal((n, p)); X = np.zeros((n, p))
    for t in range(1, n):
        X[t] = phi * X[t - 1] + e[t]
    neff = MFCF(acceptance="bic").fit(X, n_effective="auto").n_effective_
    assert abs(neff / (n * (1 - phi**2) / (1 + phi**2)) - 1) < 0.15


def test_fit_rejects_weight_gain():
    with pytest.raises(ValueError):
        MFCF(gain_function_type="sumsquares").fit(_data())
