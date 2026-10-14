"""Robustness and integration tests: edge cases, determinism, relabelling,
scikit-learn API and DBHT. Run with: pytest -q src/test_mfcf_robustness.py"""
import itertools

import networkx as nx
import numpy as np
import pytest

from fast_fast_mfcf import MFCF


def _data(n=200, p=20, seed=0, k=3):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, k)) @ rng.standard_normal((k, p)) + rng.standard_normal((n, p))


def _graph(cliques, p):
    g = nx.Graph()
    g.add_nodes_from(range(p))
    g.add_edges_from(e for q in cliques for e in itertools.combinations(sorted(int(v) for v in q), 2))
    return g


def _valid(cliques, p, kmax):
    g = _graph(cliques, p)
    assert nx.is_chordal(g)
    assert max(len(q) for q in cliques) <= kmax
    assert {frozenset(int(v) for v in q) for q in cliques} == {frozenset(q) for q in nx.find_cliques(g)}
    assert sorted({int(v) for q in cliques for v in q}) == list(range(p))


# ---------------------------------------------------------------- edge cases
@pytest.mark.parametrize("gain", ["loglik", "sumsquares"])
@pytest.mark.parametrize("p", [1, 2, 3])
@pytest.mark.parametrize("kmin,kmax", [(1, 2), (1, 4), (4, 4), (6, 8)])
def test_tiny_problems(gain, p, kmin, kmax):
    C = np.corrcoef(_data(p=max(p, 2))[:, :p], rowvar=False).reshape(p, p)
    cl, seps, peo, J = MFCF(min_clique_size=kmin, max_clique_size=kmax, gain_function_type=gain).run(C)
    assert sorted(int(v) for v in peo) == list(range(p))
    _valid(cl, p, max(kmax, kmin))


@pytest.mark.parametrize("rule", ["threshold", "ftest", "bic"])
@pytest.mark.parametrize("kmax", [3, 10, 40])
def test_fewer_observations_than_variables(rule, kmax):
    X = _data(n=25, p=60)
    m = MFCF(max_clique_size=kmax, acceptance=rule, threshold=0.05 if rule == "threshold" else 0.0)
    m.fit(X, shrinkage=0.1)
    _valid(m.cliques_, 60, kmax)
    assert np.all(np.isfinite(m.precision_)) and np.linalg.eigvalsh(m.precision_).min() > 0


def test_duplicated_and_collinear_columns():
    X = _data(n=100, p=10)
    X = np.hstack([X, X[:, :2], X[:, [0]] + X[:, [1]]])          # exact duplicates and a sum
    m = MFCF(max_clique_size=5).fit(X, shrinkage=0.05)
    _valid(m.cliques_, 13, 5)
    assert np.all(np.isfinite(m.precision_))
    m2 = MFCF(max_clique_size=5, acceptance="ftest").fit(X, shrinkage=0.05)
    _valid(m2.cliques_, 13, 5)


def test_bad_data_is_rejected():
    X = _data()
    with pytest.raises(ValueError):
        MFCF().fit(np.hstack([X, np.ones((X.shape[0], 1))]))     # constant column
    Xn = X.copy(); Xn[3, 4] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        MFCF().fit(Xn)                                            # missing values
    with pytest.raises(ValueError):
        MFCF().fit(X[:2])                                         # too few observations


@pytest.mark.parametrize("dtype", [np.float32, np.int64])
def test_input_dtypes(dtype):
    X = (_data() * 10).astype(dtype)
    m = MFCF(max_clique_size=4).fit(X)
    _valid(m.cliques_, X.shape[1], 4)


def test_clique_size_larger_than_p():
    X = _data(p=8)
    m = MFCF(max_clique_size=50).fit(X)
    assert len(m.cliques_) == 1 and len(m.cliques_[0]) == 8      # threshold 0: complete graph


# ------------------------------------------------- determinism and labels
@pytest.mark.parametrize("kw", [dict(), dict(acceptance="ftest", alpha=0.05),
                                dict(gain_function_type="sumsquares", threshold=0.05)])
def test_deterministic(kw):
    X = _data(seed=5)
    C = np.corrcoef(X, rowvar=False)
    n = dict(n_samples=X.shape[0]) if kw.get("acceptance") else {}
    a = MFCF(max_clique_size=5, **kw, **n).run(C)
    b = MFCF(max_clique_size=5, **kw, **n).run(C)
    assert [sorted(q) for q in a[0]] == [sorted(q) for q in b[0]] and list(a[2]) == list(b[2])


@pytest.mark.parametrize("kw", [dict(), dict(acceptance="ftest", alpha=0.05)])
def test_relabelling_equivariance(kw):
    """Without ties, relabelling the variables relabels the clique forest."""
    X = _data(seed=7)
    perm = np.random.default_rng(1).permutation(X.shape[1])
    a = MFCF(max_clique_size=5, **kw).fit(X)
    b = MFCF(max_clique_size=5, **kw).fit(X[:, perm])
    mapped = {frozenset(int(perm[v]) for v in q) for q in b.cliques_}
    assert mapped == {frozenset(int(v) for v in q) for q in a.cliques_}
    assert np.allclose(b.precision_, a.precision_[np.ix_(perm, perm)])


def test_fit_is_reusable():
    m = MFCF(max_clique_size=4)
    m.fit(_data(seed=1)); first = [sorted(q) for q in m.cliques_]
    m.fit(_data(seed=2)); m.fit(_data(seed=1))
    assert [sorted(q) for q in m.cliques_] == first


# ------------------------------------------------------ scikit-learn API
def test_sklearn_estimators():
    pytest.importorskip("sklearn")
    pytest.importorskip("optuna")
    from sklearn.base import clone
    from sklearn.model_selection import cross_val_score
    from mfcf_logo import MFCFLoGo, MFCFLoGoCV

    X = _data(n=300, p=15) * np.random.default_rng(2).uniform(0.1, 50, 15)
    est = MFCFLoGo(max_clique_size=5, acceptance="ftest", alpha=0.01, shrinkage=0.05)
    c = clone(est)
    assert c.get_params()["acceptance"] == "ftest" and c.get_params()["shrinkage"] == 0.05
    scores = cross_val_score(est, X, cv=3)
    assert np.all(np.isfinite(scores))
    est.fit(X)
    S = np.cov(X, rowvar=False, bias=True)
    assert np.allclose(np.diag(est.covariance_), np.diag(S) * 1, rtol=0.1)
    cv = MFCFLoGoCV(max_clique_size_grid=[3, 5], acceptance="bic", cv=3).fit(X)
    assert cv.best_max_clique_size_ in (3, 5)


# --------------------------------------------------------------------- DBHT
@pytest.mark.parametrize("gain", ["loglik", "sumsquares"])
def test_dbht(gain):
    pytest.importorskip("riskfolio")
    from mfcf_dbht import mfcf_dbhts
    from scipy.cluster.hierarchy import fcluster
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(4), 10)                    # 4 planted groups of 10
    X = rng.standard_normal((500, 4))[:, groups] + 0.8 * rng.standard_normal((500, 40))
    S = np.corrcoef(X, rowvar=False)
    D = np.sqrt(np.clip(2 * (1 - S), 0, None))
    np.fill_diagonal(D, 0.0)
    T8, Rpm, Adjv, Dpm, Mv, Z = mfcf_dbhts(D, S, max_clique_size=4, gain_function_type=gain)
    labels = fcluster(Z, 4, "maxclust")
    # each planted group falls in a single cluster
    assert all(len(set(labels[groups == k])) == 1 for k in range(4))
