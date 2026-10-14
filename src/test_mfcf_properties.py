"""Mathematical, algorithmic and statistical properties of the MFCF builder.

- identities: gain = Gaussian mutual information; log det of the LoGo matrix
- greedy optimality: every insertion has the largest gain among all allowed
  candidates (brute force, small problems)
- calibration of the partial F-test and consistency of structure recovery
- randomized fuzzing over all options, numerical stress, pickling

Run with:  pytest -q src/test_mfcf_properties.py
"""
import itertools
import pickle

import networkx as nx
import numpy as np
import pytest
from scipy import stats

from fast_fast_mfcf import MFCF, Gains


def _corr(p, n=400, seed=0, k=3):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, k)) @ rng.standard_normal((k, p)) + rng.standard_normal((n, p))
    return np.corrcoef(X, rowvar=False)


def _ldet(M):
    return np.linalg.slogdet(M)[1]


# ------------------------------------------------------------- identities
@pytest.mark.parametrize("seed", range(10))
def test_gain_is_gaussian_mutual_information(seed):
    C = _corr(15, seed=seed)
    rng = np.random.default_rng(seed)
    S = frozenset(int(x) for x in rng.choice(15, size=int(rng.integers(1, 6)), replace=False))
    mask = np.ones(15, dtype=bool)
    mask[list(S)] = False
    g, v, sw = Gains(C, 0.0, 1, "loglik").get_best_gain(mask, S)
    s = sorted(S)
    B = sorted(S | {v})
    mi = 0.5 * (np.log(C[v, v]) + _ldet(C[np.ix_(s, s)]) - _ldet(C[np.ix_(B, B)]))
    assert np.isclose(g, mi)


@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize("kw", [dict(max_clique_size=4, min_clique_size=4),
                                dict(max_clique_size=6, acceptance="ftest", alpha=0.05, n_samples=400),
                                dict(max_clique_size=5, threshold=0.05)])
def test_logo_log_determinant(seed, kw):
    """log det J = -sum_C log det R_C + sum_S m_S log det R_S (decomposable model)."""
    C = _corr(25, seed=seed)
    cl, seps, peo, J = MFCF(**kw).run(C)
    expected = -sum(_ldet(C[np.ix_(sorted(q), sorted(q))]) for q in cl) \
        + sum(m * _ldet(C[np.ix_(sorted(s), sorted(s))]) for s, m in seps.items() if s)
    assert np.isclose(_ldet(J), expected)


# ------------------------------------------------------- greedy optimality
class _Recorder(MFCF):
    """Records, for every insertion, the vertex, the separator and the cliques
    available just before it."""

    def _add_new_clique(self, parent_clique, sep, v):
        self.steps.append((int(v), frozenset(int(x) for x in sep),
                           [frozenset(int(x) for x in q) for q in self._cliques],
                           self._outstanding_nodes_mask.copy()))
        return super()._add_new_clique(parent_clique, sep, v)

    def run(self, C, cov_matrix=None):
        self.steps = []
        return super().run(C, cov_matrix)


def _mi(C, v, S):
    if not S:
        return 0.0
    s = sorted(S)
    r2 = C[v, s] @ np.linalg.solve(C[np.ix_(s, s)], C[s, v])
    return -0.5 * np.log(1 - r2)


@pytest.mark.parametrize("seed", range(12))
@pytest.mark.parametrize("k", [3, 4, 5])
def test_each_insertion_maximises_the_gain(seed, k):
    """Fixed clique size, threshold 0: the admissible separators are the facets
    of the current cliques (cliques are full); every insertion must attain the
    largest gain over all outstanding vertices and all facets."""
    p = 14
    C = _corr(p, seed=seed)
    rec = _Recorder(min_clique_size=k, max_clique_size=k)
    rec.run(C, cov_matrix=np.eye(p))
    for v, sep, cliques, outstanding in rec.steps:
        chosen = _mi(C, v, sep)
        facets = {q - {w} for q in cliques for w in q} if all(len(q) == k for q in cliques) else set(cliques)
        best = max(_mi(C, u, f) for u in np.flatnonzero(outstanding) for f in facets if len(f) == k - 1)
        assert chosen >= best - 1e-10


# ----------------------------------------------------- statistical checks
def test_ftest_single_step_is_calibrated():
    """Under the null (v independent of a single separator vertex), the
    first-step F-test must accept at rate alpha."""
    rng = np.random.default_rng(0)
    n, alpha, reps = 60, 0.05, 4000
    acc = 0
    for _ in range(reps):
        X = rng.standard_normal((n, 2))
        C = np.corrcoef(X, rowvar=False)
        g, v, sw = Gains(C, 0.0, 1, "loglik", "ftest", alpha, n).get_best_gain(
            np.array([False, True]), frozenset({0}))
        acc += len(sw.separator) == 1
    rate = acc / reps
    # the F-test is exact for a prespecified regression with intercept
    lo, hi = stats.binom.ppf([0.0005, 0.9995], reps, alpha) / reps
    assert lo <= rate <= hi, rate


def _chordal_model(p=30, seed=0, kmax=4):
    """Random chordal graph by clique expansion and a precision with exactly
    that support (L L^T on a perfect elimination ordering)."""
    rng = np.random.default_rng(seed)
    cliques = [[0, 1, 2]]
    later = {v: set(range(v)) if v < 3 else set() for v in range(p)}
    for v in range(3, p):
        q = cliques[rng.integers(len(cliques))]
        s = list(rng.choice(q, size=int(rng.integers(1, min(len(q), kmax - 1) + 1)), replace=False))
        cliques.append(s + [v]) if len(s) < len(q) else q.append(v)
        later[v] = set(int(x) for x in s)
    L = np.eye(p)
    for v in range(p):
        for u in later[v]:
            L[u, v] = rng.choice([-1, 1]) * rng.uniform(0.6, 0.9)
    J = L @ L.T
    S = np.linalg.inv(J)
    adj = np.abs(J) > 1e-12
    np.fill_diagonal(adj, False)
    return S, adj


def _edges(cliques, p):
    E = np.zeros((p, p), dtype=bool)
    for q in cliques:
        q = sorted(q)
        E[np.ix_(q, q)] = True
    np.fill_diagonal(E, False)
    return E


@pytest.mark.parametrize("seed", range(5))
def test_tree_recovery_chow_liu(seed):
    """For a Gaussian tree, MFCF with cliques of size 2 and the log-likelihood
    gain is Prim's maximum spanning tree on pairwise mutual information
    (Chow-Liu), which recovers the true tree: exactly in the population and
    with a large sample."""
    rng = np.random.default_rng(seed)
    p = 30
    L = np.eye(p)
    for v in range(1, p):
        L[rng.integers(0, v), v] = rng.choice([-1, 1]) * rng.uniform(0.4, 0.9)
    J = L @ L.T
    S = np.linalg.inv(J)
    tree = np.abs(J) > 1e-12
    np.fill_diagonal(tree, False)
    d = 1 / np.sqrt(np.diag(S))
    R = S * d[:, None] * d[None, :]
    cl, _, _, _ = MFCF(min_clique_size=2, max_clique_size=2).run(R)
    assert np.array_equal(_edges(cl, p), tree)
    X = rng.standard_normal((20000, p)) @ np.linalg.cholesky(S).T
    m = MFCF(min_clique_size=2, max_clique_size=2).fit(X)
    assert np.array_equal(_edges(m.cliques_, p), tree)


@pytest.mark.parametrize("rule", ["ftest", "bic"])
def test_chordal_recovery_quality_in_population(rule):
    """MFCF is greedy and order dependent: a vertex inserted before its true
    neighbours attaches through proxies, so even the population covariance
    does not guarantee exact recovery of a general chordal graph. Regression
    guard on recovery quality over 20 random chordal graphs (observed: mean
    Matthews correlation 0.90, minimum 0.80)."""
    vals = []
    for seed in range(20):
        S, adj = _chordal_model(seed=seed)
        d = 1 / np.sqrt(np.diag(S))
        R = S * d[:, None] * d[None, :]
        cl, _, _, _ = MFCF(max_clique_size=6, acceptance=rule, alpha=1e-4, n_samples=1e6).run(R)
        E = _edges(cl, S.shape[0])
        iu = np.triu_indices_from(adj, 1)
        tp = np.sum(E[iu] & adj[iu]); fp = np.sum(E[iu] & ~adj[iu])
        fn = np.sum(~E[iu] & adj[iu]); tn = np.sum(~E[iu] & ~adj[iu])
        vals.append((tp * tn - fp * fn) / np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    vals = np.array(vals)
    assert vals.mean() >= 0.88 and vals.min() >= 0.75, (vals.mean(), vals.min())


# ------------------------------------------------------------------- fuzz
FUZZ = range(150)


@pytest.mark.parametrize("seed", FUZZ)
def test_fuzz_all_options(seed):
    rng = np.random.default_rng(10_000 + seed)
    p = int(rng.integers(2, 35))
    n = int(rng.integers(5, 200))
    gain = str(rng.choice(["loglik", "sumsquares", "sum"]))
    acceptance = str(rng.choice(["threshold", "ftest", "bic"])) if gain == "loglik" else "threshold"
    kmax = int(rng.integers(2, 9))
    kmin = int(rng.integers(1, kmax + 1))
    kw = dict(max_clique_size=kmax, min_clique_size=kmin,
              coordination_number=rng.choice([1, 2, 3, np.inf]),
              threshold=float(rng.choice([0.0, 0.01, 0.1, 0.3])),
              gain_function_type=gain, acceptance=acceptance, alpha=float(rng.choice([1e-3, 0.05, 0.3])))
    X = rng.standard_normal((n, p)) @ rng.standard_normal((p, p)) * rng.uniform(0, 1) + rng.standard_normal((n, p))
    C = np.corrcoef(X, rowvar=False) if p > 1 else np.ones((1, 1))
    if gain == "sum":
        C = np.abs(C)
    if acceptance != "threshold":
        kw["n_samples"] = n
    cl, seps, peo, J = MFCF(**kw).run(C, cov_matrix=np.eye(p))
    g = nx.Graph()
    g.add_nodes_from(range(p))
    g.add_edges_from(e for q in cl for e in itertools.combinations(sorted(int(v) for v in q), 2))
    assert sorted(int(v) for v in peo) == list(range(p))
    assert nx.is_chordal(g)
    assert max(len(q) for q in cl) <= max(kmax, kmin)
    assert {frozenset(int(v) for v in q) for q in cl} == {frozenset(q) for q in nx.find_cliques(g)}
    if np.isfinite(kw["coordination_number"]) and seps:
        assert max(seps.values()) <= kw["coordination_number"]
    # the separators form a junction forest of the cliques
    assert sum(seps.values()) == len(cl) - nx.number_connected_components(g)


# --------------------------------------------------------- numerical stress
def test_near_perfect_correlation():
    rng = np.random.default_rng(0)
    z = rng.standard_normal((500, 1))
    X = np.hstack([z + 1e-6 * rng.standard_normal((500, 1)) for _ in range(6)] + [rng.standard_normal((500, 6))])
    m = MFCF(max_clique_size=4).fit(X, shrinkage=0.01)
    assert np.all(np.isfinite(m.precision_))
    assert np.linalg.eigvalsh(m.precision_).min() > 0


def test_extreme_scales():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((300, 3)) @ rng.standard_normal((3, 12)) + rng.standard_normal((300, 12))
    s = np.logspace(-8, 8, 12)
    a = MFCF(max_clique_size=4).fit(X)
    b = MFCF(max_clique_size=4).fit(X * s)
    assert {frozenset(q) for q in a.cliques_} == {frozenset(q) for q in b.cliques_}
    assert np.allclose(b.covariance_ / np.outer(s, s), a.covariance_, rtol=1e-6)


def test_float32_matches_float64():
    X = _corr(5)  # placeholder to reuse helper's RNG layout
    rng = np.random.default_rng(2)
    X = rng.standard_normal((400, 3)) @ rng.standard_normal((3, 20)) + rng.standard_normal((400, 20))
    a = MFCF(max_clique_size=5).fit(X)
    b = MFCF(max_clique_size=5).fit(X.astype(np.float32))
    assert {frozenset(q) for q in a.cliques_} == {frozenset(q) for q in b.cliques_}


# ------------------------------------------------------------- engineering
def test_pickle_round_trip():
    rng = np.random.default_rng(3)
    X = rng.standard_normal((200, 10))
    m = MFCF(max_clique_size=4, acceptance="bic").fit(X)
    m2 = pickle.loads(pickle.dumps(m))
    assert [sorted(q) for q in m2.cliques_] == [sorted(q) for q in m.cliques_]
    assert np.array_equal(m2.precision_, m.precision_)


def test_sklearn_estimator_checks():
    pytest.importorskip("sklearn")
    pytest.importorskip("optuna")
    from sklearn.utils.estimator_checks import (check_estimators_pickle, check_fit_idempotent,
                                                check_get_params_invariance, check_dont_overwrite_parameters,
                                                check_n_features_in)
    from mfcf_logo import MFCFLoGo
    est = MFCFLoGo(max_clique_size=3, shrinkage=0.1)
    for check in (check_get_params_invariance, check_dont_overwrite_parameters, check_fit_idempotent,
                  check_estimators_pickle, check_n_features_in):
        check("MFCFLoGo", est)


# ------------------------------------------------- exact acceptance boundaries
def _ols_partial_f_pvalue(y, X_old, x_new):
    """p-value of adding x_new to the regression of y on X_old (with intercept)."""
    n = len(y)
    def rss(M):
        A = np.column_stack([np.ones(n), M]) if M.size else np.ones((n, 1))
        beta, *_ = np.linalg.lstsq(A, y, rcond=None)
        r = y - A @ beta
        return r @ r, A.shape[1]
    r0, _ = rss(X_old)
    r1, k1 = rss(np.column_stack([X_old, x_new]) if X_old.size else x_new[:, None])
    df2 = n - k1
    F = (r0 - r1) / (r1 / df2)
    return stats.f.sf(F, 1, df2)


@pytest.mark.parametrize("n", [6, 9, 15])
def test_ftest_matches_ols_nested_test(n):
    """The first two greedy steps must accept exactly when an independent OLS
    nested-model F-test (with intercept) rejects at level alpha."""
    rng = np.random.default_rng(n)
    alpha, agree, total = 0.1, 0, 0
    for _ in range(300):
        X = rng.standard_normal((n, 3))
        X[:, 0] += rng.uniform(0, 1.5) * X[:, 1] + rng.uniform(0, 1.5) * X[:, 2]
        C = np.corrcoef(X, rowvar=False)
        g, v, sw = Gains(C, 0.0, 1, "loglik", "ftest", alpha, n).get_best_gain(
            np.array([True, False, False]), frozenset({1, 2}))
        Z = (X - X.mean(0)) / X.std(0)
        y = Z[:, 0]
        # greedy order: the regressor with the larger squared correlation first
        first = 1 if C[0, 1] ** 2 >= C[0, 2] ** 2 else 2
        second = 3 - first
        p1 = _ols_partial_f_pvalue(y, np.empty((n, 0)), Z[:, first])
        expect = set()
        if p1 < alpha:
            expect.add(first)
            if _ols_partial_f_pvalue(y, Z[:, [first]], Z[:, second]) < alpha:
                expect.add(second)
        agree += set(sw.separator) == expect
        total += 1
    assert agree == total


@pytest.mark.parametrize("n", [20, 100, 1000])
def test_bic_boundary(n):
    """BIC accepts a step iff n * (-log(1 - rho^2)) > log(n)."""
    rho2_star = 1 - np.exp(-np.log(n) / n)
    for rho2, accept in ((rho2_star * 1.001, True), (rho2_star * 0.999, False)):
        r = np.sqrt(rho2)
        C = np.array([[1.0, r], [r, 1.0]])
        g, v, sw = Gains(C, 0.0, 1, "loglik", "bic", 0.05, n).get_best_gain(
            np.array([True, False]), frozenset({1}))
        assert (len(sw.separator) == 1) == accept


def test_mandatory_picks_bypass_the_test():
    """With min_clique_size = k, attachments of at least k - 1 vertices are
    forced even on pure noise, whatever the statistical rule says."""
    rng = np.random.default_rng(0)
    C = np.corrcoef(rng.standard_normal((300, 15)), rowvar=False)
    for rule in ("ftest", "bic"):
        cl, seps, peo, _ = MFCF(min_clique_size=3, max_clique_size=5, acceptance=rule,
                                alpha=1e-8, n_samples=300).run(C)
        assert all(len(q) >= 3 for q in cl)


@pytest.mark.parametrize("rule", ["ftest", "bic", "threshold"])
def test_mandatory_picks_at_gain_level(rule):
    """On pure noise, with min_clique_size = 3 a candidate keeps exactly the
    2 mandatory vertices of a 3-vertex separator, whatever the rule says."""
    rng = np.random.default_rng(1)
    C = np.corrcoef(rng.standard_normal((300, 6)), rowvar=False)
    mask = np.zeros(6, dtype=bool)
    mask[3:] = True
    g, v, sw = Gains(C, 0.99, 3, "loglik", rule, 1e-8, 300).get_best_gain(mask, frozenset({0, 1, 2}))
    assert len(sw.separator) == 2 and g > 0


@pytest.mark.parametrize("tau", [0.01, 0.05, 0.2])
def test_component_threshold_on_gain_scale(tau):
    """With acceptance='threshold' the new-component rule compares the total
    gain with -1/2 log(1 - tau), the gain of one edge at the threshold: a pair
    with rho^2 just above tau is connected, just below it is not."""
    for rho2, connected in ((tau * 1.2, True), (tau * 0.8, False)):
        r = np.sqrt(rho2)
        C = np.array([[1.0, r], [r, 1.0]])
        cl, _, _, _ = MFCF(max_clique_size=2, threshold=tau).run(C)
        assert (len(cl) == 1) == connected


# ------------------------------------------------------ rank correlations
from fast_fast_mfcf import _kendall_tau_b  # noqa: E402


def test_kendall_tau_b_matches_scipy_with_ties():
    rng = np.random.default_rng(3)
    X = np.round(rng.standard_normal((57, 5)), 1)              # many ties
    T = _kendall_tau_b(X, block=7)
    np.testing.assert_allclose(_kendall_tau_b(X, gram_max_n=10), T, atol=1e-12)   # pairwise path
    for i in range(5):
        for j in range(5):
            ref = 1.0 if i == j else stats.kendalltau(X[:, i], X[:, j]).statistic
            assert T[i, j] == pytest.approx(ref, abs=1e-12)


def test_spearman_mapping():
    rng = np.random.default_rng(4)
    X = rng.standard_normal((200, 4)) @ rng.standard_normal((4, 4))
    R = MFCF(max_clique_size=3).fit(X, correlation="spearman").correlation_
    rs = stats.spearmanr(X).statistic
    np.testing.assert_allclose(R, 2 * np.sin(np.pi * rs / 6), atol=1e-12)


@pytest.mark.parametrize("method", ["spearman", "kendall"])
def test_rank_correlation_invariant_to_monotone_transforms(method):
    rng = np.random.default_rng(5)
    X = rng.standard_normal((150, 12)) @ rng.standard_normal((12, 12))
    Y = np.column_stack([np.exp(X[:, 0]), X[:, 1] ** 3, -1 / (1 + np.exp(X[:, 2])), X[:, 3:] * 7 + 2])
    a = MFCF(max_clique_size=4, acceptance="ftest").fit(X, correlation=method)
    b = MFCF(max_clique_size=4, acceptance="ftest").fit(Y, correlation=method)
    np.testing.assert_allclose(a.correlation_, b.correlation_, atol=1e-12)
    assert [sorted(q) for q in a.cliques_] == [sorted(q) for q in b.cliques_]


@pytest.mark.parametrize("method", ["spearman", "kendall"])
def test_rank_correlation_consistent_for_gaussian_data(method):
    rng = np.random.default_rng(6)
    S, _ = _chordal_model(p=8, seed=1)
    X = rng.multivariate_normal(np.zeros(8), S, size=4000)
    d = 1 / np.sqrt(np.diag(S))
    R = MFCF().fit(X, correlation=method).correlation_
    assert np.max(np.abs(R - S * np.outer(d, d))) < 0.05


@pytest.mark.parametrize("method", ["spearman", "kendall"])
def test_rank_correlation_recovers_tree_under_heavy_tails(method):
    """Chow-Liu recovery of a tree from a Gaussian copula with Cauchy-like
    margins and gross outliers, where Pearson correlations are unreliable."""
    rng = np.random.default_rng(7)
    p, n = 15, 400
    parent = [None] + [int(rng.integers(v)) for v in range(1, p)]
    Z = np.zeros((n, p))
    Z[:, 0] = rng.standard_normal(n)
    for v in range(1, p):
        Z[:, v] = 0.8 * Z[:, parent[v]] + 0.6 * rng.standard_normal(n)
    X = np.tan(0.49 * np.pi * np.tanh(Z))                     # heavy-tailed margins
    X[rng.integers(n, size=5), rng.integers(p, size=5)] = 1e6  # gross outliers
    m = MFCF(max_clique_size=2).fit(X, correlation=method)
    E = {frozenset(q) for q in m.cliques_}
    assert E == {frozenset((v, parent[v])) for v in range(1, p)}


@pytest.mark.parametrize("method", ["spearman", "kendall"])
def test_rank_correlation_positive_definite_when_p_exceeds_n(method):
    rng = np.random.default_rng(8)
    X = rng.standard_normal((10, 30))
    m = MFCF(max_clique_size=4, threshold=0.05).fit(X, correlation=method)
    R = m.correlation_
    assert np.allclose(np.diag(R), 1) and np.allclose(R, R.T)
    assert np.linalg.eigvalsh(R).min() > 0
    assert m.precision_ is not None and np.all(np.isfinite(m.precision_))


def test_rank_correlation_option_validation_and_estimators():
    X = np.random.default_rng(9).standard_normal((80, 6))
    with pytest.raises(ValueError):
        MFCF().fit(X, correlation="distance")
    from mfcf_logo import MFCFLoGo
    m = MFCFLoGo(max_clique_size=3, correlation="kendall").fit(X)
    assert m.get_params()["correlation"] == "kendall"
    assert np.all(np.isfinite(m.precision_))


# ------------------------------------------------------------- n_starts
def _sample_corr(seed=0, p=30, n=200):
    S, _ = _chordal_model(p=p, seed=seed)
    X = np.random.default_rng(seed).multivariate_normal(np.zeros(p), S, size=n)
    return np.corrcoef(X, rowvar=False)


@pytest.mark.parametrize("bad", [0, 6, 2.0, True, -1])
def test_n_starts_validation(bad):
    with pytest.raises(ValueError):
        MFCF(gain_function_type="loglik", max_clique_size=5, n_starts=bad)


@pytest.mark.parametrize("seed", range(4))
def test_n_starts_default(seed):
    """None: max_clique_size starts for 'loglik' (any acceptance), 1 for the weight gains."""
    R = _sample_corr(seed)
    for kw, k in [(dict(acceptance="ftest", n_samples=200), 5), (dict(acceptance="bic", n_samples=200), 5),
                  (dict(threshold=0.02), 5)]:
        kw = dict(gain_function_type="loglik", max_clique_size=5, **kw)
        a, b = MFCF(**kw).run(R), MFCF(n_starts=k, **kw).run(R)
        assert a[0] == b[0] and a[1] == b[1]
    assert MFCF(gain_function_type="loglik", max_clique_size=3)._n_starts == 3
    assert MFCF(gain_function_type="sumsquares", max_clique_size=5)._n_starts == 1
    assert MFCF(gain_function_type="sum", max_clique_size=5)._n_starts == 1


@pytest.mark.parametrize("gain,kw", [("loglik", dict(acceptance="ftest", n_samples=200)),
                                     ("loglik", dict(threshold=0.02)),
                                     ("sumsquares", dict(threshold=0.01)),
                                     ("sum", dict(threshold=0.01))])
@pytest.mark.parametrize("seed", range(6))
def test_n_starts_never_lowers_the_objective(gain, kw, seed):
    R = _sample_corr(seed)
    C = np.abs(R) if gain == "sum" else R
    one = MFCF(gain_function_type=gain, max_clique_size=5, **kw)
    one.run(C)
    many = MFCF(gain_function_type=gain, max_clique_size=5, n_starts=5, **kw)
    cl, seps, peo, J = many.run(C)
    assert 0 <= many.start_ < 5
    assert many._objective() >= one._objective() - 1e-9
    assert max(len(q) for q in cl) <= 5 and nx.is_chordal(nx.from_numpy_array(_edges(cl, C.shape[0]).astype(int)))
    assert sorted(int(v) for v in peo) == list(range(C.shape[0]))


def test_n_starts_objective_is_decomposable_loglik():
    R = _sample_corr(1)
    m = MFCF(gain_function_type="loglik", max_clique_size=4, n_starts=3, acceptance="ftest", n_samples=200)
    cl, seps, _, J = m.run(R)
    # decomposable MLE: -1/2 logdet(R J^-1 ...) equals -1/2 log det(J^-1) on R's scale
    assert m._objective() == pytest.approx(0.5 * np.linalg.slogdet(J)[1], rel=1e-8)


def test_n_starts_relabelling_and_min_clique_size():
    R = _sample_corr(2)
    kw = dict(gain_function_type="loglik", max_clique_size=5, min_clique_size=3,
              acceptance="ftest", n_samples=200, n_starts=4)
    base = {frozenset(q) for q in MFCF(**kw).run(R)[0]}
    for s in range(5):
        P = np.random.default_rng(s).permutation(len(R))
        cl = MFCF(**kw).run(R[np.ix_(P, P)])[0]
        assert {frozenset(int(P[v]) for v in q) for q in cl} == base


def test_n_starts_in_estimators():
    from mfcf_logo import MFCFLoGo, MFCFLoGoCV
    X = np.random.default_rng(3).multivariate_normal(np.zeros(12), _chordal_model(p=12)[0], size=150)
    m = MFCFLoGo(gain_function_type="loglik", max_clique_size=4, n_starts=4, acceptance="ftest").fit(X)
    assert np.all(np.isfinite(m.precision_))
    cv = MFCFLoGoCV(gain_function_type="loglik", max_clique_size_grid=[2, 4], n_starts=4).fit(X)
    assert np.all(np.isfinite(cv.precision_))                        # n_starts capped at k = 2


# ------------------------------------------------- direct path (Numba)
@pytest.mark.parametrize("gain", ["sumsquares", "sum"])
@pytest.mark.parametrize("K,c", [(2, np.inf), (3, 1), (4, 2), (4, np.inf), (6, 3)])
@pytest.mark.parametrize("new_component", ["strongest", "first"])
@pytest.mark.parametrize("seed", range(3))
def test_direct_path_equals_general_path(gain, K, c, new_component, seed):
    """With threshold 0 and the weight gains the Numba direct path must give
    the same clique forest, separators and order as the general path."""
    import fast_fast_mfcf as ff
    if not ff._HAS_NUMBA:
        pytest.skip("numba not installed")
    R = _sample_corr(seed, p=40, n=120)
    M = np.abs(R) if gain == "sum" else R
    out = []
    for fast in (True, False):
        m = MFCF(gain_function_type=gain, max_clique_size=K, coordination_number=c,
                 new_component=new_component, n_starts=min(2, K))
        m._use_fast_path = fast
        assert m._fast_path_applies() == fast
        cl, sp, peo, J = m.run(M)
        out.append(([frozenset(map(int, q)) for q in cl], dict(sp), [int(v) for v in peo], m.start_))
    assert out[0] == out[1]


def test_direct_path_not_used_outside_its_regime():
    for kw in (dict(gain_function_type="loglik"), dict(gain_function_type="sumsquares", threshold=0.1),
               dict(gain_function_type="sumsquares", min_clique_size=3)):
        assert not MFCF(max_clique_size=4, **kw)._fast_path_applies()


# ------------------------------------------- mutual information similarity
def test_mutual_information_similarity_in_fit():
    rng = np.random.default_rng(11)
    X = rng.standard_normal((400, 8)) @ rng.standard_normal((8, 8))
    m = MFCF(gain_function_type="sumsquares", max_clique_size=3).fit(
        X, similarity="mutual_information", mi_estimator="histogram")
    S = m.similarity_
    assert np.allclose(np.diag(S), 1) and S.min() >= 0 and S.max() <= 1
    assert np.all(np.isfinite(m.precision_))
    # precomputed matrix gives the same structure
    m2 = MFCF(gain_function_type="sumsquares", max_clique_size=3).fit(
        X, similarity="mutual_information", mi_matrix=S)
    assert {frozenset(q) for q in m.cliques_} == {frozenset(q) for q in m2.cliques_}
    with pytest.raises(ValueError, match="mutual_information"):
        MFCF(gain_function_type="loglik").fit(X, similarity="mutual_information", mi_matrix=S)
    import warnings
    with warnings.catch_warnings():                 # unset gain: sumsquares, as before
        warnings.simplefilter("error")
        m3 = MFCF(max_clique_size=3).fit(X, similarity="mutual_information", mi_matrix=S)
    assert m3._gf_type == "sumsquares" and m3._n_starts == 1
    assert {frozenset(q) for q in m3.cliques_} == {frozenset(q) for q in m2.cliques_}
    with pytest.raises(ValueError):
        MFCF(gain_function_type="sumsquares").fit(X, similarity="distance")


def test_mutual_information_finds_nonlinear_dependence():
    """Chain of non-monotone links (y = z^2 + noise, z a symmetric score of x):
    the first link has zero correlation, so the correlation-based tree misses
    it, while the mutual-information tree recovers the whole chain."""
    rng = np.random.default_rng(12)
    n, p = 2000, 6
    X = np.zeros((n, p))
    X[:, 0] = rng.standard_normal(n)
    for v in range(1, p):
        z = stats.norm.ppf(stats.rankdata(X[:, v - 1]) / (n + 1))   # symmetric score
        X[:, v] = z ** 2 - 1 + 0.3 * rng.standard_normal(n)
    chain = {frozenset((v - 1, v)) for v in range(1, p)}
    mi = MFCF(gain_function_type="sumsquares", max_clique_size=2).fit(
        X, similarity="mutual_information", mi_estimator="ksg")
    assert {frozenset(q) for q in mi.cliques_} == chain
    corr = MFCF(gain_function_type="sumsquares", max_clique_size=2).fit(X)
    assert {frozenset(q) for q in corr.cliques_} != chain


def test_mfcflogo_mutual_information_interfaces():
    from mfcf_logo import MFCFLoGo
    rng = np.random.default_rng(13)
    X = rng.standard_normal((300, 7)) @ rng.standard_normal((7, 7))
    a = MFCFLoGo(gain_function_type="sumsquares", similarity="mutual_information",
                 mi_estimator="histogram").fit(X)
    assert np.all(np.isfinite(a.precision_)) and a.get_params()["similarity"] == "mutual_information"
    # interface of the mutual_info branch: precomputed similarity and covariance
    from mutual_information import mutual_information_matrix
    S = mutual_information_matrix(X, estimator="histogram")
    b = MFCFLoGo(gain_function_type="sumsquares", similarity="mutual_information").fit(
        X, mi_matrix=S, cov_matrix=np.cov(X, rowvar=False, bias=True))
    assert {frozenset(q) for q in a.cliques_} == {frozenset(q) for q in b.cliques_}
    c = MFCFLoGo(gain_function_type="sumsquares").fit(X, corr_matrix=np.corrcoef(X, rowvar=False))
    assert np.all(np.isfinite(c.precision_))


def test_sumsquares_accepts_correlation_coefficients_not_weights():
    S = np.array([[1, .9, .9], [.9, 1, 0.], [.9, 0., 1]])      # valid coefficients, not PSD
    assert np.linalg.eigvalsh(S).min() < 0
    MFCF(gain_function_type="sumsquares", max_clique_size=2).run(S)
    W = np.array([[1, 2., .5], [2., 1, .3], [.5, .3, 1]])       # weights above 1
    with pytest.raises(ValueError, match="sum"):
        MFCF(gain_function_type="sumsquares").run(W)


# ------------------------------------------------- original-code options
def test_drop_sep_uses_each_separator_once():
    R = _sample_corr(4, p=40, n=200)
    m = MFCF(gain_function_type="sumsquares", max_clique_size=4, drop_sep=True)
    assert not m._fast_path_applies()
    cl, seps, peo, J = m.run(R)
    assert max(seps.values()) == 1
    assert sorted(int(v) for v in peo) == list(range(40))
    assert nx.is_chordal(nx.from_numpy_array(_edges(cl, 40).astype(int)))
    base = MFCF(gain_function_type="sumsquares", max_clique_size=4).run(R)[1]
    assert max(base.values()) > 1                                  # the option matters here


def test_tmfg_seed():
    R = _sample_corr(5, p=20, n=200)
    with pytest.raises(ValueError):
        MFCF(gain_function_type="sumsquares", max_clique_size=3, seed="tmfg")
    with pytest.raises(ValueError):
        MFCF(seed="other")
    m = MFCF(gain_function_type="sumsquares", max_clique_size=4, seed="tmfg", n_starts=1)
    cl, _, peo, _ = m.run(R)
    W = R ** 2
    top4 = set(np.argsort(-np.sum(W * (W > W.mean()), axis=1), kind="stable")[:4].tolist())
    assert set(int(v) for v in peo[:4]) == top4
    assert all(len(q) == 4 for q in cl)                            # a TMFG-type graph


def test_original_interface():
    from fast_fast_mfcf import mfcf_control
    R = _sample_corr(6, p=25, n=200)
    ctl = mfcf_control()
    assert ctl["drop_sep"] is True and ctl["method"] == "MFCF"
    ctl.update(max_clique_size=4, drop_sep=False)
    with pytest.warns(DeprecationWarning):
        cl, seps, peo, J = MFCF().fast_mfcf(R, ctl, "sumsquares")
    ref = MFCF(gain_function_type="sumsquares", max_clique_size=4, new_component="first", n_starts=1).run(R)
    assert {frozenset(q) for q in cl} == {frozenset(q) for q in ref[0]} and seps == ref[1]
    with pytest.raises(ValueError), pytest.warns(DeprecationWarning):
        MFCF().fast_mfcf(R, dict(ctl, method="PMFG"), "sumsquares")
