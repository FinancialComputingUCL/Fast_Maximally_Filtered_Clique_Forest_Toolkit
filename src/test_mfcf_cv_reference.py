"""Reference check of the cross-validated estimators.

MFCFLoGoCV and MFCFLoGoCVAll are compared with an independent, plain
re-implementation of their specification:

* folds: sklearn KFold(cv, shuffle, random_state if shuffle);
* each fold: MFCFLoGo with the candidate parameters fitted on the training
  part (n_starts capped at the clique size; a numeric n_effective scaled to
  the training size);
* score: Gaussian log-likelihood of the validation fold,
  log_likelihood(empirical_covariance(X_val), precision) (the convention of
  sklearn's GraphicalLassoCV);
* selection: the largest mean score among candidates whose folds all
  succeeded; refit on all data with the selected parameters.

The comparison is exact (same floating-point operations), so any deviation
in folds, scoring, selection or refit shows up.
"""
import warnings

import numpy as np
import pytest
from sklearn.covariance import empirical_covariance, log_likelihood
from sklearn.model_selection import KFold

from mfcf_logo import MFCFLoGo, MFCFLoGoCV, MFCFLoGoCVAll


def _data(n, p, seed):
    rng = np.random.default_rng(seed)
    k = max(1, p // 4)
    return rng.standard_normal((n, k)) @ rng.standard_normal((k, p)) + rng.standard_normal((n, p))


def _reference_fold_scores(X, params, cv, shuffle, random_state, n_train_for_neff=None):
    """Mean and per-fold validation log-likelihoods of MFCFLoGo(**params)."""
    kf = KFold(n_splits=cv, shuffle=shuffle, random_state=random_state if shuffle else None)
    n = X.shape[0]
    scores = []
    for tr, va in kf.split(X):
        kw = dict(params)
        if kw.get("n_starts") is not None:
            kw["n_starts"] = min(kw["n_starts"], kw["max_clique_size"])
        ne = kw.get("n_effective")
        if ne is not None and not isinstance(ne, str):
            n_train = len(tr) if n_train_for_neff is None else n_train_for_neff
            kw["n_effective"] = max(3.0, float(ne) * n_train / n)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                est = MFCFLoGo(**kw).fit(X[tr])
            S_val = empirical_covariance(X[va], assume_centered=kw.get("assume_centered", False))
            scores.append(log_likelihood(S_val, est.precision_))
        except ValueError:
            scores.append(np.nan)
    return np.array(scores)


OPTIONS = [
    dict(gain_function_type="loglik"),
    dict(gain_function_type="loglik", acceptance="ftest", alpha=0.05),
    dict(gain_function_type="loglik", acceptance="bic", shrinkage=0.1),
    dict(gain_function_type="loglik", threshold=0.05, n_starts=2),
    dict(gain_function_type="loglik", acceptance="ftest", n_effective=40),
    dict(gain_function_type="sumsquares", threshold=0.0),
    dict(gain_function_type="loglik", correlation="spearman", shrinkage=0.05),
    dict(gain_function_type="loglik", n_starts=4),             # cap binds for k = 2, 3
]
SPLITS = [dict(cv=3, shuffle=False, random_state=None), dict(cv=5, shuffle=True, random_state=4)]
DATA = [(40, 8, 0), (90, 12, 1), (200, 10, 2)]


@pytest.mark.parametrize("opts", OPTIONS, ids=lambda o: "-".join(f"{k}={v}" for k, v in o.items()))
@pytest.mark.parametrize("split", SPLITS, ids=lambda s: f"cv{s['cv']}-shuffle{s['shuffle']}")
@pytest.mark.parametrize("data", DATA, ids=lambda d: f"n{d[0]}-p{d[1]}")
def test_cv_matches_reference(opts, split, data):
    n, p, seed = data
    X = _data(n, p, seed)
    grid = [2, 3, 4, 6]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cv = MFCFLoGoCV(max_clique_size_grid=grid, **split, **opts).fit(X)
    # fold scores, candidate by candidate
    ref = {k: _reference_fold_scores(X, dict(opts, max_clique_size=k), **split) for k in grid}
    for k in grid:
        got = np.array([r.score if r.success else np.nan for r in cv.fold_scores_ if r.max_clique_size == k])
        np.testing.assert_allclose(got, ref[k], rtol=0, atol=1e-10, equal_nan=True)
    # selection: best mean among candidates with no failed fold
    eligible = {k: v.mean() for k, v in ref.items() if np.all(np.isfinite(v))}
    best = max(eligible, key=eligible.get)
    assert cv.best_max_clique_size_ == best
    # refit on all data with the selected size
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        refit = dict(opts, max_clique_size=best)
        if refit.get("n_starts") is not None:
            refit["n_starts"] = min(refit["n_starts"], best)
        final = MFCFLoGo(**refit).fit(X)
    np.testing.assert_allclose(cv.precision_, final.precision_, rtol=0, atol=1e-12)


@pytest.mark.parametrize("tunable", [("max_clique_size",), ("max_clique_size", "threshold"),
                                     ("alpha", "max_clique_size")])
@pytest.mark.parametrize("data", DATA[:2], ids=lambda d: f"n{d[0]}-p{d[1]}")
def test_cvall_matches_reference(tunable, data):
    optuna = pytest.importorskip("optuna")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    n, p, seed = data
    X = _data(n, p, seed)
    fixed = dict(gain_function_type="loglik", max_clique_size=4)
    if "alpha" in tunable:
        fixed["acceptance"] = "ftest"
    kw = dict(tunable_params=tunable, n_trials=10, cv=4, shuffle=True, random_state=3, **fixed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = MFCFLoGoCVAll(**kw).fit(X)
        m2 = MFCFLoGoCVAll(**kw).fit(X)
    assert m.cv_results_["params"] == m2.cv_results_["params"]     # reproducible search
    np.testing.assert_array_equal(m.cv_results_["mean_test_score"], m2.cv_results_["mean_test_score"])
    # every completed trial: its score is the reference mean CV score
    n_train_min = n - -(-n // 4)
    trials = [dict(params=p_, mean_test_score=v_, state=s_) for p_, v_, s_ in
              zip(m.cv_results_["params"], m.cv_results_["mean_test_score"], m.cv_results_["state"])]
    complete = [t for t in trials if t["state"] == "COMPLETE"]
    assert complete
    for t in complete:
        params = dict(fixed, **t["params"])
        ref = _reference_fold_scores(X, params, cv=4, shuffle=True, random_state=3,
                                     n_train_for_neff=n_train_min)
        assert np.all(np.isfinite(ref))
        assert t["mean_test_score"] == pytest.approx(ref.mean(), abs=1e-10)
    # selection and refit
    best = max(complete, key=lambda t: t["mean_test_score"])
    assert m.best_params_ == best["params"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        final = MFCFLoGo(**dict(fixed, **best["params"])).fit(X)
    np.testing.assert_allclose(m.precision_, final.precision_, rtol=0, atol=1e-12)


def test_cv_is_deterministic():
    X = _data(120, 10, 5)
    a = MFCFLoGoCV(gain_function_type="loglik", max_clique_size_grid=[2, 3, 5], cv=5,
                   shuffle=True, random_state=1).fit(X)
    b = MFCFLoGoCV(gain_function_type="loglik", max_clique_size_grid=[2, 3, 5], cv=5,
                   shuffle=True, random_state=1).fit(X)
    assert a.best_max_clique_size_ == b.best_max_clique_size_
    np.testing.assert_array_equal(a.cv_results_["mean_test_score"], b.cv_results_["mean_test_score"])
    np.testing.assert_array_equal(a.precision_, b.precision_)


@pytest.mark.parametrize("error_score", [np.nan, 0.0, -1e6])
def test_cv_with_failing_folds_matches_reference(error_score):
    """Small n: large clique sizes fail in every or some folds. A finite
    error_score must not make a failed size eligible."""
    X = _data(14, 10, 6)
    grid = [2, 3, 6, 10]
    split = dict(cv=3, shuffle=True, random_state=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cv = MFCFLoGoCV(gain_function_type="loglik", max_clique_size_grid=grid,
                        error_score=error_score, **split).fit(X)
    ref = {k: _reference_fold_scores(X, dict(gain_function_type="loglik", max_clique_size=k), **split)
           for k in grid}
    assert any(not np.all(np.isfinite(v)) for v in ref.values())          # failures do occur
    for k in grid:
        recs = [r for r in cv.fold_scores_ if r.max_clique_size == k]
        np.testing.assert_array_equal([not r.success for r in recs], ~np.isfinite(ref[k]))
    eligible = {k: v.mean() for k, v in ref.items() if np.all(np.isfinite(v))}
    assert cv.best_max_clique_size_ == max(eligible, key=eligible.get)


def test_cvall_with_failing_trials_matches_reference():
    optuna = pytest.importorskip("optuna")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    X = _data(17, 14, 7)                       # folds of unequal size: partial failures
    fixed = dict(gain_function_type="loglik", max_clique_size=4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = MFCFLoGoCVAll(tunable_params=("max_clique_size",), n_trials=25, cv=4, shuffle=True,
                          random_state=2, error_score=0.0, **fixed).fit(X)
    n_train_min = 17 - -(-17 // 4)
    trials = [dict(params=p_, mean_test_score=v_, state=s_) for p_, v_, s_ in
              zip(m.cv_results_["params"], m.cv_results_["mean_test_score"], m.cv_results_["state"])]
    states = {t["state"] for t in trials}
    assert "FAIL" in states and "COMPLETE" in states                       # failures do occur
    partial = [t for t in trials
               if 0 < np.isnan(_reference_fold_scores(X, dict(fixed, **t["params"]), cv=4, shuffle=True,
                                                      random_state=2, n_train_for_neff=n_train_min)).sum() < 4]
    assert partial                                                          # some folds only
    for t in trials:
        ref = _reference_fold_scores(X, dict(fixed, **t["params"]), cv=4, shuffle=True,
                                     random_state=2, n_train_for_neff=n_train_min)
        assert (t["state"] == "COMPLETE") == bool(np.all(np.isfinite(ref)))
    best = max((t for t in trials if t["state"] == "COMPLETE"), key=lambda t: t["mean_test_score"])
    assert m.best_params_ == best["params"]
