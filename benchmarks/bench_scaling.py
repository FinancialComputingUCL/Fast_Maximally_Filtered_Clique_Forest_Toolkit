"""Scaling benchmark of MFCF.run: time (and peak memory) over p and K."""
import json, sys, time, tracemalloc
import numpy as np
sys.path.insert(0, sys.argv[1] if len(sys.argv) > 1 else "src")
from fast_fast_mfcf import MFCF
out = sys.argv[2] if len(sys.argv) > 2 else "benchmarks/bench_scaling.json"
SETTINGS = {
    "loglik fixed":        lambda K, n: dict(min_clique_size=K, max_clique_size=K),
    "loglik F-test":       lambda K, n: dict(max_clique_size=K, acceptance="ftest", alpha=0.01, n_samples=n, n_starts=1),
    "sumsquares fixed":    lambda K, n: dict(min_clique_size=K, max_clique_size=K, gain_function_type="sumsquares"),
    "sumsquares thr 0.05": lambda K, n: dict(max_clique_size=K, threshold=0.05, gain_function_type="sumsquares"),
}
rng = np.random.default_rng(0); n = 4000
rows = []
for p in (100, 200, 400, 800, 1600):
    X = rng.standard_normal((n, 10)) @ rng.standard_normal((10, p)) + rng.standard_normal((n, p))
    C = np.corrcoef(X, rowvar=False)
    for K in (4, 8, 16):
        for name, kw in SETTINGS.items():
            m = MFCF(**kw(K, n))
            t = time.perf_counter(); m.run(C, cov_matrix=np.eye(p)); dt = time.perf_counter() - t
            mem = None
            if K == 8 or p <= 400:
                tracemalloc.start(); MFCF(**kw(K, n)).run(C, cov_matrix=np.eye(p))
                mem = tracemalloc.get_traced_memory()[1] / 1e6; tracemalloc.stop()
            rows.append(dict(p=p, K=K, setting=name, seconds=dt, peak_mb=mem))
            print(f"{p:5d} {K:3d} {name:22s} {dt:9.3f} s  {'' if mem is None else f'{mem:8.1f} MB'}", flush=True)
            json.dump(rows, open(out, "w"), indent=1)
