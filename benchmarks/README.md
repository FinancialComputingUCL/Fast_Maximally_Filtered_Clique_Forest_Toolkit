# Scaling benchmark

`python benchmarks/bench_scaling.py` times `MFCF.run` on a p x p correlation
matrix (5-factor data, n = 4000) for p = 100 ... 1600 and K = 4, 8, 16,
single core (`OMP_NUM_THREADS=1`), and records peak memory (tracemalloc).
All timings are for a single start (`n_starts=1`). With the `"loglik"` gain
the default is `n_starts=max_clique_size`, which multiplies the loglik times
by K.
Results of the run below: `bench_scaling.json` (Apple M-series, Python 3.12).

## Runtimes (seconds)

| Setting | K | p=100 | 200 | 400 | 800 | 1600 |
|---|---|---|---|---|---|---|
| loglik, fixed size | 4 | 0.02 | 0.06 | 0.14 | 0.48 | 1.6 |
| | 8 | 0.04 | 0.11 | 0.37 | 1.05 | 3.6 |
| | 16 | 0.12 | 0.36 | 1.27 | 4.3 | 16.0 |
| loglik, F-test | 4 | 0.19 | 0.47 | 1.1 | 3.0 | 8.3 |
| | 8 | 0.75 | 2.1 | 5.5 | 15.8 | 47.5 |
| | 16 | 0.24 | 2.1 | 8.2 | 29.3 | 91.7 |
| sumsquares, fixed size | 8 | 0.08 | 0.17 | 0.53 | 1.6 | 5.5 |
| sumsquares, threshold 0.05 | 8 | 0.03 | 0.08 | 0.24 | 0.74 | 2.3 |

## Empirical scaling (fit on p >= 200)

| Setting | time ~ p^a | time ~ K^b | memory ~ p^c | peak memory, p=1600 |
|---|---|---|---|---|
| loglik, fixed size | a = 1.69 | b = 1.53 | c = 1.34 | 69 MB |
| loglik, F-test | 1.56 | 1.48 | 1.37 | 121 MB |
| sumsquares, fixed size | 1.63 | 1.62 | 1.44 | 85 MB |
| sumsquares, threshold | 1.87 | 0.82 | 1.66 | 85 MB |

The exponent in p rises towards 2 with K (about 1.6 at K = 4, 1.8 at
K = 16), consistent with O(p^2 K^2) local evaluations. Memory is dominated by
the input matrix (the gain cache keeps 32 candidates per separator).
Extrapolated single-start runs at K = 8: about 30 s (loglik, fixed) and 3 min
(F-test) at p = 5000; about 1.5 min and 10 min at p = 10000, where the input
matrix alone takes 800 MB.
