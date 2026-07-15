#!/usr/bin/env python3
import sys, time
import numpy as np
from simulating_data import simulate_3clock
from mcmc_improve import gibbs_rj_variable

########################################################################
## Timing of the sampler: seconds per 1000 sweeps as a function of N ##
########################################################################

## Measures, in the manner of the timing table of the report (Table 1 of
## SPH), the elapsed time for 1000 sweeps of the extended sampler
## (single chain, single core; SAMPLING LOOP ONLY, excluding initialisation)
## as a function of the number of events N, on the canonical three-clock
## benchmark scaled to N events (K = 3, R_true = (3, 2, 1), full reallocation
## sweeps -- the library default and the algorithm as documented).
## The initialisation is excluded by a two-point measurement: the sampler is
## run twice on the same data with S1 and S2 total sweeps and otherwise
## identical settings, and the per-sweep cost is the slope
## (T2 - T1)/(S2 - S1); every fixed per-call cost (the EM initialisation,
## the density grids) cancels in the difference.
## Absolute times are hardware-dependent and grow approximately linearly in
## N; with full reallocation sweeps the cost is dominated by the per-event
## (K+1)-way allocation, so the largest cells take HOURS on one core.
## Usage:  python3 timing_table.py [--quick] [N ...]
##         default N = 100 1000 5000 10000 25000 50000; --quick uses N = 100 1000


## Canonical three-clock benchmark scaled to N events
## (clock/human proportions 480 : 320 : 210 : 240, as in test_model)
def scaled_data(N, seed=0):
    base = np.array([480, 320, 210, 240], dtype=float)
    n = np.floor(N * base / base.sum()).astype(int)
    n[0] += N - n.sum()                              ## rounding remainder to clock 1
    cl = [{"p": 10.0, "n": int(n[0]), "mus": [1.2, 2.8, 4.4], "sigma2": 0.06},
          {"p": 23.0, "n": int(n[1]), "mus": [1.5, 4.3],      "sigma2": 0.06},
          {"p": 37.0, "n": int(n[2]), "mus": [3.0],           "sigma2": 0.06}]
    return simulate_3clock(cl, {"kind": "step", "n": int(n[3])}, n_days=7, seed=seed)


## One timed call of S total sweeps (burnin=0, so every sweep is in the loop)
def timed_call(t, S, init_iter=1500):
    t0 = time.perf_counter()
    gibbs_rj_variable(t, [10.0, 23.0, 37.0], Rmax=6, R_init=4, n_samp=S,
                      n_chains=1, burnin=0, zeta0=1.0, seed=11, verbose=False,
                      init_iter=init_iter)
    return time.perf_counter() - t0


#### Timing table
def timing_table(Ns, S1=20, S2=120):
    print(f"{'N':>8s} {'s / 1000 sweeps':>16s}    "
          f"(slope between S = {S1} and {S2} sweeps; sampling loop only)")
    for N in Ns:
        t = scaled_data(N)["t"]
        T1 = timed_call(t, S1)
        T2 = timed_call(t, S2)
        per1000 = (T2 - T1) / (S2 - S1) * 1000.0
        print(f"{len(t):>8d} {per1000:>16.1f}", flush=True)


## Command-line interface
if __name__ == "__main__":
    quick = "--quick" in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    Ns = [int(a) for a in args] if args else ([100, 1000] if quick else
                                              [100, 1000, 5000, 10000, 25000, 50000])
    timing_table(Ns)
