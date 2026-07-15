#!/usr/bin/env python3
"""
ks_gof.py -- SELF-CONTAINED nonparametric Wold KS goodness-of-fit for event
timestamps.  Implements the Fig.-9 goodness-of-fit of Sanna Passino & Heard
(2020): the nonparametric self-exciting Wold process of Price-Williams & Heard
(2020), estimator code adapted from the authors' reference implementation
(github.com/Matt0312/SToCND, Modelling_Network_Data_git.py).

  * fit the non-increasing Wold step hazard to the waiting times of the first
    `train_days` days;
  * eq.-14 predictive p-values p_i = 1 - exp(-H(d_i)) on the remaining days;
  * KS distance of {p_i} from Uniform(0,1).  SMALLER = cleaner human residual.

Library:   from ks_gof import wold_ks, ks_table
CLI:       python3 ks_gof.py stream.txt [stream2.txt ...] [--train-days 4]
"""

from numpy import log, diff, cumsum, mean, argmin
import numpy as np

####################################################
## Nonparametric Wold KS goodness-of-fit (Fig. 9) ##
####################################################

## The module docstring above doubles as the command-line usage text.
## Estimator functions below follow Price-Williams & Heard (2020), Sect. 5.2.


## Exponential segment cost (eq. 9): -2 times the log-likelihood of the
## spacings in the segment under their mean rate
def C(lst, index=0):
    m = mean(lst)
    if m == 0:
        m = 1
    log_like = [log(1.0 / m) - 1.0 / m * t for t in lst]
    return -2 * sum(log_like)


## Greatest-convex-minorant pass giving the candidate changepoints (Sect. 5.2,
## eq. 13): recursively pick the index maximising the slope (i+1-j)/(PP[i]-start)
def step_function(times):
    n = len(times)
    PP = times
    cps = [0]
    rates = []
    while cps[-1] < n:
        j = cps[-1]
        start = 0 if j == 0 else PP[j - 1]
        max_lhd = 0
        l = j
        for i in range(j, len(PP)):
            lhd = (i + 1 - j) / (PP[i] - start) if PP[i] != start else 999999
            if lhd >= max_lhd:
                l = i
                max_lhd = lhd
        cps += [l + 1]
        rates += [max_lhd]
    return [PP[i - 1] for i in cps], rates, cps


## PELT pruning (Sect. 5.2, Step 3) restricted to the GCM candidate steps, using
## the exponential segment cost C (eq. 9) -> parsimonious monotone step intensity
def PELT_MLE(list1, beta):
    possible_steps = step_function(list1)[2]
    possible_steps += [0, len(list1) - 1]
    possible_steps += [min(i for i in range(len(list1)) if list1[i] != 0)]
    possible_steps = sorted(set(possible_steps))

    cp = [[0]]
    R = [0]
    list1 = list(list1) + [0]
    list1.sort()
    F = [-beta]
    D = diff(list1)
    possible_steps.remove(0)

    for i in range(1, len(possible_steps)):
        F_r = [F[j] + C(D[possible_steps[j]:possible_steps[i]]) + beta for j in R]
        tau = argmin(F_r)
        F += [F_r[tau]]
        R = [p for p in range(len(possible_steps)) if possible_steps[p] <= possible_steps[i]]
        cp += [cp[R[tau]] + [possible_steps[R[tau]]]]
    return cp[-1]


## Fit the non-increasing Wold step hazard to the event times (Sect. 5.2)
## Returns (cutoffs, rates): hazard = rates[k] on (cutoffs[k-1], cutoffs[k]]
def fit_step_hazard(times):
    dy = diff(np.sort(np.asarray(times, dtype=float)))
    beta = 2 * log(len(times))
    sdy = [0] + sorted(dy)
    n = len(sdy)
    Di = list(diff(sdy))
    P = [Di[i] * (n - i) for i in range(len(Di))]      ## delta_i (exponential spacings)
    PP = list(cumsum(P))                                ## Delta_i (rescaled times)
    cps = PELT_MLE(PP, beta)
    cps += [n]
    rates = []
    for i in range(len(cps) - 1):
        seg = sum(P[cps[i]:cps[i + 1]])
        rates += [(cps[i + 1] - cps[i]) / seg if seg > 0 else 0.0]
    cps.remove(0)
    if n in cps:
        cps.remove(n)
    cutoffs = [sdy[i - 1] for i in cps]
    return cutoffs, rates


## eq.-14 predictive p-value p = 1 - exp(-H(x)) for a test waiting time x,
## where H is the cumulative step hazard given by (cutoffs, rates)
def pvalue_WS(x, cutoffs, rates):
    from numpy import exp
    C_ = 0.0
    T = 1.0
    if len(cutoffs) == 0:
        return 1 - exp(-rates[0] * x)
    if x < cutoffs[0]:
        return 1 - exp(-rates[0] * x)
    T *= exp(-rates[0] * cutoffs[0])
    C_ += 1 - exp(-rates[0] * cutoffs[0])
    for i in range(1, len(cutoffs)):
        if x > cutoffs[i]:
            C_ += T * (1 - exp(-rates[i] * (cutoffs[i] - cutoffs[i - 1])))
            T *= exp(-rates[i] * (cutoffs[i] - cutoffs[i - 1]))
        else:
            C_ += T * (1 - exp(-rates[i] * (x - cutoffs[i - 1])))
            return C_
    C_ += T * (1 - exp(-rates[-1] * (x - cutoffs[-1])))
    return C_



import sys
import numpy as np
from scipy import stats

DAY = 86400.0


#### Wold KS
## KS of the eq.-14 Wold predictive p-values against Uniform(0,1), training the
## step hazard on the first train_days days of `times` and testing on the rest.
## Returns a dict: ks, n_test (number of test waiting times), p (the p-values),
## ncp (number of hazard changepoints), train_n (number of training events).
## Returns ks=nan when either side has fewer than min_n events.
def wold_ks(times, train_days=4.0, min_n=60):
    t = np.array(sorted(set(np.asarray(times, dtype=float))))     ## dedupe (continuous model)
    out = {"ks": float("nan"), "n_test": 0, "p": np.array([]), "ncp": 0, "train_n": 0}
    if len(t) < min_n:
        return out
    split = t[0] + train_days * DAY
    tr = t[t <= split]
    te_idx = np.where(t > split)[0]
    te_idx = te_idx[te_idx > 0]
    if len(tr) < min_n or len(te_idx) < min_n:
        return out
    cut, rate = fit_step_hazard(tr)
    p = np.array([pvalue_WS(t[i] - t[i - 1], cut, rate) for i in te_idx])
    p = np.clip(p[np.isfinite(p)], 0.0, 1.0)
    if len(p) == 0:
        return out
    out.update(ks=float(stats.kstest(p, "uniform").statistic),
               n_test=len(p), p=p, ncp=len(cut), train_n=len(tr))
    return out


## Compute and print the Wold KS for each named stream {label: timestamps}
## Smaller KS = better (human-like) fit
def ks_table(named_streams, train_days=4.0, min_n=60, title="nonparametric Wold KS"):
    res = {k: wold_ks(v, train_days=train_days, min_n=min_n) for k, v in named_streams.items()}
    w = max(len(k) for k in res) if res else 6
    print(f"{title}  (train = first {train_days:g} days; smaller = cleaner human residual)")
    print(f"  {'stream':<{w}}{'n':>9}{'KS':>10}{'n_test':>9}{'ncp':>6}")
    for k, r in res.items():
        ks = f"{r['ks']:.4f}" if np.isfinite(r["ks"]) else "  nan"
        n = r["train_n"] + r["n_test"]
        print(f"  {k:<{w}}{n:>9}{ks:>10}{r['n_test']:>9}{r['ncp']:>6}")
    return res


## Command-line interface: print the KS table for the given stream files
def _main(argv):
    files = [a for a in argv if not a.startswith("--")]
    train_days = 4.0
    for a in argv:
        if a.startswith("--train-days"):
            train_days = float(a.split("=", 1)[1]) if "=" in a else float(argv[argv.index(a) + 1])
    if not files:
        print(__doc__); return
    streams = {}
    for f in files:
        try:
            streams[f] = np.loadtxt(f)
        except Exception as e:
            print(f"  ! could not read {f}: {e}")
    if streams:
        ks_table(streams, train_days=train_days)


if __name__ == "__main__":
    _main(sys.argv[1:])
