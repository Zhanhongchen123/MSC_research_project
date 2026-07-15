#!/usr/bin/env python3
import sys
import numpy as np
from math import comb
from fractions import Fraction
from collections import Counter
from scipy.signal import periodogram

###################################################
## Periodicity detection following Siegel (1980) ##
###################################################

## Self-contained: the binned periodogram (a standard FFT, identical to the SPH
## baseline), the Fisher threshold g_F, Siegel's compound statistic and Siegel's
## exact null distribution are all defined here.  There is no iterative Fisher
## peeling ("multiple Fisher"), no Monte-Carlo and no BIC.
## Reference: Siegel, A.F. (1980), "Testing for periodicity in a time series",
## JASA 75(370), 345-348.  Procedure (his Sections 2-4):
## 1. normalised periodogram ordinates Y_j = P_j / sum_i P_i, so that sum_j Y_j = 1
## 2. Fisher critical value g_F = 1 - (alpha/n)^{1/(n-1)}; Siegel threshold
##    g = lambda * g_F with lambda = 0.6 (his conservative recommendation;
##    lambda = 1 reduces to Fisher's test)
## 3. compound statistic T_lambda = sum_j (Y_j - lambda*g_F)_+, his eq (3.1)
## 4. overall test: reject the white-noise H0 iff T_lambda > t_lambda, with the
##    critical value from the EXACT null distribution of his eq (4.1),
##       P_H0(T_lambda > t) = sum_{l=1}^{n} sum_{k=0}^{l-1} (-1)^{k+l+1}
##             C(n,l) C(n-1,k) t^k (1 - lambda*l*g_F - t)_+^{n-k-1},
##    the vacancy of n random arcs of length lambda*g_F on the unit circle
##    (Siegel 1978); computable for small n only, since the binomials overflow
##    for n beyond a few hundred, which is why Siegel tabulates n <= 50
## 5. the detected frequencies are Siegel's adaptively-chosen contributing
##    ordinates Y_j > lambda*g_F; adjacent bins (one peak leaking across its
##    neighbours) are merged to their local maximum
## A single deterministic post-step (collapse_harmonics) reduces the contributing
## frequencies to fundamental clocks: a sharp period-P signal necessarily has
## power at P, P/2, P/3, ... (VanderPlas 2018), so each harmonic family is mapped
## to its longest member -- a frequency-ratio rule, not a test or a simulation.


## Periodogram: standard FFT on the binned, mean-centred event counts, at
## resolution delta; returns (fk, Sk, n_time)
def binned_periodogram(t, delta=1.0, full_day=True):
    t = np.asarray(t, dtype=float)
    start = int((np.min(t) // 86400) * 86400) if full_day else int(np.floor(np.min(t)))
    if not full_day:
        t = t - start
        start = 0
    end = int(((np.max(t) // 86400) + 1) * 86400) if full_day else int(np.floor(np.max(t)) + 1)
    grid = np.arange(start, end + 1, step=delta)
    counts = Counter(grid[np.searchsorted(grid, t)])
    dN = np.array([counts[g] for g in grid], dtype=float)
    dN_star = dN - len(t) / len(grid)               ## remove the mean rate
    fk, Sk = periodogram(dN_star)
    return fk, Sk, len(grid)


## Fisher critical value g_F = 1 - (alpha/n)^{1/(n-1)} (Siegel's table)
def fisher_gF(n, alpha):
    return 1.0 - (alpha / n) ** (1.0 / (n - 1))


## Siegel's exact null tail P_H0(T_lambda > t), his eq (4.1)
def siegel_exact_tail(t, n, lam, gF):
    thr = lam * gF
    total = 0.0
    for l in range(1, n + 1):
        base = 1.0 - l * thr - t
        if base <= 0.0:                             ## (.)_+ vanishes from here on
            break
        cn_l = comb(n, l)
        for k in range(0, l):
            total += ((-1) ** (k + l + 1)) * cn_l * comb(n - 1, k) \
                     * (t ** k) * (base ** (n - k - 1))
    return float(total)


## Exact upper-alpha critical value t_lambda, by bisection on eq (4.1)
## Reproduces Siegel's Table 1
def siegel_exact_critical(n, lam=0.6, alpha=0.05, tol=1e-7):
    gF = fisher_gF(n, alpha)
    lo, hi = 0.0, 1.0
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if siegel_exact_tail(mid, n, lam, gF) > alpha:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


## Harmonic collapse: map each harmonic family (P, P/2, P/3, ...) to its
## fundamental (the longest period).  Deterministic frequency-ratio rule --
## no test, no Monte-Carlo
def collapse_harmonics(peaks, qmax=6, rel=0.02, strength_mult=2.0, thr=0.0):
    cand = [p for p in peaks if p["Y"] > strength_mult * thr]
    cand.sort(key=lambda d: -d["period"])           ## fundamental = longest first
    funds = []
    for p in cand:
        harmonic = False
        for f in funds:
            r = f["period"] / p["period"]           ## > 1; harmonic if ~ integer ratio p/q
            fr = Fraction(r).limit_denominator(qmax)
            if (fr.numerator <= qmax and fr.denominator <= qmax
                    and fr.numerator >= 2 and abs(r - float(fr)) / r < rel):
                harmonic = True
                break
        if not harmonic:
            funds.append(p)
    funds.sort(key=lambda d: -d["Y"])               ## strongest first
    return funds


#### Periodicity detection: Siegel's compound test and the harmonic collapse
def siegel_detect_periods(t, delta=1.0, full_day=True, lam=0.6, alpha=0.01,
                          min_cycles=None, group_gap=2, siegel_exact_max=200):
    fk, Sk, n_time = binned_periodogram(t, delta=delta, full_day=full_day)
    Sk = np.array(Sk, dtype=float)
    n = len(Sk)
    Sk[0] = 0.0                                     ## drop the DC ordinate
    Y = Sk / Sk.sum()                               ## normalised ordinates, sum = 1
    gF = fisher_gF(n, alpha)                        ## Fisher critical value g_F
    thr = lam * gF                                  ## Siegel threshold lambda*g_F
    T_lambda = float(np.clip(Y - thr, 0.0, None).sum())          ## eq (3.1)

    ## Overall periodicity test via the exact null (eq 4.1), small n only
    if n <= siegel_exact_max:
        t_crit = siegel_exact_critical(n, lam=lam, alpha=alpha)
        pval = siegel_exact_tail(T_lambda, n, lam, gF)
        periodic = bool(T_lambda > t_crit)
        test_mode = "exact eq(4.1)"
    else:
        t_crit = pval = float("nan")
        periodic = None
        test_mode = "n too large for eq(4.1)"

    ## Contributing frequencies: Y_j > lambda*g_F (Siegel's adaptive set)
    idx = np.where(Y > thr)[0]
    if min_cycles is not None:                      ## require enough cycles in the window
        fmin = min_cycles / (n_time * delta)
        idx = idx[fk[idx] >= fmin]

    ## Merge adjacent contributing bins to their local maximum
    peaks = []
    if idx.size:
        for g in np.split(idx, np.where(np.diff(idx) > group_gap)[0] + 1):
            kb = g[int(np.argmax(Y[g]))]            ## local max of the merged run
            peaks.append({"period": float(delta / fk[kb]), "freq": float(fk[kb]),
                          "kmax": int(kb), "Y": float(Y[kb]),
                          "excess": float(Y[kb] - thr)})
    peaks.sort(key=lambda d: -d["Y"])               ## strongest first
    funds = collapse_harmonics(peaks, thr=thr)      ## -> fundamental clocks

    ## Return output
    return {"T_lambda": T_lambda, "t_crit": t_crit, "pval": pval,
            "periodic": periodic, "test_mode": test_mode,
            "gF": gF, "threshold": thr, "n": n,
            "periods": [p["period"] for p in peaks], "peaks": peaks,
            "fundamentals": [p["period"] for p in funds], "fund_peaks": funds}


## Self-test on the three-clock synthetic generator
if __name__ == "__main__":
    from simulating_data import simulate_3clock
    SIGMA2 = 0.06
    cl = [{"p": 10.0, "n": 480, "mus": [1.2, 2.8, 4.4], "sigma2": SIGMA2},
          {"p": 23.0, "n": 320, "mus": [1.5, 4.3],      "sigma2": SIGMA2},
          {"p": 37.0, "n": 210, "mus": [3.0],           "sigma2": SIGMA2}]
    sd = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    for kind in ("step", "heavisine", "bumps"):
        data = simulate_3clock(cl, {"kind": kind, "n": 240}, n_days=7, seed=sd)
        r = siegel_detect_periods(data["t"], delta=1.0, full_day=True,
                                  lam=0.6, alpha=0.01, min_cycles=60)
        print(f"\n[{kind}]  n={r['n']}  T_lambda={r['T_lambda']:.4f}  "
              f"lam*gF={r['threshold']:.2e}  ({r['test_mode']})")
        print(f"  {len(r['peaks'])} contributing freqs -> "
              f"fundamentals = {sorted(round(p,1) for p in r['fundamentals'])}")
