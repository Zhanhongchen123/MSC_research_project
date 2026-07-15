#!/usr/bin/env python3
import numpy as np

#######################################################################
## Synthetic data generators for the extension (faithful to SPH 6.1) ##
#######################################################################

## Follows Sanna Passino & Heard (2020) Sect. 6.1 for BOTH components and adds
## the multi-period / phase-shift structure of the extension.
## What it reproduces from the paper (eq. references to SPH 2020):
## - HUMAN events: a non-trivial daily density on [0, 2*pi) -- one of the three
##   Donoho-Johnstone (1994) test signals the paper uses (the step function of
##   eq. (18), the heavisine and the 11-bump function), then assigned to a
##   RANDOM DAY (p' = 86400 s).  Not uniform, so that the time-of-day model
##   s(y) has real structure to recover and second-period leakage is
##   distinguishable from genuine human activity.
## - PERIODIC events: x ~ WN(mu_k, sigma2_k) on the p_k-clock, eq. (4)/(6),
##   assigned at random to windows of p_k seconds (the paper's recipe), with
##   the paper's deliberately-large sigma2 = 1 to make inference hard.
## What it adds (the two sweeps and the limitation demonstration):
## - K independent periodic clocks (tests the multi-period extension)
## - an optional PHASE SHIFT: a clock's mean jumps mu1 -> mu2 mid-observation,
##   which exercises R and the two-peak shared-variance phase mixture
## - sweep_second_period_fraction(): vary the 2nd-clock share 0% -> 50%
## - sweep_phase_gap():              vary the phase jump |mu2 - mu1| 0 -> pi
## Ground truth returned per event: the component of origin (0 = human,
## k = clock k), the binary auto/human label z_true, and the phase segment
## r_true (0/1 for a shifted clock, -1 for human).  This supports BOTH
## classification metrics (AUC/FPR/FNR) and structural recovery (inferred K,
## period values, phase behaviour).

TWO_PI = 2.0 * np.pi
DAY = 86400

## Standard Donoho & Johnstone (1994) "Bumps" parameters
_BUMPS_POS = np.array([0.10, 0.13, 0.15, 0.23, 0.25, 0.40, 0.44, 0.65, 0.76, 0.78, 0.81])
_BUMPS_HGT = np.array([4.0, 5.0, 3.0, 4.0, 5.0, 4.2, 2.1, 4.3, 3.1, 5.1, 4.2])
_BUMPS_WID = np.array([0.005, 0.005, 0.006, 0.010, 0.010, 0.030, 0.010, 0.010, 0.005, 0.008, 0.005])


### Human daily density (paper Sect. 6.1: step / heavisine / bumps on [0,2pi))

## Daily density evaluated on a grid over [0, 2*pi)
## kind in {step, heavisine, bumps, uniform}
def daily_density_on_grid(kind, rng, n_grid=8000):
    y = np.linspace(0.0, TWO_PI, n_grid, endpoint=False)
    u = y / TWO_PI                                    ## map to [0,1) for the DJ signals

    if kind == "step":
        ## 10-segment circular step density: 10 changepoints ~ U[0,2pi), heights ~ Dir(1..1)
        cps = np.sort(rng.uniform(0.0, TWO_PI, 10))
        heights = rng.dirichlet(np.ones(10))
        arc_len = np.empty(10)
        arc_len[0] = (TWO_PI - cps[-1]) + cps[0]       ## the wraparound arc
        arc_len[1:] = np.diff(cps)
        level = heights / arc_len                      ## density height per arc
        seg = np.searchsorted(cps, y, side="right")    ## 0..10
        seg[seg == 10] = 0                             ## y >= last cp -> wrap arc 0
        f = level[seg]
    elif kind == "heavisine":
        f = 6.0 + 4.0 * np.sin(2.0 * y) - np.sign(u - 0.3) - np.sign(0.72 - u)
        f = np.clip(f, 0.0, None)
    elif kind == "bumps":
        f = np.zeros_like(y)
        for h, v, w in zip(_BUMPS_HGT, _BUMPS_POS, _BUMPS_WID):
            f += h * (1.0 + np.abs((u - v) / w)) ** (-4)
    elif kind == "uniform":
        f = np.ones_like(y)
    else:
        raise ValueError(f"unknown human density kind: {kind!r}")

    f = np.clip(f, 1e-12, None)
    f /= f.sum()                                       ## discrete normalisation (for the CDF)
    return y, f


## Draw n samples of the time-of-day y in [0,2pi) from the chosen daily density
## Inverse-CDF sampling on a fine grid with within-cell jitter (continuous);
## returns (samples, (grid_y, pdf)) -- the pdf is kept for truth and plots
def sample_daily(kind, n, rng, n_grid=8000):
    y, f = daily_density_on_grid(kind, rng, n_grid)
    cdf = np.cumsum(f)
    cdf /= cdf[-1]
    idx = np.searchsorted(cdf, rng.uniform(0.0, 1.0, n))
    idx = np.clip(idx, 0, n_grid - 1)
    dy = TWO_PI / n_grid
    samples = (y[idx] + rng.uniform(0.0, dy, n)) % TWO_PI
    return samples, (y, f)


### One periodic clock (paper Sect. 6.1: WN on the p-clock, random p-windows)

## Generate n periodic event times on a p-clock
## mu_spec: float -> static mean (paper-style single phase);
## (mu_before, mu_after, frac) -> the mean JUMPS mid-observation at frac*T
## (phase shift; exercises R=2).  Returns (times, r_seg) where r_seg in {0,1}
## marks pre/post the shift (0 if static)
def _periodic_clock(p, n, mu_spec, sigma2, T, rng):
    win = rng.integers(0, int(T // p), n)              ## which p-window each event lands in
    win_start = win.astype(float) * p
    if np.isscalar(mu_spec):
        mu_arr = np.full(n, float(mu_spec))
        r_seg = np.zeros(n, dtype="i")
    else:
        mu_before, mu_after, frac = mu_spec
        pre = win_start < (frac * T)
        mu_arr = np.where(pre, mu_before, mu_after)
        r_seg = np.where(pre, 0, 1).astype("i")
    x = (rng.normal(mu_arr, np.sqrt(sigma2))) % TWO_PI  ## x ~ WN(mu, sigma2)  (eq. 6)
    times = win_start + x / TWO_PI * p                  ## back to seconds (eq. 4 inverse)
    return times, r_seg


### Assemble a full multi-period edge

#### Simulate one synthetic NetFlow edge with K periodic clocks and a human stream
## clocks is a list of dicts {p, n, mu, sigma2}, where mu is a float (static)
## OR (mu1, mu2, frac) for a mid-window phase shift; human is {kind, n} with
## kind in {step, heavisine, bumps, uniform}.
## Returns a dict with t (sorted event times in seconds), the ground truth
## comp (component of origin per event: 0 = human, k = clock k), z_true
## (1 = automated, 0 = human) and r_true (phase segment 0/1 for shifted
## clocks, -1 for human), plus periods, sigma2, mu_true, human_density
## (grid, pdf), T and n_days
def simulate_multiperiod(clocks, human, n_days=7, seed=0):
    rng = np.random.default_rng(seed)
    T = n_days * DAY
    times, comp, rseg = [], [], []

    ## Generate each periodic clock
    for k, cl in enumerate(clocks, start=1):
        tk, rk = _periodic_clock(cl["p"], cl["n"], cl["mu"], cl["sigma2"], T, rng)
        times.append(tk)
        comp.append(np.full(len(tk), k, dtype="i"))
        rseg.append(rk)

    ## Generate the human stream on random days
    yk, dens = sample_daily(human["kind"], human["n"], rng)
    day = rng.integers(0, n_days, human["n"])
    th = day.astype(float) * DAY + yk / TWO_PI * DAY           ## p' = 86400 (paper)
    times.append(th)
    comp.append(np.zeros(human["n"], dtype="i"))
    rseg.append(np.full(human["n"], -1, dtype="i"))

    ## Merge, sort and return with the ground truth
    t = np.concatenate(times)
    comp = np.concatenate(comp)
    rseg = np.concatenate(rseg)
    o = np.argsort(t)
    return {"t": t[o], "comp": comp[o], "z_true": (comp[o] > 0).astype(int),
            "r_true": rseg[o], "periods": [cl["p"] for cl in clocks],
            "sigma2": [cl["sigma2"] for cl in clocks],
            "mu_true": [cl["mu"] for cl in clocks],
            "human_density": dens, "T": T, "n_days": n_days}


### The two sweeps

## Sweep 1: vary the SHARE of 2nd-clock events from 0 to ~50% (frac in [0,0.5])
## At frac=0 the data are single-period (the original model should suffice); as
## frac grows a single-period model increasingly misclassifies clock-2 events
## as human -> the multi-period model should pull ahead
def sweep_second_period_fraction(fracs, p1=10.0, p2=23.0, mu1=5.0, mu2=2.0,
                                  sigma2=1.0, n_total=900, human_kind="step",
                                  n_human=400, n_days=7, seed=0):
    out = []
    for f in fracs:
        n2 = int(round(f * n_total))
        n1 = n_total - n2
        clocks = [{"p": p1, "n": n1, "mu": mu1, "sigma2": sigma2}]
        if n2 > 0:
            clocks.append({"p": p2, "n": n2, "mu": mu2, "sigma2": sigma2})
        out.append((f, simulate_multiperiod(clocks, {"kind": human_kind, "n": n_human},
                                             n_days=n_days, seed=seed)))
    return out


## Sweep 2: ONE clock whose mean jumps by `gap` mid-window (gap in [0, pi])
## At gap=0 there is no shift (a single phase; R=1 should suffice); as gap
## grows the single-peak model fits one peak across two clusters and inflates
## sigma2 -> the two-peak phase mixture (R=2) should pull ahead
def sweep_phase_gap(gaps, p=10.0, mu1=2.0, sigma2=1.0, n_auto=900,
                    human_kind="step", n_human=400, n_days=7, frac=0.5, seed=0):
    out = []
    for g in gaps:
        clocks = [{"p": p, "n": n_auto, "mu": (mu1, (mu1 + g) % TWO_PI, frac),
                   "sigma2": sigma2}]
        out.append((g, simulate_multiperiod(clocks, {"kind": human_kind, "n": n_human},
                                             n_days=n_days, seed=seed)))
    return out


### Three-clock / R-phase generator

## Each clock cycles its wrapped-normal mean over R equal time segments; the
## human background comes from sample_daily above
TWO_PI = 2.0 * np.pi
DAY = 86400


## n events on a p-clock whose mean cycles through `mus` over equal time
## segments; len(mus) = R_true.  Returns (times, r_seg in {0,...,R-1})
def clock_phases(p, n, mus, sigma2, T, rng):
    mus = np.atleast_1d(np.asarray(mus, float))
    R = len(mus)
    win = rng.integers(0, int(T // p), n)
    win_start = win.astype(float) * p
    seg = np.clip((win_start / (T / R)).astype(int), 0, R - 1)
    mu_arr = mus[seg]
    x = rng.normal(mu_arr, np.sqrt(sigma2)) % TWO_PI
    times = win_start + x / TWO_PI * p
    return times, seg.astype("i")


## Assemble K cycling clocks and a human stream
## clocks: list of {p, n, mus (list), sigma2};  human: {kind, n}
def simulate_3clock(clocks, human, n_days=7, seed=0):
    rng = np.random.default_rng(seed)
    T = n_days * DAY
    times, comp, rseg = [], [], []
    for k, cl in enumerate(clocks, start=1):
        tk, rk = clock_phases(cl["p"], cl["n"], cl["mus"], cl["sigma2"], T, rng)
        times.append(tk); comp.append(np.full(len(tk), k, "i")); rseg.append(rk)
    yk, dens = sample_daily(human["kind"], human["n"], rng)
    day = rng.integers(0, n_days, human["n"])
    th = day.astype(float) * DAY + yk / TWO_PI * DAY
    times.append(th); comp.append(np.zeros(human["n"], "i"))
    rseg.append(np.full(human["n"], -1, "i"))
    t = np.concatenate(times); comp = np.concatenate(comp); rseg = np.concatenate(rseg)
    o = np.argsort(t)
    return {"t": t[o], "comp": comp[o], "z_true": (comp[o] > 0).astype(int),
            "r_true": rseg[o], "periods": [c["p"] for c in clocks],
            "R_true": [len(c["mus"]) for c in clocks], "T": T}


## Default 3-clock 3-phase design: R_true = (3, 2, 1)
CLOCKS = [
    {"p": 10.0, "n": 700, "mus": [1.0, 3.1, 5.2], "sigma2": 0.06},  ## R_true=3
    {"p": 23.0, "n": 450, "mus": [1.5, 4.0],      "sigma2": 0.06},  ## R_true=2
    {"p": 37.0, "n": 300, "mus": [3.0],           "sigma2": 0.06},  ## R_true=1
]
HUMAN = {"kind": "step", "n": 120}
