#!/usr/bin/env python3
import os, sys, time, contextlib
import numpy as np
from simulating_data import simulate_3clock
from siegel_periods import siegel_detect_periods
from mcmc_original import collapsed_gibbs
from mcmc_improve import gibbs_rj_variable

###########################################################################
## End-to-end test of the full model on the three synthetic multi-period ##
###########################################################################

## Usage:  python3 test_model.py [SEED]      (~4 min: ~75 s per dataset)
## Runs the full pipeline on the THREE synthetic multi-period datasets
## (Donoho-Johnstone human densities: step, heavisine, bumps).  For each:
##   Stage 1  Siegel (1980) period detection (siegel_periods)  -> K clocks
##   Stage 2  RJMCMC over R (mcmc_improve.gibbs_rj_variable)   -> R, labels
##   Eval     period match [10,23,37], R match [3,2,1] and the AUC of the
##            single-clock baseline (mcmc_original.collapsed_gibbs, the
##            original model) against the multi-clock / variable-R extension

SIGMA2 = 0.06
TRUE_P = [10.0, 23.0, 37.0]
TRUE_R = [3, 2, 1]


## Canonical three-clock design: R_true = (3, 2, 1)
def clocks():
    return [
        {"p": 10.0, "n": 480, "mus": [1.2, 2.8, 4.4], "sigma2": SIGMA2},  ## R=3
        {"p": 23.0, "n": 320, "mus": [1.5, 4.3],      "sigma2": SIGMA2},  ## R=2
        {"p": 37.0, "n": 210, "mus": [3.0],           "sigma2": SIGMA2},  ## R=1
    ]


## Context manager silencing stdout during the sampler runs
@contextlib.contextmanager
def quiet():
    with open(os.devnull, "w") as dn:
        old = sys.stdout; sys.stdout = dn
        try: yield
        finally: sys.stdout = old


## Rank-based AUC of the scores against the binary labels
def auc(score, label):
    o = np.argsort(score); s = label[o]
    P = s.sum(); Nn = len(s) - P
    ranks = np.empty(len(s)); ranks[o] = np.arange(1, len(s) + 1)
    return (ranks[label == 1].sum() - P * (P + 1) / 2) / (P * Nn)


#### Full pipeline for one (seed, density) cell; returns a result dict
def run_cell(sd, kind, verbose=True):
    data = simulate_3clock(clocks(), {"kind": kind, "n": 240}, n_days=7, seed=sd)
    t, ztrue = data["t"], data["z_true"]

    ## Stage 1: Siegel period detection -> fundamental clocks
    t0 = time.time()
    sieg = siegel_detect_periods(t, delta=1.0, full_day=True, lam=0.6,
                                 alpha=0.01, min_cycles=60)
    det = sorted(np.round(sieg["fundamentals"], 1))
    pmatch = (det == TRUE_P)
    periods = det if len(det) == 3 else TRUE_P
    t1 = time.time() - t0

    ## Stage 2: RJMCMC over R (extended model) and the single-clock baseline
    t0 = time.time()
    with quiet():
        ## n_realloc=N//20 is the fast partial-sweep setting (valid for zeta0=1,
        ## slower mixing only); it reproduces the README reference numbers and
        ## the ~75 s runtime.  The library default is now a FULL sweep
        ## (n_realloc=None -> N), a correctness requirement under sparse priors
        ## (zeta0 < 1)
        out = gibbs_rj_variable(t, periods, Rmax=6, R_init=4, n_samp=800,
                                n_chains=1, burnin=700, zeta0=1.0, seed=11,
                                verbose=False, init_iter=3000,
                                n_realloc=len(t) // 20)
        base = collapsed_gibbs(t, sorted(TRUE_P)[0], n_samp=700, n_chains=1, burnin=500)
    Rmode = [int(x) for x in out["R_mode"]]
    rmatch = (Rmode == TRUE_R) if pmatch else None
    ab = float(auc(base[4].mean(axis=0), ztrue))
    ae = float(auc(out["cluster"].mean(axis=0), ztrue))
    t2 = time.time() - t0

    ## Print the per-cell results
    if verbose:
        print(f"[seed {sd}] {kind}: N={len(t)} events, "
              f"periodic(z=1) fraction={ztrue.mean():.2f}", flush=True)
        print(f"  Stage-1 Siegel: {len(sieg['peaks'])} contributing freqs "
              f"-> fundamentals={[int(x) for x in det]} "
              f"({'MATCH' if pmatch else 'MISMATCH'})  [{t1:.0f}s]", flush=True)
        print(f"  Stage-2 RJMCMC: R_mode={Rmode} "
              f"({'R MATCH [3,2,1]' if rmatch else ('R off' if pmatch else 'R n/a')})  "
              f"[{t2:.0f}s]", flush=True)
        print(f"  Eval: AUC original={ab:.3f}  full={ae:.3f}  "
              f"gain={ae-ab:+.3f}", flush=True)
    return {"kind": kind, "pmatch": pmatch, "R_mode": Rmode, "rmatch": rmatch,
            "auc_base": ab, "auc_ext": ae}



### Thesis figures (simulation)

## Command-line flags added for the dissertation:
##   --bench   run the step-background cell (canonical settings) and save the
##             scores of the original and full models + posterior peak snapshots
##   --sweep1  AUC vs share of a second clock          (original vs full)
##   --sweep2  AUC vs phase gap of a single clock      (original vs full)
##   --figs    render figures/fig_folded.pdf and figures/fig_class.pdf from the saved runs
import json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

## Colours and histogram style of the thesis figures
RED, BLUE = "#b5330f", "#1f5fa8"
HIST = dict(color="#c9d7e2", edgecolor="white")


## Wrapped-normal density (truncated wrap sum, |kappa| <= 3)
def _wn_pdf(x, mu, s2):
    p = np.zeros_like(x)
    for k in range(-3, 4):
        p += np.exp(-(x + 2*np.pi*k - mu)**2 / (2.0*s2)) / np.sqrt(2*np.pi*s2)
    return p


## Step-background cell with the canonical test_model settings; saves the
## per-event scores of the original and full models and the posterior peak
## snapshots needed by the folded-phase figure
def bench_step(save="bench_step.npz"):
    data = simulate_3clock(clocks(), {"kind": "step", "n": 240}, n_days=7, seed=0)
    t, ztrue, comp = data["t"], data["z_true"], data["comp"]
    sieg = siegel_detect_periods(t, delta=1.0, full_day=True, lam=0.6,
                                 alpha=0.01, min_cycles=60)
    det = sorted(np.round(sieg["fundamentals"], 1))
    periods = det if len(det) == 3 else TRUE_P
    with quiet():
        out = gibbs_rj_variable(t, periods, Rmax=6, R_init=4, n_samp=800,
                                n_chains=1, burnin=700, zeta0=1.0, seed=11,
                                verbose=False, init_iter=3000,
                                n_realloc=len(t)//20)
        base = collapsed_gibbs(t, sorted(TRUE_P)[0], n_samp=700, n_chains=1, burnin=500)
    sb, se = base[4].mean(0), out["cluster"].mean(0)
    Rm = [int(x) for x in out["R_mode"]]
    ## Posterior-mean peak locations and weights at the modal R of each clock
    mus, oms = [], []
    for k in range(3):
        sel = [(np.sort(m[k]), np.asarray(w[k])[np.argsort(m[k])])
               for R, m, w in out["snaps"] if R[k] == Rm[k] and len(m[k]) == Rm[k]]
        mus.append(np.mean([a for a, _ in sel], axis=0))
        oms.append(np.mean([b for _, b in sel], axis=0))
    s2 = out["sigma2"].reshape(-1, len(Rm)).mean(0)
    np.savez(save, t=t, z=ztrue, comp=comp, sb=sb, se=se, Rm=Rm, s2=s2,
             mus=np.array(mus, dtype=object), oms=np.array(oms, dtype=object),
             auc_b=auc(sb, ztrue), auc_e=auc(se, ztrue), periods=periods)
    print(f"bench: AUC original={auc(sb, ztrue):.3f} full={auc(se, ztrue):.3f} "
          f"R_mode={Rm} sigma2={np.round(s2, 3)}", flush=True)


## Folded phases of each clock's (true) events with the fitted shared-variance
## wrapped-normal phase mixture (posterior mean at the modal R)
def fig_folded(npz="bench_step.npz", out="figures/fig_folded"):
    os.makedirs("figures", exist_ok=True)
    d = np.load(npz, allow_pickle=True)
    t, comp, Rm, s2 = d["t"], d["comp"], d["Rm"], d["s2"]
    P = TRUE_P; TRUE_MU = [c["mus"] for c in clocks()]
    xg = np.linspace(0, 2*np.pi, 600)
    fig, axes = plt.subplots(1, 3, figsize=(12, 2.75))
    for k, ax in enumerate(axes):
        ## Transformation using the periodicity of clock k (true events only)
        x = 2*np.pi*(t[comp == k+1] % P[k]) / P[k]
        ax.hist(x, bins=40, range=(0, 2*np.pi), density=True,
                label="clock events (true)", **HIST)
        mix = sum(w*_wn_pdf(xg, m, float(s2[k]))
                  for m, w in zip(d["mus"][k], d["oms"][k]))
        ax.plot(xg, mix, color=RED, lw=1.8, label="fitted phase mixture")
        for m in TRUE_MU[k]:
            ax.axvline(m, color="0.35", ls="--", lw=0.9)
        ax.set_title(f"clock {k+1}: $p={int(P[k])}$ s, $\\hat R={int(Rm[k])}$",
                     fontsize=10)
        ax.set_xlim(0, 2*np.pi)
        ax.set_xlabel("$x$ (folded phase, radians)", fontsize=9)
        ax.tick_params(labelsize=8)
    axes[0].set_ylabel("density", fontsize=9)
    axes[0].legend(fontsize=7, frameon=False)
    fig.tight_layout(); fig.savefig(out + ".pdf"); fig.savefig(out + ".png", dpi=150)
    plt.close(fig); print(f"wrote {out}.pdf")


## AUC of the original vs the full model along the two controlled sweeps
## (generators from simulating_data), averaged over the simulation seeds; the
## sampler seed is fixed at 11 as in the benchmark
def run_sweeps(which, save, seeds=(0, 1, 2), grid=None):
    from simulating_data import sweep_second_period_fraction, sweep_phase_gap
    ## Reduced sweep-study settings.  The sweeps use FULL reallocation sweeps
    ## (n_realloc=N): with the fast partial setting the single-phase EM
    ## initialisation cannot be corrected at large phase gaps and occasional
    ## chains collapse
    kw = dict(n_samp=150, n_chains=1, burnin=100, zeta0=1.0, seed=11,
              verbose=False, init_iter=1200)
    if grid is None:
        grid = ([0.0, 0.1, 0.2, 0.3, 0.4, 0.5] if which == 1
                else [0.0, 0.63, 1.26, 1.88, 2.51, 3.14159])
    rows = []
    for v in grid:
        bs, es = [], []
        for sd in seeds:
            pts = (sweep_second_period_fraction([v], seed=sd, n_total=450, n_human=200)
                   if which == 1 else
                   sweep_phase_gap([v], seed=sd, n_auto=450, n_human=200))
            _, d = pts[0]
            t, z = d["t"], d["z_true"]
            with quiet():
                base = collapsed_gibbs(t, 10.0, n_samp=500, n_chains=1, burnin=350)
                if which == 1:
                    per = [10.0, 23.0] if v > 0 else [10.0]
                    o = gibbs_rj_variable(t, per, Rmax=2, R_init=1,
                                          n_realloc=len(t), **kw)
                else:
                    o = gibbs_rj_variable(t, [10.0], Rmax=3, R_init=2,
                                          n_realloc=len(t), **kw)
            bs.append(float(auc(base[4].mean(0), z)))
            es.append(float(auc(o["cluster"].mean(0), z)))
        rows.append([float(v), float(np.mean(bs)), float(np.mean(es)), bs, es])
        print(f"v={v:.2f}  original={np.mean(bs):.3f} {np.round(bs,3)}  "
              f"full={np.mean(es):.3f} {np.round(es,3)}", flush=True)
    json.dump(rows, open(save, "w")); print(f"wrote {save}")


## (a) ROC of the original vs the full model on the step benchmark (the 0.5
## threshold is marked); (b) AUC vs second-clock share; (c) AUC vs phase gap
def fig_class(npz="bench_step.npz", s1="sweep1.json", s2="sweep2.json",
              out="figures/fig_class"):
    os.makedirs("figures", exist_ok=True)
    d = np.load(npz, allow_pickle=True)
    z, sb, se = d["z"], d["sb"], d["se"]
    P, Nn = int((z == 1).sum()), int((z == 0).sum())

    ## Empirical ROC curve of a score vector
    def roc(s):
        thr = np.r_[np.inf, np.sort(np.unique(s))[::-1]]
        fpr = [float(((s >= c) & (z == 0)).sum())/Nn for c in thr] + [1.0]
        tpr = [float(((s >= c) & (z == 1)).sum())/P for c in thr] + [1.0]
        return np.asarray(fpr), np.asarray(tpr)

    fig, axes = plt.subplots(1, 3, figsize=(12, 2.75))
    ax = axes[0]
    for s, c, lab, a in ((sb, RED, "baseline", float(d["auc_b"])),
                         (se, BLUE, "extended model", float(d["auc_e"]))):
        f, tp = roc(s)
        ax.plot(f, tp, color=c, lw=1.7, label=f"{lab} (AUC {a:.3f})")
        ax.plot(float(((s >= .5) & (z == 0)).sum())/Nn,
                float(((s >= .5) & (z == 1)).sum())/P,
                "s", color=c, ms=5, mfc="white")
    ax.plot([0, 1], [0, 1], color="0.7", ls=":", lw=1)
    ax.set_xlabel("false positive rate", fontsize=9)
    ax.set_ylabel("true positive rate", fontsize=9)
    ax.set_title("ROC, step background", fontsize=10)
    ax.legend(fontsize=7, frameon=False, loc="lower right")
    for ax, fn, xl, ti in ((axes[1], s1, "share of second clock",
                            "second clock share sweep"),
                           (axes[2], s2, "phase gap (radians)",
                            "phase gap sweep")):
        rows = json.load(open(fn))
        v = np.array([r[0] for r in rows])
        for i, (c, lab) in enumerate(((RED, "baseline"), (BLUE, "extended model"))):
            mid = np.array([r[1 + i] for r in rows])
            lo = np.array([min(r[3 + i]) for r in rows])
            hi = np.array([max(r[3 + i]) for r in rows])
            ax.fill_between(v, lo, hi, color=c, alpha=0.13, lw=0)
            ax.plot(v, mid, "-o", color=c, ms=4, lw=1.6, label=lab)
        ax.set_xlabel(xl, fontsize=9); ax.set_ylabel("AUC", fontsize=9)
        ax.set_title(ti, fontsize=10); ax.legend(fontsize=7, frameon=False)
    for ax in axes:
        ax.tick_params(labelsize=8)
    axes[2].set_xticks([0, np.pi/2, np.pi])
    axes[2].set_xticklabels(["0", r"$\pi/2$", r"$\pi$"])
    fig.tight_layout(); fig.savefig(out + ".pdf"); fig.savefig(out + ".png", dpi=150)
    plt.close(fig); print(f"wrote {out}.pdf")

## Command-line interface
if __name__ == "__main__":
    if "--bench" in sys.argv:  bench_step();               sys.exit(0)
    if "--sweep1" in sys.argv: run_sweeps(1, "sweep1.json"); sys.exit(0)
    if "--sweep2" in sys.argv: run_sweeps(2, "sweep2.json"); sys.exit(0)
    if "--figs" in sys.argv:   fig_folded(); fig_class();   sys.exit(0)
    sd = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rows = [run_cell(sd, k, verbose=True) for k in ("step", "heavisine", "bumps")]
    print("\n================ SUMMARY (seed %d) ================" % sd)
    print(f"{'density':10s} | {'period':>12s} | {'R':>12s} | "
          f"{'AUC base':>8s} {'ext':>6s} {'gain':>7s}")
    for r in rows:
        per = "[10,23,37]" if r["pmatch"] else "MISS"
        rstr = str(r["R_mode"]) + (" ok" if r["rmatch"] else "")
        print(f"{r['kind']:10s} | {per:>12s} | {rstr:>12s} | "
              f"{r['auc_base']:8.3f} {r['auc_ext']:6.3f} "
              f"{r['auc_ext']-r['auc_base']:+7.3f}")
