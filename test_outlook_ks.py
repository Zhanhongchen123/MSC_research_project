#!/usr/bin/env python3
import sys, os, contextlib
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from fft_detect import detect_periodicity
from mcmc_original import collapsed_gibbs
from mcmc_improve import gibbs_rj_variable
from ks_gof import wold_ks

########################################################
## Single-period negative control on the Outlook edge ##
########################################################

## Outlook edge of Sanna Passino & Heard (2020): N=7583, one 8 s polling clock.
## Pipeline: fft_detect -> the original model (mcmc_original.collapsed_gibbs)
## and the extended model (mcmc_improve.gibbs_rj_variable, R sampled) at the
## detected period; three-way nonparametric Wold KS (ks_gof) and a Q-Q figure.
## Expected (full run): period 8.0009 s; the extension degenerates to K=1,
## R_mode=1; KS: unfiltered 0.3529 -> original 0.2125, extended 0.2155 (the paper
## reports 0.3675 -> 0.1438 after multi-pass filtering; the single-pass DROP is
## the faithful comparison).
## Usage:  python3 test_outlook_ks.py [Dataset/outlook.txt] [--smoke]
## Needs the patched mcmc_improve.py (n_realloc parameter).

## Command-line arguments and run sizes
SMOKE = "--smoke" in sys.argv
ARGS = [a for a in sys.argv[1:] if not a.startswith("--")]
DATA = ARGS[0] if ARGS else "Dataset/outlook.txt"
NS, BU, IT = (120, 100, 3000) if SMOKE else (400, 300, 20000)


## Context manager silencing stdout during the sampler runs
@contextlib.contextmanager
def quiet():
    with open(os.devnull, "w") as dn:
        o = sys.stdout; sys.stdout = dn
        try: yield
        finally: sys.stdout = o


#### Main routine
def main():
    ## Import the times and detect the periodicity
    t = np.sort(np.loadtxt(DATA)); N = len(t)
    td = 0.6 * (t[-1] - t[0]) / 86400.0
    p = detect_periodicity(t)["p"]
    print(f"N={N}  span={(t[-1]-t[0])/86400:.2f} d   detected period p={p:.4f} s")
    ## Wold KS of the unfiltered stream
    ku = wold_ks(t, train_days=td)

    ## Original model at the detected period
    np.random.seed(0)
    with quiet():
        b = collapsed_gibbs(t, p, n_samp=NS, n_chains=1, burnin=BU)
    pb = b[4].mean(0); kb = wold_ks(t[pb < 0.5], train_days=td)

    ## Extended model with R sampled
    with quiet():
        o = gibbs_rj_variable(t, [p], Rmax=6, R_init=3, n_samp=NS, n_chains=1,
                              burnin=BU, zeta0=1.0, seed=0, verbose=False,
                              init_iter=IT, n_realloc=N // 20)
    po = o["cluster"].mean(0); ko = wold_ks(t[po < 0.5], train_days=td)

    ## Print output
    print(f"\nunfiltered            KS = {ku['ks']:.4f}  (n_test={ku['n_test']})")
    print(f"original @{p:.4f}s  KS = {kb['ks']:.4f}  (periodic {int((pb>=.5).sum())})")
    print(f"ours (R sampled)      KS = {ko['ks']:.4f}  (periodic {int((po>=.5).sum())}, "
          f"R_mode={list(np.atleast_1d(o['R_mode']))})")
    print("-> extension degenerates to K=1, R=1 on a genuinely single-period edge.")

    ## Q-Q figure of the Wold predictive p-values
    os.makedirs("figures", exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    for r, lab, c, ls in [(ku, f"unfiltered KS={ku['ks']:.3f}", "crimson", "--"),
                          (kb, f"original KS={kb['ks']:.3f}", "#b5651d", "-"),
                          (ko, f"ours KS={ko['ks']:.3f}", "#2a6f97", "-")]:
        q = np.sort(r["p"]); u = (np.arange(1, len(q) + 1) - .5) / len(q)
        ax.plot(u, q, ls, color=c, lw=1.6, label=lab)
    ax.plot([0, 1], [0, 1], "k:", lw=1)
    ax.set_xlabel("Uniform quantiles"); ax.set_ylabel("Wold predictive p-value quantiles")
    ax.set_title("Outlook: Wold KS before/after filtering")
    ax.legend(fontsize=8, loc="lower right"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fig.tight_layout(); fig.savefig("figures/outlook_ks.png", dpi=130)
    print("saved figures/outlook_ks.png")


if __name__ == "__main__":
    main()
