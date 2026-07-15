#!/usr/bin/env python3
import os, sys, contextlib, warnings; warnings.filterwarnings("ignore")
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from simulating_data import simulate_3clock
from mcmc_improve import gibbs_rj_variable

#########################################
## Density-recovery figure (Chapter 3) ##
#########################################

## Regenerates the Chapter-3 density-recovery figure: the posterior-mean human
## daily density (traced by gibbs_rj_variable with trace_density=True) against
## a histogram of the TRUE human event times, for the step / heavisine / bumps
## backgrounds (seed 0).  Output: figures/fig_density.pdf

SIGMA2 = 0.06; TRUE_P = [10.0, 23.0, 37.0]
## Canonical three-clock design
def clocks():
    return [{"p":10.0,"n":480,"mus":[1.2,2.8,4.4],"sigma2":SIGMA2},
            {"p":23.0,"n":320,"mus":[1.5,4.3],"sigma2":SIGMA2},
            {"p":37.0,"n":210,"mus":[3.0],"sigma2":SIGMA2}]
## Context manager silencing stdout during the sampler runs
@contextlib.contextmanager
def quiet():
    with open(os.devnull,"w") as d:
        o=sys.stdout; sys.stdout=d
        try: yield
        finally: sys.stdout=o
## One panel per human background: fit with the density trace on, then overlay
fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), sharey=False)
for ax, kind in zip(axes, ("step", "heavisine", "bumps")):
    data = simulate_3clock(clocks(), {"kind": kind, "n": 240}, n_days=7, seed=0)
    t, z = data["t"], data["z_true"]
    with quiet():
        o = gibbs_rj_variable(t, TRUE_P, Rmax=6, R_init=4, n_samp=800, n_chains=1,
                              burnin=700, zeta0=1.0, seed=11, verbose=False,
                              init_iter=3000, n_realloc=len(t)//20,
                              trace_density=True)
    ## Transformation to daily arrival time of the true human events
    yh = (t[z == 0] % 86400.0) / 86400.0 * 2 * np.pi
    ax.hist(yh, bins=30, range=(0, 2*np.pi), density=True, color="#c9d7e2",
            edgecolor="white", label="true human events")
    ax.plot(o["density_y"], o["density"], color="#b5330f", lw=1.8,
            label="posterior-mean density")
    ax.set_title(kind); ax.set_xlabel(r"$y$ (time of day, radians)")
    ax.set_xlim(0, 2*np.pi)
axes[0].set_ylabel("density"); axes[0].legend(fontsize=8)
os.makedirs("figures", exist_ok=True)
fig.tight_layout(); fig.savefig("figures/fig_density.pdf")
print("saved figures/fig_density.pdf")
