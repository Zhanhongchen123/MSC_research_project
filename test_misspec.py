#!/usr/bin/env python3
## Misspecification checks (Section 3.1): one-clock control and the 4:3
## harmonic pair; bench settings mirror test_model.run_cell.
##   exp a: one-clock truth  -> detection, R_mode, AUC baseline vs extended
##   exp b: harmonic pair 15/20 s (4:3) -> detection-level collapse behaviour
## Bench settings mirror test_model.run_cell exactly (AUC-table convention).
import sys, json, os, numpy as np
from simulating_data import simulate_3clock
from siegel_periods import siegel_detect_periods
from mcmc_improve import gibbs_rj_variable
from mcmc_original import collapsed_gibbs

def auc(score, label):
    o = np.argsort(score); s = label[o]
    P = s.sum(); Nn = len(s) - P
    ranks = np.empty(len(s)); ranks[o] = np.arange(1, len(s) + 1)
    return (ranks[label == 1].sum() - P * (P + 1) / 2) / (P * Nn)

exp, sd = sys.argv[1], int(sys.argv[2])
if exp == "a":
    cl = [{"p": 10.0, "n": 480, "mus": [1.2], "sigma2": 0.06}]
    d = simulate_3clock(cl, {"kind": "step", "n": 240}, n_days=7, seed=sd)
    t, ztrue = d["t"], (d["z_true"] >= 1).astype(int)
    sieg = siegel_detect_periods(t, delta=1.0, full_day=True, lam=0.6,
                                 alpha=0.01, min_cycles=60)
    P = [round(p, 4) for p in sieg["fundamentals"]]
    out = gibbs_rj_variable(t, sieg["fundamentals"], Rmax=6, R_init=4, n_samp=800,
                            n_chains=1, burnin=700, zeta0=1.0, seed=11,
                            n_realloc=len(t)//20, verbose=False)
    base = collapsed_gibbs(t, sieg["fundamentals"][0], n_samp=700, n_chains=1, burnin=500)
    res = {"seed": sd, "fundamentals": P,
           "R_mode": [int(x) for x in out["R_mode"]],
           "auc_base": float(auc(base[4].mean(axis=0), ztrue)),
           "auc_ext": float(auc(out["cluster"].mean(axis=0), ztrue))}
else:
    cl = [{"p": 15.0, "n": 400, "mus": [2.0], "sigma2": 0.06},
          {"p": 20.0, "n": 400, "mus": [4.0], "sigma2": 0.06}]
    d = simulate_3clock(cl, {"kind": "step", "n": 240}, n_days=7, seed=sd)
    sieg = siegel_detect_periods(d["t"], delta=1.0, full_day=True, lam=0.6,
                                 alpha=0.01, min_cycles=60)
    res = {"seed": sd,
           "fundamentals": [round(p, 4) for p in sieg["fundamentals"]],
           "top_peaks": [[round(p["period"], 4), round(p["Y"]/sieg["threshold"], 1)]
                         for p in sieg["peaks"][:4]]}
outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "misspec_out")
os.makedirs(outdir, exist_ok=True)
json.dump(res, open(os.path.join(outdir, f"{exp}{sd}.json"), "w"))
print(res)
