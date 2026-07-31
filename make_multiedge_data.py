#!/usr/bin/env python3
import os, sys
import numpy as np
from simulating_data import simulate_3clock

######################################################################
## Synthetic multi-edge data (Section 3.2): a shared service clock  ##
######################################################################

## Generates the M-edge families of the combining-edges experiment: every
## edge carries a weak shared service clock (period P_S plus an edge-level
## skew, phases NOT locked across edges) and its own human stream; one edge
## carries a strong private clock as a decoy.  For each seed a matched
## no-shared-clock control ("null") is generated with the same settings.
## Deterministic given the seeds; streams are written one timestamp per
## line, as the real edges are.
## Usage:  python3 make_multiedge_data.py            (seeds 0 1 2, fam+null)
## Output: Dataset/multiedge/{fam|null}{seed}_edge{0..7}.txt  +  meta.txt

M        = 8                    ## edges in the family (as the observed ones)
P_S      = 15.0                 ## shared service period (s)
SKEW_SD  = 0.0015               ## edge-level skew (s), calibrated on the
                                ##   short-period LANL families (<~2 ms)
N_SHARED = 165                  ## shared-clock events per edge (weak)
N_HUMAN  = 2500                 ## human events per edge (step density)
DECOY    = (2, 20.0, 2500)      ## (edge index, period, events): private clock
N_DAYS   = 3
SIGMA2   = 0.06
SEEDS    = (0, 1, 2)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "Dataset", "lanl_mega")


## Simulate the M edges; phases are NOT locked across edges (fresh mus)
def simulate(seed, null):
    rng = np.random.default_rng(1000 + seed)
    edges, periods = [], []
    for e in range(M):
        clocks = []
        p_e = P_S + float(rng.normal(0.0, SKEW_SD))
        mu_s = float(rng.uniform(0, 2 * np.pi))
        if not null:
            clocks.append({"p": p_e, "n": N_SHARED, "mus": [mu_s],
                           "sigma2": SIGMA2})
        if e == DECOY[0]:
            clocks.append({"p": DECOY[1], "n": DECOY[2],
                           "mus": [float(rng.uniform(0, 2 * np.pi))],
                           "sigma2": SIGMA2})
        d = simulate_3clock(clocks, {"kind": "step", "n": N_HUMAN},
                            n_days=N_DAYS, seed=seed * 10 + e)
        edges.append(np.array(sorted(set(d["t"])), float))
        periods.append(p_e if not null else float("nan"))
    return edges, periods


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    meta = [f"M\t{M}\nP_S\t{P_S}\nSKEW_SD\t{SKEW_SD}\nN_SHARED\t{N_SHARED}\n"
            f"N_HUMAN\t{N_HUMAN}\nDECOY\t{DECOY}\nN_DAYS\t{N_DAYS}\n"
            f"SIGMA2\t{SIGMA2}\n"]
    for null in (False, True):
        for sd in SEEDS:
            tag = f"{'null' if null else 'fam'}{sd}"
            edges, periods = simulate(sd, null)
            for e, t in enumerate(edges):
                path = os.path.join(OUT, f"{tag}_edge{e}.txt")
                with open(path, "w") as fh:
                    fh.write("\n".join(repr(float(x)) for x in t) + "\n")
            ns = "/".join(str(len(t)) for t in edges)
            ps = "/".join("-" if np.isnan(p) else f"{p:.6f}" for p in periods)
            meta.append(f"{tag}\tN={ns}\tshared_periods={ps}\n")
            print(f"{tag}: {M} edges written  (N = {ns})")
    with open(os.path.join(OUT, "meta.txt"), "w") as fh:
        fh.writelines(meta)
    print(f"design record -> {os.path.join(OUT, 'meta.txt')}")
