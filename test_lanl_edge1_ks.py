#!/usr/bin/env python3
import sys, os, contextlib
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from fft_detect import detect_periodicity
from mcmc_original import collapsed_gibbs
from mcmc_improve import gibbs_rj_variable
from ks_gof import wold_ks

#####################################################################
## Natural LANL edge Comp683203>Comp186884:443 (300 s, R=3 phases) ##
#####################################################################

## LANL netflow edge, days 02-04, N=6969: a 300 s scheduled template with
## R=3 phases.  Pipeline:
## 1. fft_detect gives p1 (27.2728 s -- in truth the 11th harmonic of the
##    300 s frame); refine it by maximising the phase concentration
## 2. FRAME SEARCH: for m=1..12 refine m*p1 and count the circular hot
##    clusters in the folded histogram; the frame maximising the compression
##    is the true template (m=11 -> 300 s, 3 clusters at ~107.5/164.9/241.1 s)
## 3. the original model at p1; the extended model at the frame with informed
##    initialisation (mu_init = the cluster centres; requires the patched
##    mcmc_improve.py)
## 4. three-way Wold KS and a two-panel figure (template + Q-Q)
## Expected (full run): R_mode=3 (P(R=3)=1.0), sigma2~0.0045;
## KS: unfiltered 0.3755 -> original 0.2390, extended 0.2500 (parity explained
## by the mod-27.27 coincidence of the three modes; the extension adds the
## CORRECT structure).
## Usage:  python3 test_lanl_edge1_ks.py [Dataset/lanl_edge1.txt] [--smoke]

## Command-line arguments and run sizes
SMOKE = "--smoke" in sys.argv
ARGS = [a for a in sys.argv[1:] if not a.startswith("--")]
DATA = ARGS[0] if ARGS else "Dataset/lanl_edge1.txt"
NS1, BU1 = (120, 100) if SMOKE else (400, 300)          ## original model
NS2, BU2 = (100, 90) if SMOKE else (350, 280)           ## extended model
IT = 3000 if SMOKE else 20000


## Context manager silencing stdout during the sampler runs
@contextlib.contextmanager
def quiet():
    with open(os.devnull, "w") as dn:
        o = sys.stdout; sys.stdout = dn
        try: yield
        finally: sys.stdout = o


## Refine a period by maximising the circular phase concentration R-bar
## (zoomed grid search around p0)
def refine_period(t, p0, rel=0.005, ngrid=1501, zoom=3):
    lo, hi = p0 * (1 - rel), p0 * (1 + rel); bp, bR = p0, -1.0
    for _ in range(zoom):
        ps = np.linspace(lo, hi, ngrid); Rs = np.empty(ngrid)
        for s in range(0, ngrid, 256):
            a = 2 * np.pi * (t[:, None] % ps[None, s:s + 256]) / ps[None, s:s + 256]
            Rs[s:s + 256] = np.abs(np.mean(np.exp(1j * a), axis=0))
        i = int(Rs.argmax()); bp, bR = float(ps[i]), float(Rs[i])
        d = (hi - lo) / (ngrid - 1); lo, hi = bp - 6 * d, bp + 6 * d
    return bp, bR


## Contiguous runs of hot bins in a circular histogram (wraparound run merged)
def hot_clusters(h, thr):
    n = len(h); hot = h >= thr
    cl, cur = [], []
    for i in range(n):
        if hot[i]: cur.append(i)
        elif cur: cl.append(cur); cur = []
    if cur: cl.append(cur)
    if len(cl) > 1 and hot[0] and hot[-1]:
        cl[0] = cl[-1] + cl[0]; cl.pop()
    return cl


## Frame search: scan the frames m*p1 and return (p_frame, mode centres in
## radians, cluster count, m).  At the true frame m* the template COLLAPSES,
## so the cluster count C(m*) falls below m (aliased frames give C(m)=m); the
## score is the compression m/C, with ties broken towards smaller m.  C must
## be >= 2 for a multi-phase frame, otherwise fall back to the single-phase p1
def find_frame(t, p1, mmax=12, nbins=60):
    best = None
    for m in range(1, mmax + 1):
        pf, _ = refine_period(t, m * p1)
        x = (t % pf) / pf * nbins
        h, _ = np.histogram(x, bins=nbins, range=(0, nbins))
        thr = 2.0 * h.mean()                    ## mean-based threshold: keeps the minor modes
        cl = hot_clusters(h.astype(float), thr)
        C = len(cl)
        if C < 2:                               ## flat / drifted / single-arc frame: not a template
            continue
        score = m / C
        if best is None or score > best[0] + 1e-9:
            ## Circular-mean centre of each hot cluster
            cent = []
            for c in cl:
                ang = (np.array(c) + 0.5) / nbins * 2 * np.pi
                w = h[c]
                cent.append(float(np.angle(np.sum(w * np.exp(1j * ang))) % (2 * np.pi)))
            best = (score, C, m, pf, sorted(cent))
    score, C, m, pf, cent = best
    if C < 2:                                   ## no multi-phase frame found
        return p1, None, 1, 1
    return pf, np.array(cent), C, m


#### Main routine
def main():
    ## Import the times, detect and refine the period, search the frame
    t = np.sort(np.loadtxt(DATA)); N = len(t)
    td = 0.6 * (t[-1] - t[0]) / 86400.0
    p1 = detect_periodicity(t)["p"]; p1, r1 = refine_period(t, p1, rel=0.02)
    pf, mu, C, m = find_frame(t, p1)
    print(f"N={N}   FFT period p1={p1:.4f}s (R-bar={r1:.2f})")
    if mu is not None:
        print(f"frame search: m={m} -> template p={pf:.4f}s with {C} phase modes at "
              + ", ".join(f"{v*pf/2/np.pi:.1f}s" for v in mu))
    else:
        print("frame search: no multi-phase frame -> single-phase at p1")
    ## Wold KS of the unfiltered stream
    ku = wold_ks(t, train_days=td)

    ## Original model at p1
    np.random.seed(0)
    with quiet():
        b = collapsed_gibbs(t, p1, n_samp=NS1, n_chains=1, burnin=BU1)
    pb = b[4].mean(0); kb = wold_ks(t[pb < 0.5], train_days=td)

    ## Extended model at the frame with informed initialisation
    kw = dict(Rmax=6, n_samp=NS2, n_chains=1, burnin=BU2, zeta0=1.0, seed=0,
              verbose=False, init_iter=IT, n_realloc=N // 10)
    if mu is not None:
        kw["mu_init"] = [mu]
    with quiet():
        o = gibbs_rj_variable(t, [pf], **kw)
    po = o["cluster"].mean(0); ko = wold_ks(t[po < 0.5], train_days=td)

    ## Print output
    print(f"\nunfiltered                KS = {ku['ks']:.4f}")
    print(f"original @{p1:.2f}s     KS = {kb['ks']:.4f}  (periodic {int((pb>=.5).sum())})")
    print(f"ours @{pf:.1f}s template   KS = {ko['ks']:.4f}  (periodic {int((po>=.5).sum())}, "
          f"R_mode={[int(x) for x in np.atleast_1d(o['R_mode'])]}, sigma2={float(np.mean(o['sigma2'])):.4f})")

    ## Two-panel figure: the folded template and the Q-Q of the Wold p-values
    os.makedirs("figures", exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    ax = axes[0]; off = t % pf
    ax.hist(off, bins=150, color="#c9d7e2")
    for v in (mu if mu is not None else []):
        ax.axvline(v * pf / 2 / np.pi, color="#b5330f", lw=1.6, ls="--")
    ax.set_xlabel(f"t mod {pf:.1f} s"); ax.set_ylabel("events")
    ax.set_title(f"folded at the {pf:.1f} s frame ($\\hat R$ = {int(np.atleast_1d(o['R_mode'])[0])})")
    ax = axes[1]
    for r, lab, c, ls in [(ku, f"unfiltered KS={ku['ks']:.3f}", "0.4", "--"),
                          (kb, f"baseline KS={kb['ks']:.3f}", "#b5330f", "-"),
                          (ko, f"extended model KS={ko['ks']:.3f}", "#1f5fa8", "-")]:
        q = np.sort(r["p"]); u = (np.arange(1, len(q) + 1) - .5) / len(q)
        ax.plot(u, q, ls, color=c, lw=1.6, label=lab)
    ax.plot([0, 1], [0, 1], "k:", lw=1)
    ax.set_xlabel("Uniform quantiles"); ax.set_ylabel("Wold p-value quantiles")
    ax.set_title("Wold KS before/after filtering")
    ax.legend(fontsize=8, loc="lower right"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fig.tight_layout(); fig.savefig("figures/lanl_edge1_ks.png", dpi=150); fig.savefig("figures/fig_lanl_edge1.pdf")
    print("saved figures/lanl_edge1_ks.png")


if __name__ == "__main__":
    main()
