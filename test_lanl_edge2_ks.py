#!/usr/bin/env python3
import sys, os, contextlib, warnings; warnings.filterwarnings("ignore")
import numpy as np
from mcmc_original import collapsed_gibbs
from mcmc_improve import gibbs_rj_variable
from ks_gof import wold_ks

##############################################################
## Natural K=2 LANL edge Comp393078>Comp186884 (two clocks) ##
##############################################################

## LANL netflow edge, days 02-04, N=21450: two clocks 15.0006 s + 20.0009 s,
## ~49% automated.  The original model at 15 s against the extended model at
## [15, 20]; expected KS 0.3413 -> 0.2105 / 0.1543, R_mode=[2,1].
## Usage:  python3 test_lanl_edge2_ks.py [Dataset/lanl_edge2.txt] [--smoke]
##                                       [--stage=ku|base|full|fin] [--fig]
## Staged execution: each stage saves its outputs into e2_cache.npz, so the
## run can be resumed across tool-call timeouts (ku = unfiltered KS,
## base = original model, full = extended model, fin = extended KS + summary
## line and, with --fig, the three-panel figure from the cache).

## Command-line arguments and run sizes
SMOKE = "--smoke" in sys.argv
A = [a for a in sys.argv[1:] if not a.startswith("--")]
DATA = A[0] if A else "Dataset/lanl_edge2.txt"
NS1, BU1, NS2, BU2, IT = (120, 90, 80, 60, 3000) if SMOKE else (300, 220, 200, 150, 10000)
P1, P2 = 15.0006, 20.000885
## Context manager silencing stdout during the sampler runs
@contextlib.contextmanager
def quiet():
    with open(os.devnull, "w") as d:
        o = sys.stdout; sys.stdout = d
        try: yield
        finally: sys.stdout = o
## Import the times
t = np.sort(np.loadtxt(DATA)); N = len(t); td = 0.6 * (t[-1] - t[0]) / 86400
## Staged execution: stage flag and cache helpers
STAGE = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--stage=")), None)
CACHE = "e2_cache.npz"
def _save(**kw):
    old = dict(np.load(CACHE, allow_pickle=True)) if os.path.exists(CACHE) else {}
    old.update(kw); np.savez(CACHE, **old)
def _load():
    return dict(np.load(CACHE, allow_pickle=True))

## Stage ku: Wold KS of the unfiltered stream
if STAGE in (None, "ku"):
    ku = wold_ks(t, train_days=td)
    _save(ku_p=ku["p"], ku_ks=ku["ks"])
    if STAGE == "ku":
        print(f"[stage ku] KS={ku['ks']:.4f}"); sys.exit(0)
else:
    _c = _load(); ku = {"p": _c["ku_p"], "ks": float(_c["ku_ks"])}
## Stage base: the original model at P1
if STAGE in (None, "base"):
    np.random.seed(0)
    with quiet(): b = collapsed_gibbs(t, P1, n_samp=NS1, n_chains=1, burnin=BU1)
    pb = b[4].mean(0); kb = wold_ks(t[pb < .5], train_days=td)
    _save(pb=pb, kb_p=kb["p"], kb_ks=kb["ks"])
    if STAGE == "base":
        print(f"[stage base] KS={kb['ks']:.4f} periodic={int((pb>=.5).sum())}"); sys.exit(0)
else:
    _c = _load(); pb = _c["pb"]; kb = {"p": _c["kb_p"], "ks": float(_c["kb_ks"])}
## Stage full: the extended model at [P1, P2]
if STAGE in (None, "full"):
    with quiet():
        o = gibbs_rj_variable(t, [P1, P2], Rmax=4, R_init=2, n_samp=NS2, n_chains=1,
                              burnin=BU2, zeta0=1.0, seed=0, verbose=False,
                              init_iter=IT, n_realloc=N // 20)
    po = o["cluster"].mean(0)
    Rm = [int(x) for x in np.atleast_1d(o["R_mode"])]
    ## Posterior-mean peak locations and weights at the modal R of each clock
    mus, oms = [], []
    for k in range(2):
        sel = [(np.sort(m[k]), np.asarray(w[k])[np.argsort(m[k])])
               for R, m, w in o["snaps"] if R[k] == Rm[k] and len(m[k]) == Rm[k]]
        mus.append(np.mean([a for a, _ in sel], axis=0))
        oms.append(np.mean([b for _, b in sel], axis=0))
    _save(po=po, R_mode=np.atleast_1d(o["R_mode"]).astype(int),
          pk=o["cluster_k"].mean(0),
          mus=np.array(mus, dtype=object), oms=np.array(oms, dtype=object),
          s2=o["sigma2"].reshape(-1, 2).mean(0))
    if STAGE == "full":
        print(f"[stage full] periodic={int((po>=.5).sum())} R_mode={[int(x) for x in np.atleast_1d(o['R_mode'])]}"); sys.exit(0)
else:
    _c = _load(); po = _c["po"]
    class _O: pass
    o = {"R_mode": _c["R_mode"]}
## Final stage: extended KS and the summary line
ko = wold_ks(t[po < .5], train_days=td)
print(f"unfiltered KS={ku['ks']:.4f} | original@{P1}s KS={kb['ks']:.4f} "
      f"(periodic {int((pb>=.5).sum())}) | ours@[{P1},{P2}] KS={ko['ks']:.4f} "
      f"(periodic {int((po>=.5).sum())}, R_mode={[int(x) for x in np.atleast_1d(o['R_mode'])]})")


## Three-panel figure (--fig): folded clocks and the Q-Q of the Wold p-values
if "--fig" in sys.argv:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs("figures", exist_ok=True)
    RED, BLUE = "#b5330f", "#1f5fa8"
    ## Wrapped-normal density (truncated wrap sum, |kappa| <= 3)
    def _wn_pdf(x, mu, v):
        p = np.zeros_like(x)
        for kk in range(-3, 4):
            p += np.exp(-(x + 2*np.pi*kk - mu)**2 / (2.0*v)) / np.sqrt(2*np.pi*v)
        return p
    _cc = _load()
    pk, mus, oms, s2v = _cc["pk"], _cc["mus"], _cc["oms"], _cc["s2"]
    Rmv = [int(x) for x in np.atleast_1d(_cc["R_mode"])]
    ## Posterior-conditioned folding: plot only the events the model allocates
    ## to clock k.  Folding ALL events at each period aliases the other clock's
    ## comb into broad humps (the two clocks have a 4:3 frequency ratio)
    assign = pk.argmax(0)
    fig, axes = plt.subplots(1, 3, figsize=(12, 2.75))
    for k, (ax, P) in enumerate(((axes[0], P1), (axes[1], P2))):
        sel = (po >= .5) & (assign == k)
        ax.hist(t[sel] % P, bins=60, density=True,
                **{"color": "#c9d7e2", "edgecolor": "white"},
                label="allocated events")
        xs = np.linspace(0, P, 500)
        dens = sum(w*_wn_pdf(2*np.pi*xs/P, m, float(s2v[k]))
                   for m, w in zip(mus[k], oms[k])) * (2*np.pi/P)
        ax.plot(xs, dens, color=RED, lw=1.8, label="fitted phase mixture")
        ax.set_xlim(0, P)
        ax.set_xlabel(f"t mod {P:.4f} (s)", fontsize=9)
        ax.set_title(f"clock {k+1}: $p={P:.4f}$ s, $\\hat R$ = {Rmv[k]}", fontsize=10)
        ax.tick_params(labelsize=8)
    axes[0].set_ylabel("density", fontsize=9)
    axes[0].legend(fontsize=7, frameon=False)
    ## Q-Q of the Wold predictive p-values (unfiltered / baseline / extended)
    ax = axes[2]
    for r, c, ls, lab in ((ku, "0.4", "--", "unfiltered"),
                          (kb, RED, "-", "baseline"),
                          (ko, BLUE, "-", "extended model")):
        p = np.sort(r["p"])
        ax.plot(np.linspace(0, 1, len(p)), p, color=c, ls=ls, lw=1.6,
                label=f"{lab} KS={r['ks']:.3f}")
    ax.plot([0, 1], [0, 1], color="0.7", ls=":", lw=1)
    ax.set_xlabel("Uniform quantiles", fontsize=9)
    ax.set_ylabel("Wold $p$-value quantiles", fontsize=9)
    ax.set_title("Wold KS before/after filtering", fontsize=10)
    ax.legend(fontsize=7, frameon=False); ax.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig("figures/fig_lanl_edge2.pdf"); fig.savefig("figures/fig_lanl_edge2.png", dpi=150)
    print("wrote figures/fig_lanl_edge2.pdf")
