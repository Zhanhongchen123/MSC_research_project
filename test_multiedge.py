#!/usr/bin/env python3
import sys, os, json, math, glob
import numpy as np
from siegel_periods import siegel_detect_periods

######################################################################
## Combining edges: a shared service clock (Section 3.2)            ##
######################################################################

## Tests one generated family (make_multiedge_data.py) on two routes.
## Single-edge route: the report's own detector per edge.  Combined route:
## locally-whitened matched-window evidence, candidates nominated by the
## union of per-edge leading ordinates, TRIMMED Fisher combination (the
## strongest edge is left out at every frequency, so a detection needs
## support from several edges and no single edge's private structure can
## masquerade as a family clock), Bonferroni over the candidates.
## Expected (three seeds): the shared 15 s clock is never a single-edge
## fundamental and contributes on at most two edges, yet the combination
## detects it on every seed (z ~ 6-8 against a bar of ~3.7); the private
## 20 s decoy never enters the family list; the null runs detect nothing.
## Usage:  python3 test_multiedge.py fam0|fam1|fam2|null0|null1|null2
##                                   [datadir=Dataset/multiedge]
## Results are printed and written to multiedge_out/ next to this script.

P_S      = 15.0                 ## the shared period under test (design value)
PMIN, PMAX = 8.0, 70.0          ## scan band: the sub-minute service range
WIN      = 3                    ## matched window half-width (Fourier bins)
ALPHA    = 0.01                 ## family-wise level of the combined test
N_DAYS   = 3
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "multiedge_out")


## Inverse standard-normal quantile by bisection on erfc (numpy-free tail)
def z_quantile(q):
    lo, hi = 0.0, 40.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if 0.5 * math.erfc(mid / math.sqrt(2.0)) > q:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


## Binned periodogram on the common grid, restricted to the period band
def spectrum(t, nb):
    c = np.zeros(nb); np.add.at(c, np.clip(t.astype(int), 0, nb - 1), 1.0)
    c -= c.mean()
    P = np.abs(np.fft.rfft(c)) ** 2
    fk = np.fft.rfftfreq(nb, d=1.0)
    band = (fk >= 1.0 / PMAX) & (fk <= 1.0 / PMIN)
    return fk[band], P[band], int(band.sum())


## Per-bin matched-window p-values of one edge.  The null mean is estimated
## by a LOCAL median (blocks of 256 bins, interpolated), which whitens broad
## coloured ridges -- the shared daily template -- while leaving sharp clock
## lines untouched; p = exp(-ln2 * P/median) is the locally exponential tail,
## Bonferroni-corrected over the matched window
def local_ratio(P):
    nblk = max(1, len(P) // 256)
    ed = np.linspace(0, len(P), nblk + 1).astype(int)
    med = np.interp(np.arange(len(P)),
                    0.5 * (ed[:-1] + ed[1:]),
                    np.array([np.median(P[a:b]) for a, b in zip(ed[:-1], ed[1:])]))
    return P / np.maximum(med, 1e-12)


def window_p(P, n):
    R = local_ratio(P)
    w = 2 * WIN + 1
    Rp = np.pad(R, WIN)                    ## zero-pad: no wrap across the band
    Rmax = np.maximum.reduce([Rp[s:s + len(R)] for s in range(w)])
    return np.clip(w * np.exp(-math.log(2.0) * Rmax), 1e-300, 1.0)


def run(tag, datadir):
    files = sorted(glob.glob(os.path.join(datadir, f"{tag}_edge*.txt")))
    if not files:
        sys.exit(f"no {tag}_edge*.txt under {datadir} -- run make_multiedge_data.py")
    edges = [np.sort(np.unique(np.loadtxt(f))) for f in files]
    M = len(edges)
    nb = N_DAYS * 86400
    ## --- single-edge route: the report's own detector, per edge ---
    single, contrib, og_claims = [], 0, []
    for t in edges:
        s = siegel_detect_periods(t, delta=1.0, full_day=True, lam=0.6,
                                  alpha=0.01, min_cycles=60)
        single.append(sorted(round(p, 2) for p in s["fundamentals"]))
        if any(abs(p["period"] - P_S) < 0.05 for p in s["peaks"]):
            contrib += 1                   ## sub-detection sightings (transparency)
        ## The ORIGINAL single-period route: Fisher's g at the strongest
        ## ordinate (threshold g_F, i.e. lambda = 1); one claim per edge
        og = s["peaks"][0] if s["peaks"] else None
        og_claims.append(round(og["period"], 2)
                         if og is not None and og["Y"] >= s["gF"] else None)
    shared_seen = any(any(abs(p - P_S) < 0.05 for p in pk) for pk in single)
    ## --- combined route: candidates nominated by the union of per-edge
    ## leading ordinates (the two-stage logic of the detector itself), then
    ## trimmed Fisher over the matched windows at the candidates only ---
    fb = None; logp = None; noms = set(); peaks_by_edge = []
    for t in edges:
        fbe, P, n = spectrum(t, nb)
        R = local_ratio(P)
        loc = np.where((R[1:-1] > R[:-2]) & (R[1:-1] > R[2:]))[0] + 1
        top = loc[np.argsort(R[loc])[::-1][:12]]
        noms |= set(int(i) for i in top)
        peaks_by_edge.append(sorted(1.0 / fbe[i] for i in top))
        pe = window_p(P, n)
        logp = [np.log(pe)] if logp is None else logp + [np.log(pe)]
        fb = fbe
    ## Dedup nominations within a window width
    cand = []
    for i in sorted(noms):
        if cand and i - cand[-1][-1] <= 2 * WIN:
            cand[-1].append(i)
        else:
            cand.append([i])
    ## Trimmed Fisher: leave out the strongest edge at every frequency
    L = np.vstack(logp)
    X2 = -2.0 * (L.sum(axis=0) - L.min(axis=0))
    dof = 2 * (M - 1)
    z = ((X2 / dof) ** (1.0 / 3) - (1 - 2.0 / (9 * dof))) / math.sqrt(2.0 / (9 * dof))
    zbar = z_quantile(ALPHA / max(1, len(cand)))   ## Bonferroni over candidates
    combined = []
    for cl in cand:
        j = cl[int(np.argmax(z[cl]))]
        if z[j] <= zbar:
            continue
        pj = 1.0 / fb[j]
        ## single-edge coverage: pj is a detected fundamental, or the m-th
        ## harmonic of a fundamental whose edge also lists pj as contributing
        cov = False
        for pks, cpk in zip(single, peaks_by_edge):
            if any(abs(F - pj) / pj < 5e-3 for F in pks):
                cov = True; break
            if any(abs(F / m - pj) / pj < 5e-3 for F in pks for m in range(2, 7)) \
               and any(abs(q - pj) / pj < 5e-3 for q in cpk):
                cov = True; break
        combined.append({"period": round(float(pj), 4), "z": round(float(z[j]), 1),
                         "single_covered": bool(cov)})
    gained = [d["period"] for d in combined if not d["single_covered"]]
    res = {"tag": tag, "M": M,
           "single_edge_fundamentals": single,
           "shared_as_fundamental": bool(shared_seen),
           "shared_contributing_on": int(contrib),
           "original_claims": og_claims,
           "original_detects_shared": int(sum(1 for c in og_claims
                if c is not None and abs(c - P_S) < 0.05)),
           "combined_detections": combined, "gained_by_combination": gained,
           "z_bar": round(float(zbar), 2)}
    return res


if __name__ == "__main__":
    tag = sys.argv[1]
    datadir = sys.argv[2] if len(sys.argv) > 2 else \
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "Dataset", "lanl_mega")
    res = run(tag, datadir)
    print(res)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, f"{tag}.json"), "w") as fh:
        json.dump(res, fh, indent=1)
