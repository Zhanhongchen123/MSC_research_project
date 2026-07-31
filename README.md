# Human-activity / automated-polling separation in network event times

Bayesian separation of automated periodic polling from human activity in network
timestamps. Extends Sanna Passino & Heard (2020) with (i) **Siegel (1980)**
multi-period detection and (ii) a **reversible-jump MCMC** that infers the number
of phase peaks R per clock.

Run:  `python3 test_model.py 0`  (from the repository root)

## Layout

- `*.py` and the caches (`bench_step.npz`, `e2_cache.npz`, `sweep1.json`, `sweep2.json`) at the root; run all commands from the root
- `Dataset/` -- the real event-time streams (`outlook.txt`, `lanl_edge2.txt`) and the multi-edge
  families of the combining-edges experiment (`lanl_mega/`: the synthetic fam/null runs and the real
  topology-defined family `realfam_edge*.txt`), found there by default by the test scripts
- `figures/` -- all generated figures (the scripts create the folder if missing)

## Files

The two model-fitting entry points are named to distinguish them:

- **`mcmc_original.py`** — the original single-clock model (Sanna Passino & Heard 2020), collapsed Gibbs (`collapsed_gibbs`, the M1 baseline).
- **`mcmc_improve.py`** — the improved model, RJMCMC over R (`gibbs_rj_variable`): joint classification + variable number of phase peaks per clock.

Everything else keeps its natural name:

| file | role |
|------|------|
| `fft_detect.py` | original single-period FFT detector (periodogram + Fisher g-test) |
| `siegel_periods.py` | **Siegel (1980)** multi-period detection (`T_lambda`, exact null eq 4.1, harmonic collapse); self-contained |
| `mix_wrapped.py` / `mix_wrapped_laplace.py` | wrapped-normal / wrapped-Laplace phase mixtures (used by both models) |
| `cps_circle.py` | piecewise-constant human daily density on the circle (changepoints) |
| `rj_phase.py` | split/combine & birth/death RJMCMC moves for R (used by `mcmc_improve`) |
| `multi_period_em.py` | EM fit that initialises the RJMCMC sampler |
| `simulating_data.py` | all data simulation: Donoho-Johnstone human densities (step/heavisine/bumps) and the 3-clock / 3-phase generator `simulate_3clock` |
| `test_model.py` | runs the full model on the three datasets; reports period / R / AUC |

## What `test_model.py` does

`simulating_data.simulate_3clock` builds three clocks (p = 10 s, R=3; 23 s, R=2;
37 s, R=1) on a chosen human background; `siegel_periods` recovers [10, 23, 37];
`mcmc_improve.gibbs_rj_variable` recovers R = [3, 2, 1] and classifies each event;
AUC compares the baseline `mcmc_original.collapsed_gibbs` (M1) against the
improved model.

### Reference result (seed 0)

```
density    |   period   |     R      | AUC base   ext    gain
step       | [10,23,37] | [3,2,1] ok |  0.703   0.832  +0.130
heavisine  | [10,23,37] | [3,2,1] ok |  0.543   0.701  +0.158
bumps      | [10,23,37] | [3,2,1] ok |  0.801   0.891  +0.091
```

`siegel_periods.siegel_exact_critical` reproduces Siegel's Table 1
(n = 10/20/50, lambda = .6/.8) to < 0.001.

## Module dependencies

```
fft_detect.py       scipy, numpy
mcmc_original.py    mix_wrapped, mix_wrapped_laplace, cps_circle
siegel_periods.py   scipy, numpy   (self-contained)
mcmc_improve.py     cps_circle, multi_period_em, rj_phase
simulating_data.py  numpy
test_model.py       simulating_data, siegel_periods, mcmc_original, mcmc_improve
```

Python 3; requires `numpy`, `scipy`, `scikit-learn`.

## References
- Sanna Passino, F. & Heard, N.A. (2020). Classification of periodic arrivals in event time data.
- Siegel, A.F. (1980). Testing for periodicity in a time series. *JASA* 75(370), 345-348.

## Consolidated additions (KS evaluation)
- ks_gof.py                self-contained nonparametric Wold KS (Price-Williams & Heard ref. code merged in)
- mcmc_improve.py          + n_realloc (full-sweep default) and mu_init (informed peak init) -- backward compatible
- test_outlook_ks.py       negative control: Outlook, degenerates K=1/R=1; KS 0.353 -> 0.213 (orig) / 0.216 (ours)
- make_multiedge_data.py   generates the synthetic multi-edge families (Dataset/lanl_mega/, deterministic)
- test_multiedge.py        combining edges: single-edge routes (original Fisher-g; the report's detector)
                           vs the trimmed-Fisher combination; tags fam0-2 / null0-2 / realfam
- timing_table.py          seconds per 1000 sweeps vs N (sampling loop only, two-point slope; hardware-dependent)
- test_lanl_edge2_ks.py    natural K=2 edge (data: Dataset/lanl_edge2.txt) Comp393078>Comp186884 (15s+20s); KS 0.341 -> 0.210 (orig) / 0.154 (ours)
All test scripts accept --smoke for a fast check; the data files live in `Dataset/`.

## Figures and tables -> scripts
- Table 3 (AUC, three densities): python3 test_model.py 0
- Fig 3 (folded phases) and Fig 4a (ROC): python3 test_model.py --bench then --figs (--bench writes bench_step.npz; --figs then renders from it in seconds)
- Fig 4b/c (sweeps): python3 test_model.py --sweep1 / --sweep2 then --figs (--sweep1 / --sweep2 write sweep1.json / sweep2.json)
- Table 4 (timing): python3 timing_table.py  (hardware-dependent; sampling loop only)
- Misspecification checks (Sect. 3.1): python3 test_misspec.py a 0|1|2  /  b 0|1|2
- Combining edges (Sect. 3.2): python3 test_multiedge.py fam0|fam1|fam2|null0|null1|null2|realfam
  (streams shipped in Dataset/lanl_mega/; the synthetic ones regenerate via python3 make_multiedge_data.py)
- Outlook row and figure: python3 test_outlook_ks.py
- LANL edge 2 row and figure: python3 test_lanl_edge2_ks.py  (staged: --stage=ku|base|full|fin, --fig; the ku/base/full stages build e2_cache.npz, after which --stage=fin --fig runs in seconds)
