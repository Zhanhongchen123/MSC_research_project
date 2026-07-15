#!/usr/bin/env python3
import sys
import numpy as np
from numpy import pi, sqrt
from scipy.stats import norm
import cps_circle
import multi_period_em
import rj_phase

#######################################################################################################
## Collapsed Metropolis-within-Gibbs sampler with Reversible Jump steps on the phase peaks per clock ##
#######################################################################################################

## Sampler for the extended model: the number of phase peaks R_k of each clock is
## learnt by SAMPLING R_k, instead of fixing R and counting the non-empty peaks.
## - top-level human/clock allocation: collapsed (K+1)-way, as in the original model
## - within-clock peak weights omega: explicit (de-collapsed), updated by Gibbs
## - R_k: variable, moved every sweep by the split/combine and birth/death moves of
##   rj_phase.py (Richardson & Green, 1997), verified by its empty-run test
## - human daily density: cps_circle changepoint moves, reused verbatim
## Per sweep and per chain: (alloc) draw (z_i, r_i, kappa_i) for the re-allocated
## events, (NIG) shared-variance update of (mu, sigma2) per clock, (omega) Dirichlet
## update of the peak weights, (split/combine) and (birth/death) one attempt per
## clock, (human) one changepoint move on (tau, ell).  The posterior of R_k is the
## visit frequency of R_traj after burnin and the selected R_k is its mode.

TWO_PI = 2.0 * pi
DAY = 86400


#### Variable-R collapsed Gibbs sampler
## zeta0 is the scalar symmetric Dirichlet concentration per peak (RG97 use 1, i.e.
## uniform on the simplex).
## n_realloc is the number of events re-allocated per sweep (default None = all N).
## Full sweeps are a correctness requirement under sparse priors (zeta0 < 1), where
## partial sweeps cause occupancy lock-in; with zeta0 = 1 a partial sweep (e.g.
## N//20) merely mixes slower.
## mu_init optionally gives per-clock initial peak locations in radians, e.g. the
## modes of the folded-phase histogram, overriding R_init and the quantile seeding.
## Quantile seeding misplaces peaks when the mode masses are very skewed (two
## quantiles land in one dominant mode, the unseeded minor mode loses its events to
## the human component, the empty peak dies and R collapses); informed
## initialisation from the histogram modes is the standard remedy, and the sampler
## remains free to move, merge or kill peaks afterwards.
## Returns a dict including 'R_post' (per-clock posterior over R) and 'R_mode'.
def gibbs_rj_variable(t, periods, Rmax=6, R_init=3, n_samp=4000, n_chains=2,
                      L=10, mu0=pi, lambda0=1.0, alpha0=1.0, beta0=1.0,
                      gamma0=1.0, alpha_auto=None, zeta0=1.0, nu=0.1, eta=1.0,
                      burnin=2000, seed=None, verbose=True, init_iter=20000,
                      n_realloc=None, mu_init=None, trace_density=False, density_grid=240):
    ## Set the seed of the legacy numpy generator (used by the cps_circle moves)
    np.random.seed(None if seed is None else int(seed))
    ## Periods of the K clocks
    p = np.atleast_1d(np.asarray(periods, float))
    K = len(p)
    ## Dirichlet parameters of the top-level allocation weights
    if alpha_auto is None:
        alpha_auto = np.ones(K)
    alpha_auto = np.asarray(alpha_auto, float)
    N = len(t)
    ## Number of events re-allocated per sweep (default: full sweep)
    n_realloc = N if n_realloc is None else min(int(n_realloc), N)
    ## Grid for the optional posterior human density read-out
    dens_g = np.linspace(0, TWO_PI, int(density_grid), endpoint=False)
    dens_sum = np.zeros(int(density_grid)); dens_n = 0
    HUMAN = 0
    ## Grid of wrapping offsets 2*pi*kappa for kappa in {-L,...,L}
    kappa_grid = TWO_PI * np.arange(-L, L + 1)
    krange = np.arange(-L, L + 1)

    ## Transformation using the periodicities (folded angular positions x_ik)
    X = np.zeros((N, K))
    for k in range(K):
        X[:, k] = (t % p[k]) / p[k] * TWO_PI
    ## Transformation to daily arrival time
    y = np.array([obs % DAY / DAY * TWO_PI for obs in t])

    ## Obtain the optimised starting values of the parameters from the EM algorithm
    em = multi_period_em.fit_multiperiod_em(t, p, L=L)
    em_mu = np.asarray(em["mu"], float)
    em_s2 = np.asarray(em["sigma2"], float)
    ## Single-period EM fit used to seed the human/automated split
    em1 = multi_period_em.fit_multiperiod_em(t, [p[0]], L=L)
    human_seed = (em1["resp"].argmax(axis=1) == 1)

    ## Define the matrices containing the sampled values (post-burnin only)
    R_traj = np.zeros((n_samp, n_chains, K), "i")
    _snaps = []  ## (R, mu, omega) read-out per kept sweep; no use of the RNG
    ## Matrix of cluster allocations
    cluster = np.zeros((n_chains, N))
    cluster_k = np.zeros((n_chains, K, N))  ## per-clock allocation read-out
    s2_store = np.zeros((n_samp, n_chains, K))

    ## Loop over different chains
    for c in range(n_chains):
        ## Chain-specific generator for the Reversible Jump proposals
        rng = np.random.default_rng((0 if seed is None else int(seed)) * 131 + c + 1)

        ## Initial per-clock configuration: R_k peaks with a shared variance
        if mu_init is not None:
            R_cur = np.array([len(np.atleast_1d(m)) for m in mu_init], "i")
        else:
            R_cur = np.full(K, int(R_init))
        mu_cur, om_cur = [], []
        s2_cur = np.zeros(K)
        for k in range(K):
            if mu_init is not None:
                mk = np.sort(np.asarray(np.atleast_1d(mu_init[k]), float) % TWO_PI)
            else:
                mk = np.array([(em_mu[k] + rr * TWO_PI / R_cur[k]) % TWO_PI
                               for rr in range(R_cur[k])])
            mu_cur.append(mk)
            om_cur.append(np.full(R_cur[k], 1.0 / R_cur[k]))
            ## Cap the EM variance LOW: a single-Gaussian EM fit to a multi-phase
            ## clock returns one broad blob over all modes; even a 0.5 cap leaves
            ## peaks ~2 sd apart and the chain stays stuck at R=1 with sigma2
            ## drifting up, whereas a tight 0.2 cap starts the R_init peaks sharp
            ## and separable so that each can lock onto a mode
            s2k = em_s2[k] if (k < len(em_s2) and em_s2[k] > 0) else 0.5
            s2_cur[k] = float(min(max(s2k, 0.05), 0.2))

        ## Per-event initialisation: clock assignment, then data-driven peak
        ## seeding.  The EM gives one broad Gaussian per clock, and spreading the
        ## R peaks off its centre leaves a multi-phase clock stuck at R=1 with
        ## sigma2 drifting up, so: (1) assign each event to its best CLOCK,
        ## (2) seed that clock's R peaks at quantiles of its events' folded
        ## positions, which land in distinct modes, (3) attach each event to its
        ## nearest seeded peak
        z = np.zeros(N, "i"); r = np.zeros(N, "i"); kap = np.zeros(N, "i")
        for j in range(N):
            if human_seed[j]:
                continue
            best_v, bk = -1.0, 0
            for k in range(K):
                v = max(np.sum(norm.pdf(kappa_grid + X[j, k], mu_cur[k][rr], sqrt(s2_cur[k])))
                        for rr in range(R_cur[k]))
                if v > best_v:
                    best_v, bk = v, k
            z[j] = bk + 1
        for k in range(K):
            idx = np.where(z == (k + 1))[0]
            if mu_init is None and len(idx) >= R_cur[k]:
                pos = np.sort(X[idx, k])
                q = (((np.arange(R_cur[k]) + 0.5) / R_cur[k]) * len(pos)).astype(int)
                mu_cur[k] = pos[np.clip(q, 0, len(pos) - 1)] % TWO_PI
            for j in idx:
                d = np.minimum(np.abs(X[j, k] - mu_cur[k]), TWO_PI - np.abs(X[j, k] - mu_cur[k]))
                rr = int(d.argmin()); r[j] = rr
                grid = norm.pdf(kappa_grid + X[j, k], mu_cur[k][rr], sqrt(s2_cur[k]))
                kap[j] = int(krange[np.argmax(grid)])

        ## Initial choice of the changepoints -> run MCMC on the human subset
        ## Last sample is the initial value of tau
        y_daily = np.sort(y[z == HUMAN]); Nz = len(y_daily)
        tau = np.sort(np.random.uniform(0, TWO_PI, 2)); ell = 2
        y_positions = np.searchsorted(y_daily, tau)
        likelihoods = np.zeros(2)
        for _ in range(init_iter):
            mt = ["insert", "shift"] + (["delete"] if ell > 1 else [])
            m = np.random.choice(mt)
            fn = {"insert": cps_circle.propose_insert_changepoint,
                  "delete": cps_circle.propose_delete_changepoint,
                  "shift": cps_circle.propose_shift_changepoint}[m]
            _, tau, ell, y_positions, likelihoods = fn(tau, y_positions, y_daily, likelihoods, nu, eta)
        ## Bin positions of the daily arrival times and segment counts
        bin_pos = np.searchsorted(tau, y); bin_pos[bin_pos == len(tau)] = 0
        if Nz == 0:
            Nj = np.repeat(0, ell)
        else:
            y_positions = np.searchsorted(y_daily, tau)
            Nj = np.insert(np.diff(y_positions), 0, y_positions[0] + Nz - y_positions[ell - 1])
        likelihoods = cps_circle.tau_likelihood(tau, y_daily, y_positions, eta)[:]

        ## Vector of cluster counts
        Nk = np.zeros(K + 1)
        for j in range(N):
            Nk[z[j]] += 1

        ## Beginning of the MCMC (burnin followed by the kept sweeps)
        indices = np.arange(N)
        for i in range(burnin + n_samp):
            ## Print status of the chain
            if verbose:
                tag = "Burnin" if i < burnin else "Samples"
                num = i + 1 if i < burnin else i + 1 - burnin
                den = burnin if i < burnin else n_samp
                sys.stdout.write("\r+++ %s (chain %d/%d) +++ %d/%d  Rk=%s   "
                                 % (tag, c + 1, n_chains, num, den, list(R_cur)))
                sys.stdout.flush()

            ## Resample the allocations (z_i, r_i, kappa_i) of n_realloc events
            np.random.shuffle(indices)
            for kk in range(n_realloc):
                j = indices[kk]
                ## Segment of the daily density containing event j and its width
                zold = z[j]; bj = bin_pos[j]
                last = len(tau) - 1
                width = (tau[bj] - tau[bj - 1]) if bj != 0 else (TWO_PI - tau[last] + tau[bj])
                ## Collapsed predictive probability of the human component
                if zold == HUMAN:
                    prob_human = 1.0 / width * (eta * width + Nj[bj] - 1) / (TWO_PI * eta + Nz - 1)
                else:
                    prob_human = 1.0 / width * (eta * width + Nj[bj]) / (TWO_PI * eta + Nz)

                ## Counts with event j removed
                Nk_minus = Nk.copy(); Nk_minus[zold] -= 1
                ## Predictive weight of the human cell and of every (clock, peak) cell
                ncells = 1 + int(R_cur.sum())
                w = np.empty(ncells)
                w[0] = (Nk_minus[0] + gamma0) * prob_human
                kp_store = {}
                idx = 1
                for k in range(K):
                    clock_w = Nk_minus[k + 1] + alpha_auto[k]
                    for rr in range(R_cur[k]):
                        kp = norm.pdf(kappa_grid + X[j, k], mu_cur[k][rr], sqrt(s2_cur[k]))
                        w[idx] = clock_w * om_cur[k][rr] * kp.sum()
                        kp_store[idx] = (k, rr, kp)
                        idx += 1
                w /= w.sum()
                ## Sample the new allocation and, if automated, the wrapping kappa
                choice = int(np.random.choice(ncells, p=w))
                if choice == 0:
                    znew, rnew = HUMAN, 0
                else:
                    kk2, rr2, kp = kp_store[choice]
                    znew, rnew = kk2 + 1, rr2
                    kp = kp / kp.sum()
                    kap[j] = int(np.random.choice(krange, p=kp))
                rold = r[j]
                z[j], r[j] = znew, rnew

                ## Update the counts if the top-level allocation changed
                if znew != zold:
                    Nk[zold] -= 1; Nk[znew] += 1
                    if znew == HUMAN:
                        Nj[bj] += 1; Nz += 1
                    if zold == HUMAN:
                        Nj[bj] -= 1; Nz -= 1

                ## Occasional changepoint move within the sub-loop to refresh tau
                if kk % 50 == 0:
                    y_daily = np.sort(y[z == HUMAN])
                    y_positions = np.searchsorted(y_daily, tau)
                    mt = ["insert", "shift"] + (["delete"] if ell > 1 else [])
                    m = np.random.choice(mt)
                    fn = {"insert": cps_circle.propose_insert_changepoint,
                          "delete": cps_circle.propose_delete_changepoint,
                          "shift": cps_circle.propose_shift_changepoint}[m]
                    _, tau, ell, y_positions, likelihoods = fn(tau, y_positions, y_daily, likelihoods, nu, eta)
                    Nz = len(y_daily)
                    if Nz == 0:
                        Nj = np.repeat(0, ell)
                    else:
                        Nj = np.insert(np.diff(y_positions), 0, y_positions[0] + Nz - y_positions[ell - 1])
                    bin_pos = np.searchsorted(tau, y); bin_pos[bin_pos == len(tau)] = 0

            ## Update the parameters of the wrapped normals: NIG update with a
            ## shared variance per clock, then explicit update of the peak weights
            for k in range(K):
                Rk = R_cur[k]
                ## Sufficient statistics per peak on the unwrapped positions u = x + 2*pi*kappa
                Nkr = np.zeros(Rk, "i"); xbar = np.zeros(Rk); lam = np.zeros(Rk); ss = np.zeros(Rk)
                for rr in range(Rk):
                    Skr = (z == (k + 1)) & (r == rr)
                    nrr = int(Skr.sum()); Nkr[rr] = nrr; lam[rr] = lambda0 + nrr
                    if nrr > 0:
                        u = X[Skr, k] + TWO_PI * kap[Skr]
                        xbar[rr] = np.mean(u); ss[rr] = np.sum((u - xbar[rr]) ** 2)
                    else:
                        xbar[rr] = mu0
                ## Posterior parameters of the shared variance
                a_post = alpha0 + Nkr.sum() / 2.0
                b_post = beta0
                for rr in range(Rk):
                    if Nkr[rr] > 0:
                        b_post += 0.5 * ss[rr] + (Nkr[rr] * lambda0) / (2.0 * lam[rr]) * (xbar[rr] - mu0) ** 2
                ## Sample the shared variance, then the peak means
                s2 = 1.0 / np.random.gamma(a_post, 1.0 / b_post)
                s2_cur[k] = s2
                mk = np.zeros(Rk)
                for rr in range(Rk):
                    mp = (lambda0 * mu0 + Nkr[rr] * xbar[rr]) / lam[rr]
                    mk[rr] = np.random.normal(mp, sqrt(s2 / lam[rr])) % TWO_PI
                mu_cur[k] = mk
                ## Update the peak weights omega_k ~ Dirichlet(zeta0 + counts)
                om_cur[k] = np.random.dirichlet(zeta0 + Nkr)

            ## Reversible Jump moves on R_k: one split/combine and one birth/death
            ## attempt per clock (Richardson & Green, 1997)
            for k in range(K):
                sel = (z == (k + 1))
                x_clk = X[sel, k]
                lab_clk = r[sel].astype(int)
                Rk = R_cur[k]
                ## Split / combine move (changes R_k by one, with re-labelling)
                Rn, om_n, mu_n, lab_n, acc, kind = rj_phase.split_combine_move(
                    Rk, om_cur[k].copy(), mu_cur[k].copy(), s2_cur[k], x_clk, lab_clk,
                    zeta0=zeta0, Rmax=Rmax, mu0=mu0, lam0=lambda0, L=L, rng=rng)
                if acc:
                    R_cur[k] = Rn; om_cur[k] = om_n; mu_cur[k] = mu_n
                    r[sel] = lab_n
                    sel = (z == (k + 1)); x_clk = X[sel, k]; lab_clk = r[sel].astype(int); Rk = Rn
                ## Birth / death move on the empty peaks
                cnt = np.array([(lab_clk == rr).sum() for rr in range(Rk)]) if len(lab_clk) else np.zeros(Rk, int)
                empty_labels = np.where(cnt == 0)[0]
                Rn, om_n, mu_n, acc, kind, killed = rj_phase.birth_death_move(
                    Rk, om_cur[k].copy(), mu_cur[k].copy(), s2_cur[k],
                    int(len(x_clk)), int(len(empty_labels)),
                    zeta0=zeta0, Rmax=Rmax, mu0=mu0, lam0=lambda0, L=L, rng=rng,
                    empty_labels=empty_labels)
                if acc:
                    R_cur[k] = Rn; om_cur[k] = om_n; mu_cur[k] = mu_n
                    ## A death removes one empty peak: shift the labels above it
                    if kind == "death" and killed is not None:
                        sh = (z == (k + 1)) & (r > killed)
                        r[sh] -= 1

            ## Resample the changepoint configuration of the human daily density
            y_daily = np.sort(y[z == HUMAN])
            y_positions = np.searchsorted(y_daily, tau)
            likelihoods = cps_circle.tau_likelihood(tau, y_daily, y_positions, eta)[:]
            mt = ["insert", "shift"] + (["delete"] if ell > 1 else [])
            m = np.random.choice(mt)
            fn = {"insert": cps_circle.propose_insert_changepoint,
                  "delete": cps_circle.propose_delete_changepoint,
                  "shift": cps_circle.propose_shift_changepoint}[m]
            _, tau, ell, y_positions, likelihoods = fn(tau, y_positions, y_daily, likelihoods, nu, eta)
            Nz = len(y_daily)
            if Nz == 0:
                Nj = np.repeat(0, ell)
            else:
                y_positions = np.searchsorted(y_daily, tau)
                Nj = np.insert(np.diff(y_positions), 0, y_positions[0] + Nz - y_positions[ell - 1])
            bin_pos = np.searchsorted(tau, y); bin_pos[bin_pos == len(tau)] = 0

            ## Store the sampled values after burnin
            if i >= burnin:
                s = i - burnin
                R_traj[s, c] = R_cur
                s2_store[s, c] = s2_cur
                cluster[c] += (z != HUMAN)
                cluster_k[c] += (z[None, :] == np.arange(1, K + 1)[:, None])
                _snaps.append(([int(_x) for _x in R_cur],
                               [np.asarray(_m, float).copy() for _m in mu_cur],
                               [np.asarray(_w, float).copy() for _w in om_cur]))
                ## Accumulate the posterior mean of the human daily density
                if trace_density and ell >= 1:
                    m_seg = (Nj + eta) / (max(Nz, 0) + ell * eta)      ## segment masses
                    seg_len = np.empty(ell)
                    if ell == 1:
                        seg_len[0] = TWO_PI
                    else:
                        seg_len[0] = tau[0] + TWO_PI - tau[ell - 1]
                        seg_len[1:] = np.diff(tau)
                    gi = np.searchsorted(tau, dens_g); gi[gi == ell] = 0
                    dens_sum += (m_seg / seg_len)[gi]; dens_n += 1
        if verbose:
            sys.stdout.write("\n")

    ## Posterior probability of automated activity for each event
    cluster /= n_samp
    cluster_k /= n_samp
    ## Per-clock posterior over R and its mode
    R_post = np.zeros((K, Rmax + 1))
    flat = R_traj.reshape(-1, K)
    for k in range(K):
        bc = np.bincount(flat[:, k], minlength=Rmax + 1).astype(float)
        R_post[k] = bc / bc.sum()
    R_mode = R_post.argmax(axis=1)
    out_extra = {}
    if trace_density:
        out_extra = {"density_y": dens_g, "density": dens_sum / max(dens_n, 1)}
    ## Return output
    return {**out_extra, "R_traj": R_traj, "R_post": R_post, "R_mode": R_mode,
            "cluster": cluster, "sigma2": s2_store, "K": K, "periods": p,
            "Rmax": Rmax, "snaps": _snaps, "cluster_k": cluster_k}
