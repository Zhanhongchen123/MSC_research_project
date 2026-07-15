#!/usr/bin/env python3
import numpy as np
from numpy import pi, sqrt, log
from scipy.special import gammaln

#######################################################################
## Reversible Jump moves on the number of phase peaks within a clock ##
#######################################################################

## Split/combine (RG97 eq 11) and birth/death (RG97 eq 12) specialised to the model:
## - the peaks SHARE one variance sigma2, so the split draws a 2-D u (no u3), the
##   sigma2 prior ratio drops and the Jacobian is |J| = omega_j * sigma / sqrt(u1(1-u1))
## - the means live on the circle [0,2pi): wrapped-normal densities; the combine
##   weighted mean is computed in a locally-unwrapped frame; pairs are chosen over
##   ALL pairs (adjacency is an acceptance-rate optimisation, not a correctness one)
## - the peak weights omega are explicit state (de-collapsed relative to the original
##   model) and labelled (no mean ordering); Beta(2,2) for u1 and u2 is symmetric so
##   the role of the two children in the combine is immaterial
## The two move functions are pure (they operate on one clock's arrays), so they can
## be unit-tested in isolation: `python3 rj_phase.py` runs the empty-run test below,
## which is the correctness gate for the acceptance ratios BEFORE they are wired
## into the sampler.

TWO_PI = 2.0 * pi


## Density of the wrapped normal at circular position(s) x: sum_kappa N(x + 2*pi*kappa; mu, s2)
def wn_pdf(x, mu, s2, L=10):
    x = np.atleast_1d(np.asarray(x, float))
    k = TWO_PI * np.arange(-L, L + 1)
    ## Shape (len(x), 2L+1)
    z = (x[:, None] + k[None, :] - mu)
    return np.exp(-0.5 * z * z / s2).sum(axis=1) / sqrt(TWO_PI * s2)


## Log wrapped-normal prior density of a peak mean: WN(mu0, s2/lam0)
def mu_logprior(mu, mu0, s2, lam0, L=10):
    v = s2 / lam0
    k = TWO_PI * np.arange(-L, L + 1)
    z = (mu + k - mu0)
    return log(np.exp(-0.5 * z * z / v).sum() / sqrt(TWO_PI * v))


## Beta(2,2) log-density: log(6 u (1-u))
def _beta22_logpdf(u):
    return log(6.0) + log(u) + log1p_safe(-u)


## Numerically safe log(1+x)
def log1p_safe(x):
    return np.log(np.maximum(1.0 + x, 1e-300))


## Split/birth probability b_R and combine/death probability d_R (RG97 convention)
def _bd_levels(R, Rmax):
    if R <= 1:
        return 1.0, 0.0
    if R >= Rmax:
        return 0.0, 1.0
    return 0.5, 0.5


## Representative of mu_b within pi of mu_a (on the real line)
def _circ_unwrap(mu_b, mu_a):
    d = (mu_b - mu_a + pi) % TWO_PI - pi
    return mu_a + d


#### Split / combine move (RG97 eq 11, shared-variance and circular specialisation)
## One split-or-combine attempt on a single clock, given the current state:
## omega (R,) peak weights, mu (R,) peak means, s2 shared variance (scalar),
## x_clk (n,) circular positions of this clock's events ([] if empty) and
## lab_clk (n,) current peak label in {0,...,R-1} for each event.
## Returns (R, omega, mu, lab_clk, accepted, kind) with kind in {"split","combine"}.
def split_combine_move(R, omega, mu, s2, x_clk, lab_clk,
                       zeta0=1.0, Rmax=8, mu0=pi, lam0=1.0, L=10, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    sigma = sqrt(s2)
    b, d = _bd_levels(R, Rmax)
    do_split = rng.random() < b

    if do_split:
        ## Draw the peak to split and the auxiliary variables u1, u2
        j = rng.integers(R)
        u1 = rng.beta(2, 2)
        u2 = rng.beta(2, 2)
        w1 = omega[j] * u1
        w2 = omega[j] * (1.0 - u1)
        root = sqrt(u1 * (1.0 - u1))
        half = u2 * sigma / root                       ## = |mu1 - mu2| on the line
        mu1 = (mu[j] - u2 * sigma * sqrt((1.0 - u1) / u1)) % TWO_PI
        mu2 = (mu[j] + u2 * sigma * sqrt(u1 / (1.0 - u1))) % TWO_PI

        ## Reallocate this peak's events between the two children
        mask = (lab_clk == j)
        idx = np.where(mask)[0]
        l1 = l2 = 0
        P_alloc_log = 0.0
        new_lab_for_idx = np.empty(len(idx), int)
        if len(idx):
            p1 = w1 * wn_pdf(x_clk[idx], mu1, s2, L)
            p2 = w2 * wn_pdf(x_clk[idx], mu2, s2, L)
            pr1 = p1 / (p1 + p2)
            draws = rng.random(len(idx))
            to1 = draws < pr1
            new_lab_for_idx[to1] = 0          ## first child -> keeps the slot of peak j
            new_lab_for_idx[~to1] = 1
            l1 = int(to1.sum()); l2 = int((~to1).sum())
            P_alloc_log = (np.log(pr1[to1]).sum() + np.log(1.0 - pr1[~to1]).sum())
            ## Likelihood ratio of the x positions
            lr = (np.log(np.where(to1, p1 / w1, p2 / w2)).sum()
                  - np.log(wn_pdf(x_clk[idx], mu[j], s2, L)).sum())
        else:
            lr = 0.0

        ## Weight prior and allocation likelihood (powers of omega)
        logw = ((zeta0 - 1 + l1) * log(w1) + (zeta0 - 1 + l2) * log(w2)
                - (zeta0 - 1 + l1 + l2) * log(omega[j]))
        logB = gammaln(zeta0) + gammaln(R * zeta0) - gammaln((R + 1) * zeta0)
        logw -= logB
        ## Mean prior ratio (wrapped normal); the R prior is uniform -> contributes 0
        logmu = mu_logprior(mu1, mu0, s2, lam0, L) + mu_logprior(mu2, mu0, s2, lam0, L) \
            - mu_logprior(mu[j], mu0, s2, lam0, L)
        ## Proposal ratio (reverse/forward): all-pairs combine in the (R+1) state
        b2, d2 = _bd_levels(R + 1, Rmax)
        n_pairs = (R + 1) * R / 2.0
        logprop = (log(d2) - log(n_pairs)) - (log(b) - log(R)
                   + _beta22_logpdf(u1) + _beta22_logpdf(u2) + P_alloc_log)
        ## Jacobian
        logJ = log(omega[j]) + log(sigma) - 0.5 * log(u1 * (1.0 - u1))

        ## Order-statistics / labelling factor (the "(k+1)" term of RG97 eq 11): the
        ## ratio (R+1)!/R! of labelled vs unordered multiplicities; omitting it makes
        ## the empty-run target pi(R) proportional to 1/R! instead of uniform
        logA = lr + logw + logmu + logprop + logJ + log(R + 1)
        if log(rng.random()) < logA:
            ## Accept: replace peak j by the first child and append the second
            omega = np.concatenate([omega, [w2]]); omega[j] = w1
            mu = np.concatenate([mu, [mu2]]);       mu[j] = mu1
            if len(idx):
                lab_clk = lab_clk.copy()
                lab_clk[idx[new_lab_for_idx == 1]] = R   ## second child = new last index
                ## The first child keeps label j
            return R + 1, omega, mu, lab_clk, True, "split"
        return R, omega, mu, lab_clk, False, "split"

    else:  ## combine R -> R-1
        ## Pick an unordered pair uniformly from all C(R,2) pairs
        a, b_ = _rand_pair(R, rng)
        wa, wb = omega[a], omega[b_]
        wj = wa + wb
        u1 = wa / wj
        mu_b_un = _circ_unwrap(mu[b_], mu[a])
        mu_j = (u1 * mu[a] + (1.0 - u1) * mu_b_un) % TWO_PI
        sigma = sqrt(s2)
        gap = abs(_circ_unwrap(mu[a], mu_j) - _circ_unwrap(mu[b_], mu_j))
        root = sqrt(u1 * (1.0 - u1))
        u2 = gap * root / sigma
        if not (0.0 < u2 < 1.0):
            return R, omega, mu, lab_clk, False, "combine"   ## outside the support of the reverse split

        ## Events of a and b merge into the combined peak
        mask = (lab_clk == a) | (lab_clk == b_)
        idx = np.where(mask)[0]
        la = int((lab_clk == a).sum()); lb = int((lab_clk == b_).sum())
        if len(idx):
            pa = wa * wn_pdf(x_clk[idx], mu[a], s2, L)
            pb = wb * wn_pdf(x_clk[idx], mu[b_], s2, L)
            pra = pa / (pa + pb)
            isa = (lab_clk[idx] == a)
            P_alloc_log = (np.log(pra[isa]).sum() + np.log(1.0 - pra[~isa]).sum())
            lr = (np.log(wn_pdf(x_clk[idx], mu_j, s2, L)).sum()
                  - np.log(np.where(isa, pa / wa, pb / wb)).sum())
        else:
            P_alloc_log = 0.0
            lr = 0.0

        ## Build the FORWARD (split) log-ratio for R-1 -> R, then accept with its inverse
        logw = ((zeta0 - 1 + la) * log(wa) + (zeta0 - 1 + lb) * log(wb)
                - (zeta0 - 1 + la + lb) * log(wj))
        logB = gammaln(zeta0) + gammaln((R - 1) * zeta0) - gammaln(R * zeta0)
        logw -= logB
        logmu = mu_logprior(mu[a], mu0, s2, lam0, L) + mu_logprior(mu[b_], mu0, s2, lam0, L) \
            - mu_logprior(mu_j, mu0, s2, lam0, L)
        bprev, dprev = _bd_levels(R - 1, Rmax)
        n_pairs = R * (R - 1) / 2.0
        logprop = (log(d) - log(n_pairs)) - (log(bprev) - log(R - 1)
                   + _beta22_logpdf(u1) + _beta22_logpdf(u2) + P_alloc_log)
        logJ = log(wj) + log(sigma) - 0.5 * log(u1 * (1.0 - u1))
        logA_split = lr + logw + logmu + logprop + logJ + log(R)   ## forward (R-1 -> R): factor R
        logA = -logA_split                                   ## the combine accepts with the inverse

        if log(rng.random()) < logA:
            keep = np.ones(R, bool); keep[a] = keep[b_] = False
            new_mu = np.concatenate([mu[keep], [mu_j]])
            new_om = np.concatenate([omega[keep], [wj]])
            ## Relabel the events: a and b map to the combined index, the others shift down
            old2new = -np.ones(R, int)
            old2new[keep] = np.arange(keep.sum())
            comb_idx = keep.sum()
            old2new[a] = old2new[b_] = comb_idx
            lab_clk = old2new[lab_clk]
            return R - 1, new_om, new_mu, lab_clk, True, "combine"
        return R, omega, mu, lab_clk, False, "combine"


## Draw an unordered pair of distinct labels uniformly at random
def _rand_pair(R, rng):
    a = rng.integers(R)
    b = rng.integers(R - 1)
    if b >= a:
        b += 1
    return (a, b) if a < b else (b, a)


#### Birth / death move (RG97 eq 12, empty component)
## One birth-or-death attempt on a single clock, where n_clk is the number of
## events allocated to this clock (all peaks), n_empty the number of currently
## empty peaks and empty_labels their indices (needed to pick one to kill).
## Returns (R, omega, mu, accepted, kind, killed_label_or_None).  Births are
## EMPTY (no reallocation), so the caller need not touch the labels on a birth;
## on a death the index of the killed (empty) peak is returned for relabelling.
def birth_death_move(R, omega, mu, s2, n_clk, n_empty,
                     zeta0=1.0, Rmax=8, mu0=pi, lam0=1.0, L=10, rng=None,
                     empty_labels=None):
    rng = np.random.default_rng() if rng is None else rng
    sigma = sqrt(s2)
    b, d = _bd_levels(R, Rmax)
    do_birth = rng.random() < b

    if do_birth:
        w_new = rng.beta(1.0, R)                    ## proposal q_b = Beta(1,R)
        mu_new = _draw_mu_prior(mu0, s2, lam0, rng)
        ## Posterior ratio (the mu prior cancels the proposal of mu): weight prior
        ## and allocation likelihood
        logpost = (-(gammaln(R * zeta0) + gammaln(zeta0) - gammaln((R + 1) * zeta0))
                   + (zeta0 - 1) * log(w_new)
                   + (R * zeta0 - R + n_clk) * log1p_safe(-w_new))
        ## Proposal ratio reverse (death) / forward (birth); after the birth there
        ## are (n_empty+1) empty peaks to choose from for the reverse death
        b2, d2 = _bd_levels(R + 1, Rmax)
        q_b_log = log(R) + (R - 1) * log1p_safe(-w_new)     ## Beta(1,R) log-density
        logprop = (log(d2) - log(n_empty + 1)) - (log(b) + q_b_log)
        logJ = (R - 1) * log1p_safe(-w_new)                 ## weight-rescale Jacobian
        logA = logpost + logprop + logJ + log(R + 1)        ## labelling factor, as in the split
        if log(rng.random()) < logA:
            omega = np.concatenate([omega * (1.0 - w_new), [w_new]])
            mu = np.concatenate([mu, [mu_new]])
            return R + 1, omega, mu, True, "birth", None
        return R, omega, mu, False, "birth", None

    else:  ## death of an empty peak
        if n_empty == 0:
            return R, omega, mu, False, "death", None
        kill = int(rng.choice(empty_labels))
        w_new = omega[kill]
        keep = np.ones(R, bool); keep[kill] = False
        ## Build the forward (birth R-1 -> R) log-ratio and accept with its inverse
        bprev, dprev = _bd_levels(R - 1, Rmax)
        logpost = (-(gammaln((R - 1) * zeta0) + gammaln(zeta0) - gammaln(R * zeta0))
                   + (zeta0 - 1) * log(w_new)
                   + ((R - 1) * zeta0 - (R - 1) + n_clk) * log1p_safe(-w_new))
        q_b_log = log(R - 1) + (R - 2) * log1p_safe(-w_new)
        logprop = (log(d) - log(n_empty)) - (log(bprev) + q_b_log)
        logJ = (R - 2) * log1p_safe(-w_new)
        logA_birth = logpost + logprop + logJ + log(R)      ## forward birth (R-1 -> R): factor R
        logA = -logA_birth
        if log(rng.random()) < logA:
            new_om = omega[keep] / (1.0 - w_new)
            new_mu = mu[keep]
            return R - 1, new_om, new_mu, True, "death", kill
        return R, omega, mu, False, "death", None


## Draw a peak mean from the wrapped-normal prior
def _draw_mu_prior(mu0, s2, lam0, rng):
    return float(rng.normal(mu0, sqrt(s2 / lam0)) % TWO_PI)


#### Empty-run test (the correctness gate for the acceptance ratios)
## With NO events the target reduces to the prior, so R must come out
## Uniform{1,...,Rmax}, omega ~ Dirichlet and mu ~ wrapped-normal prior.
## Run move-only sampling (plus prior Gibbs on omega and mu) on an empty clock
## and return the histogram of R
def _empty_run(move, n_iter=200000, Rmax=6, zeta0=1.0, s2=1.0, mu0=pi, lam0=1.0,
               seed=0):
    rng = np.random.default_rng(seed)
    R = 1
    omega = np.array([1.0])
    mu = np.array([_draw_mu_prior(mu0, s2, lam0, rng)])
    empty_x = np.zeros(0)
    empty_lab = np.zeros(0, int)
    Rcount = np.zeros(Rmax + 1)
    mu_samples = []
    for it in range(n_iter):
        ## Prior Gibbs (no data): omega ~ Dirichlet(zeta0), mu_r ~ wrapped-normal prior
        omega = rng.dirichlet(np.full(R, zeta0))
        mu = np.array([_draw_mu_prior(mu0, s2, lam0, rng) for _ in range(R)])
        if move == "sc":
            R, omega, mu, _, _, _ = split_combine_move(
                R, omega, mu, s2, empty_x, empty_lab,
                zeta0=zeta0, Rmax=Rmax, mu0=mu0, lam0=lam0, rng=rng)
        else:
            R, omega, mu, _, _, _ = birth_death_move(
                R, omega, mu, s2, 0, R,                 ## empty clock: all R peaks empty
                zeta0=zeta0, Rmax=Rmax, mu0=mu0, lam0=lam0, rng=rng,
                empty_labels=np.arange(R))
        if it > n_iter // 5:
            Rcount[R] += 1
            if R >= 1:
                mu_samples.append(mu[0])
    Rcount = Rcount[1:Rmax + 1]
    return Rcount / Rcount.sum(), np.array(mu_samples)


## Run the empty-run test for both move types and print PASS/FAIL
def main():
    Rmax = 6
    target = np.full(Rmax, 1.0 / Rmax)
    print(f"EMPTY-RUN TEST  (Rmax={Rmax}; target = Uniform = {1/Rmax:.3f} each)\n")
    for tag, mv in [("split/combine", "sc"), ("birth/death", "bd")]:
        hist, mus = _empty_run(mv, n_iter=300000, Rmax=Rmax, seed=1)
        tv = 0.5 * np.abs(hist - target).sum()        ## total-variation distance
        print(f"[{tag}]  R-hist = {np.round(hist, 3)}")
        print(f"            total-variation from uniform = {tv:.3f}   "
              f"{'PASS' if tv < 0.03 else 'FAIL'}")
        ## The mu marginal should be wrapped-normal(mu0, s2/lam0): check the mean
        if len(mus):
            print(f"            E[mu] = {mus.mean():.3f}  (prior mean = {pi:.3f})\n")
    print("If both PASS, the acceptance ratios satisfy detailed balance w.r.t. the")
    print("prior and are ready to wire into the sampler.")


if __name__ == "__main__":
    main()
