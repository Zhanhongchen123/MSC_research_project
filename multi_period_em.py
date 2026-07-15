#!/usr/bin/env python3
import numpy as np
from numpy import pi
from scipy.stats import norm

####################################################################
## Multi-period uniform-wrapped normal mixture model fitted by EM ##
####################################################################

## Generalises the original mix_wrapped.em_wrap_mix (a single wrapped-normal +
## uniform mixture, SPH Sect. 4.1) to K periods, each on its own p_k-clock:
##    f(t_i) = (2*pi/T) [ theta_0 * (1/2pi) + sum_k theta_k * WN(x_ik; mu_k, sigma_k^2) ]
## This is the SIMPLIFIED model used only for fast K selection (no time-of-day
## step function, single phase per period).  It is fitted by EM and the
## log-likelihood is returned so that the caller can compute the BIC.
## Differences from the original:
## - K wrapped-normal components instead of 1, and each event has K wrapped
##   coordinates x_ik = 2*pi*(t_i mod p_k)/p_k
## - the mixing proportions are a (K+1)-vector (Dirichlet-style) instead of a scalar
## - when K = 1 this reduces exactly to the original EM, up to the dropped
##   time-of-day component, which is uniform here by construction


## Grid of wrap indices kappa in {-L,...,L}
def _wrap_grid(L):
    return np.arange(-L, L + 1)


## Transformation using the periodicities: X[i,k] = 2*pi*(t_i mod p_k)/p_k
def wrapped_coords(t, periods):
    t = np.asarray(t, dtype=float)
    P = np.asarray(periods, dtype=float)
    return (t[:, None] % P[None, :]) / P[None, :] * 2 * pi


#### Multi-period EM
## t is the vector of event times (seconds) and periods the list of K candidate
## periods (K may be 0).  T is the observation length in seconds, used for the
## additive log-density constant (None -> max(t)-min(t)); it is constant across
## K and therefore does not affect the K comparison.  L is the wrap-index
## truncation (kappa in -L..L).
## Returns a dict: mu (K,), sigma2 (K,), theta (K+1,), loglik, bic, K, n_params,
## resp (N, K+1), converged.
def fit_multiperiod_em(t, periods, T=None, L=10, max_iter=500, eps=1e-5,
                       mu_init=None, sigma2_init=0.5, verbose=False):
    t = np.asarray(t, dtype=float)
    N = len(t)
    K = len(periods)
    if T is None:
        T = float(np.max(t) - np.min(t)) or 1.0
    log_const = N * np.log(2 * pi / T)      ## constant in N and in K -> offset only

    ## K = 0: pure uniform human model
    if K == 0:
        loglik = N * np.log(1.0 / (2 * pi)) + log_const
        return {"mu": np.array([]), "sigma2": np.array([]),
                "theta": np.array([1.0]), "loglik": float(loglik),
                "bic": float(-2 * loglik + 0 * np.log(N)), "K": 0,
                "n_params": 0, "resp": np.ones((N, 1)), "converged": True}

    X = wrapped_coords(t, periods)          ## (N, K)
    kk = _wrap_grid(L)                       ## (2L+1,)

    ## Initialise the parameters
    mu = np.array(mu_init, dtype=float) if mu_init is not None else \
        np.array([X[:, k].mean() for k in range(K)])
    mu = mu % (2 * pi)
    sigma2 = np.full(K, float(sigma2_init))
    theta = np.full(K + 1, 1.0 / (K + 1))    ## [theta_1..theta_K, theta_0(human)]

    ## Return (WN, WN_by_wrap): WN[i,k] = sum_m N(x_ik + 2*pi*m; mu_k, sd_k),
    ## with WN_by_wrap[i,k,m] kept for the M-step unwrapping
    def wrapped_density_per_k(mu, sigma2):
        WN_by_wrap = np.zeros((N, K, len(kk)))
        for k in range(K):
            arg = X[:, k][:, None] + 2 * pi * kk[None, :]    ## (N, 2L+1)
            WN_by_wrap[:, k, :] = norm.pdf(arg, mu[k], np.sqrt(sigma2[k]))
        return WN_by_wrap.sum(axis=2), WN_by_wrap

    prev_loglik = -np.inf
    converged = False
    for it in range(max_iter):
        WN, WN_by_wrap = wrapped_density_per_k(mu, sigma2)        ## (N,K),(N,K,2L+1)

        ## E-step with normalised rows: responsibilities over {periods 1..K, human}
        comp = np.empty((N, K + 1))
        comp[:, :K] = theta[:K] * WN
        comp[:, K] = theta[K] * (1.0 / (2 * pi))
        row = comp.sum(axis=1, keepdims=True)
        resp = comp / np.where(row > 0, row, 1.0)                 ## (N, K+1)

        ## Circular log-likelihood (mixture density at each event's coordinate)
        loglik = float(np.sum(np.log(np.where(row[:, 0] > 0, row[:, 0], 1e-300)))) + log_const

        ## M-step: update the mixing proportions
        Nk = resp.sum(axis=0)                                     ## (K+1,)
        theta = Nk / N

        ## M-step: update mu_k and sigma2_k using the wrap-resolved weights
        for k in range(K):
            wk = WN_by_wrap[:, k, :]                              ## (N, 2L+1)
            denom = wk.sum(axis=1, keepdims=True)
            split = np.where(denom > 0, wk / denom, 0.0)          ## P(kappa | i,k)
            w = resp[:, k][:, None] * split                       ## (N, 2L+1) weights
            sw = w.sum()
            if sw <= 0:
                continue
            unwrapped = X[:, k][:, None] + 2 * pi * kk[None, :]    ## (N, 2L+1)
            mu_k = (w * unwrapped).sum() / sw
            sigma2_k = (w * (unwrapped - mu_k) ** 2).sum() / sw
            mu[k] = mu_k % (2 * pi)
            sigma2[k] = max(sigma2_k, 1e-6)

        ## Print iterations
        if verbose:
            print(f"  EM it{it:3d}  loglik={loglik:.3f}  theta={np.round(theta,3)}")
        ## Compute the change in the log-likelihood between two consecutive iterations
        if abs(loglik - prev_loglik) < eps:
            converged = True
            break
        prev_loglik = loglik

    ## BIC for the K comparison
    n_params = 3 * K                                              ## mu_k, sigma2_k, theta_k
    bic = -2 * loglik + n_params * np.log(N)
    ## Return parameters, responsibilities and information criteria
    return {"mu": mu, "sigma2": sigma2, "theta": theta, "loglik": float(loglik),
            "bic": float(bic), "K": K, "n_params": n_params,
            "resp": resp, "converged": converged}
