"""
Set-size experiment + comparison to Item-Limit (IL), Equal-Precision (EP),
Slots-plus-Averaging (SA), and Variable-Precision (VP) models from
Van den Berg, Ma, Eichele, Schjølberg, Sanders, & Woldorff (PNAS 2012;
doi:10.1073/pnas.1117465109), using the same Fisher-information mapping
J = κ I₁(κ)/I₀(κ), κ = Φ(J) as in the supplement.

For each set size we fit a uniform + Von Mises mixture to signed estimation
errors (Zhang & Luck, 2008): w = mixture weight on the Von Mises component;
CSD = circular SD of that component, sqrt(-2 ln(I₁(κ)/I₀(κ))).

Empirical summaries are compared to predictions from each model after
maximum-likelihood parameter estimation on pooled circular errors.
Figure layout follows Figure 4A of the PNAS paper (2×4: w and CSD × four models).
"""
from __future__ import annotations

import os
import sys
import argparse
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, brentq, differential_evolution
from scipy.special import i0e, i1e
from scipy.stats import vonmises, gamma as gamma_dist

# package imports (run from repo root or with src on path)
_SRC = os.path.dirname(os.path.abspath(__file__))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import generator
from simulation import run_simulation_trial, retrieve_color
import validation


# --- Fisher information ↔ Von Mises concentration (Eq. 2, supplement) ---


def J_from_kappa(kappa: float) -> float:
    if kappa <= 0:
        return 0.0
    # I₁/I₀ = i₁e/i₀e avoids overflow for large κ.
    return float(kappa * (i1e(kappa) / i0e(kappa)))


def kappa_from_J(J: float) -> float:
    """κ = Φ(J), inverse of J = κ I₁(κ)/I₀(κ)."""
    J = float(np.asarray(J).flat[0])
    if not np.isfinite(J) or J <= 0:
        return 1e-8
    lo = 1e-12
    flo = J_from_kappa(lo) - J
    hi = 50.0
    while hi < 1e8:
        fhi = J_from_kappa(hi) - J
        if flo * fhi <= 0:
            break
        hi *= 2
    else:
        return max(hi / 2, 1e-8)
    return float(brentq(lambda k: J_from_kappa(k) - J, lo, hi, xtol=1e-9, rtol=1e-7))


def csd_from_kappa(kappa: float) -> float:
    """Circular SD from Von Mises concentration (Mardia & Jupp)."""
    if kappa <= 0:
        return np.sqrt(2 * np.log(2))  # undefined limit; avoid NaN
    r = float(i1e(kappa) / i0e(kappa))
    r = max(r, 1e-12)
    return float(np.sqrt(-2.0 * np.log(r)))


def vm_convolution_density(err: np.ndarray, kappa_mem: float, kappa_r: float) -> np.ndarray:
    """
    Angular error x = r - s density when estimate ~ VM(κ_mem) and response noise ~ VM(κ_r)
    (Eq. S11 / convolution form in supplement).
    """
    kappa_mem = max(kappa_mem, 1e-10)
    kappa_r = max(kappa_r, 1e-10)
    err = np.asarray(err, dtype=float)
    rho = np.sqrt(
        kappa_mem**2 + kappa_r**2 + 2.0 * kappa_mem * kappa_r * np.cos(err)
    )
    # Use_log unstable at large κ; ratio via exponentials cancels.
    num = i0e(rho) * np.exp(rho - kappa_mem - kappa_r)
    den = (2.0 * np.pi) * i0e(kappa_mem) * i0e(kappa_r)
    return num / den


# --- Model log-densities for MLE ---


def log_density_il(err: np.ndarray, N: int, K: float, kappa_mem: float, kappa_r: float) -> np.ndarray:
    w = min(1.0, max(0.0, K / N))
    vm_d = vm_convolution_density(err, kappa_mem, kappa_r)
    if w >= 1.0 - 1e-12:
        return np.log(np.clip(vm_d, 1e-300, None))
    mix = w * vm_d + (1.0 - w) / (2.0 * np.pi)
    return np.log(np.clip(mix, 1e-300, None))


def log_density_ep(
    err: np.ndarray,
    N: int,
    J1: float,
    alpha: float,
    kappa_r: float,
) -> np.ndarray:
    """EP-A: all items always remembered; κ_mem = Φ(J₁ N^α)."""
    J = J1 * (N ** alpha)
    kappa_mem = kappa_from_J(max(J, 1e-12))
    vm_d = vm_convolution_density(err, kappa_mem, kappa_r)
    return np.log(np.clip(vm_d, 1e-300, None))


def log_density_sa(
    err: np.ndarray,
    N: int,
    K: float,
    J1: float,
    kappa_r: float,
) -> np.ndarray:
    kappa1 = kappa_from_J(max(J1, 1e-12))
    Kn = int(np.clip(round(K), 1, 64))

    if Kn <= N:
        w = min(1.0, Kn / N)
        vm_d = vm_convolution_density(err, kappa1, kappa_r)
        if w >= 1.0 - 1e-12:
            return np.log(np.clip(vm_d, 1e-300, None))
        mix = w * vm_d + (1.0 - w) / (2.0 * np.pi)
        return np.log(np.clip(mix, 1e-300, None))

    n_high = Kn % N
    n_low = N - n_high
    base = Kn // N
    if n_high == 0:
        kappa_est = kappa_from_J(max(J1 * base, 1e-12))
        vm_d = vm_convolution_density(err, kappa_est, kappa_r)
        return np.log(np.clip(vm_d, 1e-300, None))

    j_lo = J1 * base
    j_hi = J1 * (base + 1)
    kappa_low = kappa_from_J(max(j_lo, 1e-12))
    kappa_high = kappa_from_J(max(j_hi, 1e-12))
    w_hi = n_high / N
    w_lo = n_low / N
    p_hi = vm_convolution_density(err, kappa_high, kappa_r)
    p_lo = vm_convolution_density(err, kappa_low, kappa_r)
    mix = w_hi * p_hi + w_lo * p_lo
    return np.log(np.clip(mix, 1e-300, None))


def _gamma_J_grid(Jbar: float, tau: float, n_q: int = 96) -> np.ndarray:
    """Deterministic quadrature points for J ~ Gamma(mean=J̄, scale τ)."""
    Jbar = max(Jbar, 1e-8)
    tau = max(tau, 1e-8)
    shape = max(Jbar / tau, 1e-6)
    scale = tau
    qs = np.linspace(0.002, 0.998, n_q)
    return gamma_dist.ppf(qs, a=shape, scale=scale)


def log_density_vp(
    err: np.ndarray,
    N: int,
    J1_bar: float,
    alpha: float,
    tau: float,
    kappa_r: float,
    n_q: int = 96,
) -> np.ndarray:
    """VP-A (K=∞): average over J with mean J̄ = J̄₁ N^α, Gamma scale τ."""
    Jbar = J1_bar * (N ** alpha)
    Js = _gamma_J_grid(Jbar, tau, n_q=n_q)
    Js = np.clip(np.asarray(Js, float), 1e-10, None)
    err = np.asarray(err, dtype=float)
    out = np.zeros_like(err)
    for jv in Js:
        k_mem = kappa_from_J(float(jv))
        out += vm_convolution_density(err, k_mem, kappa_r)
    out /= len(Js)
    return np.log(np.clip(out, 1e-300, None))


# --- Uniform + Von Mises mixture (summary statistics) ---


def _neg_loglik_mixture(params: np.ndarray, data: np.ndarray) -> float:
    w, kappa = params
    w = np.clip(w, 1e-5, 1.0 - 1e-5)
    kappa = max(kappa, 1e-6)
    u = np.full_like(data, 1.0 / (2.0 * np.pi))
    vm = vonmises.pdf(x=data, kappa=kappa, scale=1.0)
    mix = (1.0 - w) * u + w * vm
    return -float(np.sum(np.log(np.clip(mix, 1e-300, None))))


def fit_uniform_vm_mixture(errors_rad: np.ndarray) -> Tuple[float, float, float]:
    """
    MLE for p(e) = (1-w)/(2π) + w VM(e; 0, κ).
    Returns (w, kappa, csd).
    """
    data = np.asarray(errors_rad, dtype=float)
    data = (data + np.pi) % (2.0 * np.pi) - np.pi

    best = (np.inf, None)
    for w0 in (0.4, 0.65, 0.85, 0.95):
        for k0 in (1.0, 6.0, 25.0, 90.0):
            res = minimize(
                _neg_loglik_mixture,
                x0=np.array([w0, k0]),
                args=(data,),
                method="L-BFGS-B",
                bounds=((1e-5, 1.0 - 1e-5), (1e-4, 500.0)),
            )
            if res.fun < best[0]:
                best = (res.fun, res.x)
    if best[1] is None:
        raise RuntimeError("mixture fit failed")
    w, kappa = float(best[1][0]), float(best[1][1])
    return w, kappa, csd_from_kappa(kappa)


# --- Simulate errors from fitted models (for ribbons / checks) ---


def sample_errors_il(
    rng: np.random.Generator, N: int, K: float, kappa_mem: float, kappa_r: float, n: int
) -> np.ndarray:
    out = np.empty(n)
    w = min(1.0, K / N)
    for i in range(n):
        if rng.random() < w:
            s_hat = vonmises.rvs(kappa_mem, random_state=rng)
            noise = vonmises.rvs(kappa_r, random_state=rng)
            out[i] = (s_hat + noise + np.pi) % (2.0 * np.pi) - np.pi
        else:
            out[i] = rng.uniform(-np.pi, np.pi)
    return out


def sample_errors_ep(
    rng: np.random.Generator, N: int, J1: float, alpha: float, kappa_r: float, n: int
) -> np.ndarray:
    J = J1 * (N ** alpha)
    kappa_mem = kappa_from_J(max(J, 1e-12))
    s_hat = vonmises.rvs(kappa_mem, size=n, random_state=rng)
    noise = vonmises.rvs(kappa_r, size=n, random_state=rng)
    return (s_hat + noise + np.pi) % (2.0 * np.pi) - np.pi


def sample_errors_sa(
    rng: np.random.Generator,
    N: int,
    K: float,
    J1: float,
    kappa_r: float,
    n: int,
) -> np.ndarray:
    kappa1 = kappa_from_J(max(J1, 1e-12))
    Kn = int(np.clip(round(K), 1, 64))
    out = np.empty(n)
    for i in range(n):
        if Kn <= N:
            w = min(1.0, Kn / N)
            if rng.random() < w:
                s_hat = vonmises.rvs(kappa1, random_state=rng)
                noise = vonmises.rvs(kappa_r, random_state=rng)
                out[i] = (s_hat + noise + np.pi) % (2.0 * np.pi) - np.pi
            else:
                out[i] = rng.uniform(-np.pi, np.pi)
            continue
        n_high = Kn % N
        if n_high == 0:
            base = Kn // N
            k_est = kappa_from_J(max(J1 * base, 1e-12))
            s_hat = vonmises.rvs(k_est, random_state=rng)
            noise = vonmises.rvs(kappa_r, random_state=rng)
            out[i] = (s_hat + noise + np.pi) % (2.0 * np.pi) - np.pi
            continue
        base = Kn // N
        j_lo = J1 * base
        j_hi = J1 * (base + 1)
        k_lo = kappa_from_J(max(j_lo, 1e-12))
        k_hi = kappa_from_J(max(j_hi, 1e-12))
        if rng.random() < n_high / N:
            s_hat = vonmises.rvs(k_hi, random_state=rng)
        else:
            s_hat = vonmises.rvs(k_lo, random_state=rng)
        noise = vonmises.rvs(kappa_r, random_state=rng)
        out[i] = (s_hat + noise + np.pi) % (2.0 * np.pi) - np.pi
    return out


def sample_errors_vp(
    rng: np.random.Generator,
    N: int,
    J1_bar: float,
    alpha: float,
    tau: float,
    kappa_r: float,
    n: int,
) -> np.ndarray:
    Jbar = J1_bar * (N ** alpha)
    shape = max(Jbar / max(tau, 1e-8), 1e-6)
    scale = max(tau, 1e-8)
    Js = rng.gamma(shape, scale, size=n)
    out = np.empty(n)
    for i, jv in enumerate(Js):
        k_mem = kappa_from_J(max(float(jv), 1e-12))
        s_hat = vonmises.rvs(k_mem, random_state=rng)
        noise = vonmises.rvs(kappa_r, random_state=rng)
        out[i] = (s_hat + noise + np.pi) % (2.0 * np.pi) - np.pi
    return out


# --- Collect MemGP experiment errors ---


def collect_set_size_errors(config: dict) -> Dict[int, np.ndarray]:
    """Signed errors in radians, keyed by set size."""
    from tqdm import trange

    set_sizes = list(config["experiment"]["set_sizes"])
    n_trials = int(config["experiment"]["n_trials"])
    base_seed = int(config["experiment"]["random_seed"])
    errors_per_n: Dict[int, List[float]] = {n: [] for n in set_sizes}

    for n_items in set_sizes:
        for t in trange(n_trials, desc=f"MemGP trials N={n_items}"):
            seed = base_seed + t * 10007 + n_items
            items = generator.generate_items(n_items, seed=seed)
            model, _, _ = run_simulation_trial(items, config, cued_item_idx=None, track_visuals=False)
            for i in range(len(items)):
                signed_deg = retrieve_color(model, items[i][0], items[i][1])
                rad = np.deg2rad(float(signed_deg))
                rad = (rad + np.pi) % (2.0 * np.pi) - np.pi
                errors_per_n[n_items].append(rad)

    return {n: np.asarray(v, dtype=float) for n, v in errors_per_n.items()}


@dataclass
class FitResult:
    name: str
    params: np.ndarray
    neg_ll: float
    x_pred_w: np.ndarray
    x_pred_csd: np.ndarray
    w_ribbon_lo: np.ndarray
    w_ribbon_hi: np.ndarray
    csd_ribbon_lo: np.ndarray
    csd_ribbon_hi: np.ndarray


def _pool_data(errors_by_n: Dict[int, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    xs = []
    ns = []
    for N in sorted(errors_by_n.keys()):
        e = errors_by_n[N]
        xs.append(e)
        ns.append(np.full(len(e), N))
    return np.concatenate(xs), np.concatenate(ns)


def mle_objective(
    theta: np.ndarray,
    err: np.ndarray,
    n_arr: np.ndarray,
    logpdf: Callable[[np.ndarray, int], np.ndarray],
) -> float:
    ll = 0.0
    for N in np.unique(n_arr):
        mask = n_arr == N
        ll += float(np.sum(logpdf(err[mask], int(N))))
    return -ll


def fit_all_models(
    errors_by_n: Dict[int, np.ndarray],
    set_sizes: List[int],
    n_synth_avg: int = 10,
    seed: int = 0,
    de_maxiter: int = 60,
) -> Tuple[Dict[int, Tuple[float, float, float, float]], Dict[str, FitResult]]:
    """
    Returns:
      empirical: N -> (w, csd, w_boot_sd, csd_boot_sd)
      fits: model name -> FitResult with mean predicted w/csd and ribbons
    """
    rng = np.random.default_rng(seed)

    # Empirical mixture + bootstrap SEM
    empirical: Dict[int, Tuple[float, float, float, float]] = {}
    n_boot = 80
    for N in set_sizes:
        e = errors_by_n[N]
        w, kappa, csd = fit_uniform_vm_mixture(e)
        ws, cs = [], []
        m = len(e)
        for _ in range(n_boot):
            idx = rng.integers(0, m, size=m)
            wb, _, cb = fit_uniform_vm_mixture(e[idx])
            ws.append(wb)
            cs.append(cb)
        empirical[N] = (w, csd, float(np.std(ws, ddof=1)), float(np.std(cs, ddof=1)))

    err_all, n_all = _pool_data(errors_by_n) 
    trials_per_n = {N: len(errors_by_n[N]) for N in set_sizes}

    def pred_curves(
        sampler: Callable[[np.random.Generator, int, int], np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Mean w, mean CSD, and 10–90% ribbons from synthetic replicate mixture fits."""
        Ws = []
        Cs = []
        for _ in range(n_synth_avg):
            row_w = []
            row_c = []
            for N in set_sizes:
                syn = sampler(rng, N, trials_per_n[N])
                ww, _, cc = fit_uniform_vm_mixture(syn)
                row_w.append(ww)
                row_c.append(cc)
            Ws.append(row_w)
            Cs.append(row_c)
        Ws = np.asarray(Ws)
        Cs = np.asarray(Cs)
        w_mean = Ws.mean(axis=0)
        c_mean = Cs.mean(axis=0)
        w_lo = np.percentile(Ws, 10, axis=0)
        w_hi = np.percentile(Ws, 90, axis=0)
        c_lo = np.percentile(Cs, 10, axis=0)
        c_hi = np.percentile(Cs, 90, axis=0)
        return w_mean, c_mean, w_lo, w_hi, c_lo, c_hi

    fits: Dict[str, FitResult] = {}

    # --- IL ---
    def pack_il(x):
        K, lk_mem, lk_r = x
        return float(np.clip(K, 0.5, 20.0)), float(np.exp(lk_mem)), float(np.exp(lk_r))

    def obj_il(xx):
        K, km, kr = pack_il(xx)
        def logp(e, N):
            return log_density_il(e, N, K, km, kr)
        return mle_objective(xx, err_all, n_all, lambda e, N: logp(e, N))

    bounds_il = [(0.5, 12.0), (-2.5, 4.5), (-2.5, 4.5)]
    r_il = differential_evolution(
        lambda x: obj_il(x), bounds_il, seed=seed + 1, maxiter=de_maxiter, tol=0.05, polish=True
    )
    K_il, km_il, kr_il = pack_il(r_il.x)

    wm, cm, wl, wh, cl, ch = pred_curves(
        lambda rg, N, ntr: sample_errors_il(rg, N, K_il, km_il, kr_il, ntr),
    )
    fits["IL"] = FitResult(
        "IL", np.array([K_il, km_il, kr_il]), float(r_il.fun),
        wm, cm, wl, wh, cl, ch,
    )

    # --- SA ---
    def pack_sa(x):
        K, lJ1, lk_r = x
        return float(np.clip(K, 0.5, 20.0)), float(np.exp(lJ1)), float(np.exp(lk_r))

    def obj_sa(xx):
        K, J1, kr = pack_sa(xx)
        def logp(e, N):
            return log_density_sa(e, N, K, J1, kr)
        return mle_objective(xx, err_all, n_all, lambda e, N: logp(e, N))

    bounds_sa = [(0.5, 12.0), (-2.0, 4.5), (-2.5, 4.5)]
    r_sa = differential_evolution(
        lambda x: obj_sa(x), bounds_sa, seed=seed + 2, maxiter=de_maxiter, tol=0.05, polish=True
    )
    K_sa, J1_sa, kr_sa = pack_sa(r_sa.x)

    wm, cm, wl, wh, cl, ch = pred_curves(
        lambda rg, N, ntr: sample_errors_sa(rg, N, K_sa, J1_sa, kr_sa, ntr),
    )
    fits["SA"] = FitResult(
        "SA", np.array([K_sa, J1_sa, kr_sa]), float(r_sa.fun),
        wm, cm, wl, wh, cl, ch,
    )

    # --- EP ---
    def pack_ep(x):
        lJ1, alpha, lk_r = x
        return float(np.exp(lJ1)), float(alpha), float(np.exp(lk_r))

    def obj_ep(xx):
        J1, al, kr = pack_ep(xx)
        def logp(e, N):
            return log_density_ep(e, N, J1, al, kr)
        return mle_objective(xx, err_all, n_all, lambda e, N: logp(e, N))

    bounds_ep = [(-1.5, 4.5), (-2.5, 1.5), (-2.5, 4.5)]
    r_ep = differential_evolution(
        lambda x: obj_ep(x), bounds_ep, seed=seed + 3, maxiter=de_maxiter, tol=0.05, polish=True
    )
    J1_ep, al_ep, kr_ep = pack_ep(r_ep.x)

    wm, cm, wl, wh, cl, ch = pred_curves(
        lambda rg, N, ntr: sample_errors_ep(rg, N, J1_ep, al_ep, kr_ep, ntr),
    )
    fits["EP"] = FitResult(
        "EP", np.array([J1_ep, al_ep, kr_ep]), float(r_ep.fun),
        wm, cm, wl, wh, cl, ch,
    )

    # --- VP ---
    def pack_vp(x):
        lj1, alpha, ltau, lk_r = x
        return float(np.exp(lj1)), float(alpha), float(np.exp(ltau)), float(np.exp(lk_r))

    def obj_vp(xx):
        J1b, al, tau, kr = pack_vp(xx)

        def logp(e, N):
            return log_density_vp(e, N, J1b, al, tau, kr, n_q=72)

        return mle_objective(xx, err_all, n_all, lambda e, N: logp(e, N))

    bounds_vp = [(-1.5, 4.5), (-2.5, 1.5), (-3.0, 2.0), (-2.5, 4.5)]
    r_vp = differential_evolution(
        lambda x: obj_vp(x),
        bounds_vp,
        seed=seed + 4,
        maxiter=max(de_maxiter - 5, 20),
        tol=0.06,
        polish=True,
    )
    J1_vp, al_vp, tau_vp, kr_vp = pack_vp(r_vp.x)

    wm, cm, wl, wh, cl, ch = pred_curves(
        lambda rg, N, ntr: sample_errors_vp(rg, N, J1_vp, al_vp, tau_vp, kr_vp, ntr),
    )
    fits["VP"] = FitResult(
        "VP", np.array([J1_vp, al_vp, tau_vp, kr_vp]), float(r_vp.fun),
        wm, cm, wl, wh, cl, ch,
    )

    return empirical, fits


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


# --- BIC / AIC model comparison ---

_MODEL_N_PARAMS = {"IL": 3, "SA": 3, "EP": 3, "VP": 4}


def compute_information_criteria(
    fits: Dict[str, FitResult],
    errors_by_n: Dict[int, np.ndarray],
) -> Dict[str, Dict[str, float]]:
    """
    Compute AIC, BIC, and AICc for each fitted model.

    Parameters
    ----------
    fits : dict of FitResult
    errors_by_n : dict mapping set size -> array of errors (radians)

    Returns
    -------
    dict : model_name -> {"neg_ll", "k", "n", "AIC", "BIC", "AICc", "delta_AIC", "delta_BIC"}
    """
    n_total = sum(len(v) for v in errors_by_n.values())
    results = {}
    for name, fr in fits.items():
        k = _MODEL_N_PARAMS[name]
        nll = fr.neg_ll
        ll = -nll

        aic = 2 * k - 2 * ll
        bic = k * np.log(n_total) - 2 * ll
        if n_total - k - 1 > 0:
            aicc = aic + (2 * k * (k + 1)) / (n_total - k - 1)
        else:
            aicc = aic

        results[name] = {
            "neg_ll": nll,
            "k": k,
            "n": n_total,
            "AIC": aic,
            "BIC": bic,
            "AICc": aicc,
        }

    min_aic = min(r["AIC"] for r in results.values())
    min_bic = min(r["BIC"] for r in results.values())
    for name in results:
        results[name]["delta_AIC"] = results[name]["AIC"] - min_aic
        results[name]["delta_BIC"] = results[name]["BIC"] - min_bic

    return results


def plot_figure4a_fixed(
    set_sizes: List[int],
    empirical: Dict[int, Tuple[float, float, float, float]],
    fits: Dict[str, FitResult],
    save_path: str,
    title_suffix: str = "",
):
    """Single RMSE text per panel without duplicate labels."""
    models = ["IL", "SA", "EP", "VP"]
    colors = {"IL": "#c0392b", "SA": "#27ae60", "EP": "#f1c40f", "VP": "#2980b9"}
    labels = {"IL": "IL", "SA": "SA", "EP": "EP", "VP": "VP"}
    xs = np.asarray(set_sizes, dtype=float)
    w_emp = np.array([empirical[N][0] for N in set_sizes])
    c_emp = np.array([empirical[N][1] for N in set_sizes])
    w_sem = np.array([empirical[N][2] for N in set_sizes])
    c_sem = np.array([empirical[N][3] for N in set_sizes])
    rms_w = [rmse(w_emp, fits[m].x_pred_w) for m in models]
    rms_c = [rmse(c_emp, fits[m].x_pred_csd) for m in models]
    best_w = np.min(rms_w)
    best_c = np.min(rms_c)

    fig, axes = plt.subplots(2, 4, figsize=(12.5, 6.0), sharex=True)
    for col, mid in enumerate(models):
        fr = fits[mid]
        axw, axc = axes[0, col], axes[1, col]
        axw.fill_between(xs, fr.w_ribbon_lo, fr.w_ribbon_hi, color=colors[mid], alpha=0.35, linewidth=0)
        axw.plot(xs, fr.x_pred_w, color=colors[mid], linewidth=2)
        axw.errorbar(xs, w_emp, yerr=w_sem, fmt="o", color="black", capsize=3, markersize=5, markerfacecolor="white")
        rw = rms_w[col]
        kw = {"fontweight": "bold"} if rw == best_w else {}
        axw.text(0.97, 0.06, f"RMSE\n{rw:.3f}", transform=axw.transAxes, ha="right", va="bottom", fontsize=9, **kw)

        axc.fill_between(xs, fr.csd_ribbon_lo, fr.csd_ribbon_hi, color=colors[mid], alpha=0.35, linewidth=0)
        axc.plot(xs, fr.x_pred_csd, color=colors[mid], linewidth=2)
        axc.errorbar(xs, c_emp, yerr=c_sem, fmt="o", color="black", capsize=3, markersize=5, markerfacecolor="white")
        rc = rms_c[col]
        kc = {"fontweight": "bold"} if rc == best_c else {}
        axc.text(0.97, 0.06, f"RMSE\n{rc:.3f}", transform=axc.transAxes, ha="right", va="bottom", fontsize=9, **kc)

        axw.set_ylim(0, 1.05)
        axc.set_ylim(0.05, None)
        mn, mx = min(set_sizes), max(set_sizes)
        axw.set_xlim(mn - 0.2, mx + 0.2)
        axc.set_xlim(mn - 0.2, mx + 0.2)
        axw.set_xticks(set_sizes)
        axc.set_xticks(set_sizes)
        axw.set_title(labels[mid], color=colors[mid], fontweight="bold")
        if col == 0:
            axw.set_ylabel("$w$")
            axc.set_ylabel("CSD (rad)")
        axc.set_xlabel("Set size")
    fig.suptitle("Uniform–Von Mises mixture vs set size (MemGP sim) " + title_suffix, fontsize=12)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved figure to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="PNAS-style WM model comparison on set-size errors.")
    parser.add_argument(
        "--config",
        default="config_set_size.yaml",
        help="YAML filename or path under src/config/ (e.g. config_set_size.yaml).",
    )
    parser.add_argument("--out_dir", default="visualizations", help="Output directory (under cwd or src parent)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--n_trials",
        type=int,
        default=None,
        help="Override config experiment.n_trials (useful for quick test runs).",
    )
    parser.add_argument("--de_maxiter", type=int, default=None, help="Max iterations for differential_evolution.")
    args = parser.parse_args()
    cfg_path = (
        args.config
        if os.path.isabs(args.config)
        else os.path.join(_SRC, "config", args.config)
    )
    config = validation.load_config(path=cfg_path)
    if args.n_trials is not None:
        config = dict(config)
        config["experiment"] = dict(config["experiment"])
        config["experiment"]["n_trials"] = int(args.n_trials)
    set_sizes = [int(x) for x in config["experiment"]["set_sizes"]]

    out_dir = args.out_dir
    if not os.path.isabs(out_dir):
        out_dir = os.path.normpath(os.path.join(_SRC, "..", out_dir))
    os.makedirs(out_dir, exist_ok=True)

    print("Collecting errors from MemGP simulations…")
    errors_by_n = collect_set_size_errors(config)
    np.savez_compressed(
        os.path.join(out_dir, "wm_gp_errors_by_set_size.npz"),
        **{f"N{n}": errors_by_n[n] for n in sorted(errors_by_n.keys())},
    )

    print("Fitting IL / SA / EP / VP (MLE) and mixture summaries…")
    de_max = args.de_maxiter if args.de_maxiter is not None else 60
    empirical, fits = fit_all_models(
        errors_by_n, set_sizes, n_synth_avg=10, seed=args.seed, de_maxiter=de_max
    )

    # Save tables
    rows = []
    for N in set_sizes:
        w, c, ws, cs = empirical[N]
        rows.append({"N": N, "w_emp": w, "CSD_emp": c, "w_sem": ws, "CSD_sem": cs})
    import pandas as pd

    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "mixture_summary_empirical.csv"), index=False)
    for name, fr in fits.items():
        print(f"  {name}: neg_loglik≈{fr.neg_ll:.1f}, params={np.round(fr.params, 4)}")

    ic = compute_information_criteria(fits, errors_by_n)
    ic_rows = []
    for name in ["IL", "SA", "EP", "VP"]:
        r = ic[name]
        ic_rows.append({
            "Model": name, "k": r["k"], "n": r["n"],
            "neg_loglik": r["neg_ll"],
            "AIC": r["AIC"], "BIC": r["BIC"], "AICc": r["AICc"],
            "delta_AIC": r["delta_AIC"], "delta_BIC": r["delta_BIC"],
        })
        print(f"  {name}: AIC={r['AIC']:.1f}, BIC={r['BIC']:.1f}, "
              f"ΔAIC={r['delta_AIC']:.1f}, ΔBIC={r['delta_BIC']:.1f}")
    pd.DataFrame(ic_rows).to_csv(
        os.path.join(out_dir, "model_comparison_information_criteria.csv"), index=False
    )

    plot_figure4a_fixed(
        set_sizes,
        empirical,
        fits,
        os.path.join(out_dir, "model_comparison_figure4a_style.png"),
        title_suffix="(PNAS-style)",
    )


if __name__ == "__main__":
    main()
