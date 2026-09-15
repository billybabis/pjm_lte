"""
Agent-side economics for the decentralised market (empirical_model_setup.tex,
"Decentralized market").

Linearisation of the contract term
----------------------------------
The profit term (p_hat - p_hy) * lambda_h * chi_z * K_z is bilinear in (chi_z, K_z).
Defining the *contracted capacity* Q_z := chi_z K_z (0 <= Q_z <= K_z) makes every
agent's problem an LP in (K_z, Q_z, q...) for given prices, and the forward clearing
sum_z chi_z K_z = Q_bar becomes sum_z Q_z = Q_bar.  chi_z = Q_z / K_z is recovered
afterwards.  This is the "chi tweak" that keeps the model linear.

Because each agent's problem is positively homogeneous of degree one in (K_z, Q_z),
its optimal value is 0 (K_z = 0) or +inf, and a competitive equilibrium requires the
*per-MW risk-adjusted profit* to be exactly zero for every active technology and
non-positive for inactive ones:

    rho_z^*(K, p_hat) := max_{chi in [0,1]} [ mean_y pi_zy(chi) + gamma * min_y pi_zy(chi) ] = 0,
    pi_zy(chi) = m_zy(K) - I_z + s_zy(p_hat) * chi ,
    s_zy      = Lambda * (p_hat - a_z) - Pbar_y ,   Lambda = sum_h lambda_h ,  Pbar_y = sum_h lambda_h p_hy ,

where m_zy(K) is the spot margin per MW of z in year y at the competitive dispatch
given K, and a_z is the E-SFPFC-RA markdown ($/MWh, VRE only in R4/R5).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from .lp import StorageArbitrageLP
from .panel import HourlyPanel
from .params import ModelParams

log = logging.getLogger("eq_model")


# ---------------------------------------------------------------------------
# risk measure  rho(Pi) = mean_y Pi_y + gamma * min_y Pi_y      (eq. rho)
# ---------------------------------------------------------------------------

def rho(pi: np.ndarray, gamma: float) -> float:
    return float(np.mean(pi) + gamma * np.min(pi))


def risk_weights(pi: np.ndarray, gamma: float) -> np.ndarray:
    """A subgradient of rho at pi: 1/Y on every year plus gamma on (one of) the worst
    year(s) -> the agent's risk-adjusted probability measure (sums to 1 + gamma)."""
    Y = len(pi)
    w = np.full(Y, 1.0 / Y)
    worst = np.flatnonzero(np.isclose(pi, pi.min(), rtol=0, atol=1e-9 * (1 + abs(pi.min()))))
    w[worst] += gamma / len(worst)
    return w


# ---------------------------------------------------------------------------
# contract choice: maximise concave piecewise-linear  g(chi) = rho(pi0 + s chi) on [0,1]
# ---------------------------------------------------------------------------

def best_chi(pi0: np.ndarray, s: np.ndarray, gamma: float, tol: float = 1e-9) -> Tuple[float, float, float]:
    """Return (chi_lo, chi_hi, value): the interval of maximisers of
    g(chi) = mean(pi0 + s*chi) + gamma*min(pi0 + s*chi) on [0, 1] and the maximum.
    g is concave and piecewise linear, so it suffices to evaluate the end points and every
    crossing chi where two of the lines pi0_y + s_y chi intersect."""
    cands = [0.0, 1.0]
    Y = len(pi0)
    for i in range(Y):
        for j in range(i + 1, Y):
            ds = s[i] - s[j]
            if abs(ds) > 1e-12:
                x = (pi0[j] - pi0[i]) / ds
                if 0.0 < x < 1.0:
                    cands.append(float(x))
    cands = np.unique(np.array(cands))
    vals = np.array([rho(pi0 + s * x, gamma) for x in cands])
    best = vals.max()
    scale = 1.0 + abs(best)
    ok = cands[vals >= best - tol * scale]
    return float(ok.min()), float(ok.max()), float(best)


# ---------------------------------------------------------------------------
# spot margins per MW from a price path
# ---------------------------------------------------------------------------

class MarginEvaluator:
    """m_zy(p): per-MW spot profit of every technology in every year at prices p (Y,H).

    thermal: sum_h (p_hy - c_m - e_m tau)^+ ; VRE: sum_h theta_rhy p_hy^+ ;
    storage: arbitrage LP with K_s = 1 (exact, also when K_s = 0 in the dispatch)."""

    def __init__(self, panel: HourlyPanel, params: ModelParams, tau: float):
        self.panel, self.params, self.tau = panel, params, tau
        self.mc = params.marginal_cost(tau)
        self._arb = {t.name: StorageArbitrageLP(panel.H, t.duration_h, params.storage_eff_oneway)
                     for t in params.storage}

    def margins(self, price: np.ndarray) -> Dict[str, np.ndarray]:
        Y = price.shape[0]
        out = {}
        pos = np.maximum(price, 0.0)
        for t in self.params.techs:
            if t.kind == "thermal":
                out[t.name] = np.maximum(price - self.mc[t.name], 0.0).sum(axis=1)
            elif t.kind == "vre":
                out[t.name] = (self.panel.theta[t.vre_key] * pos).sum(axis=1)
            else:
                out[t.name] = np.array([self._arb[t.name].value(price[y]) for y in range(Y)])
        return out


# ---------------------------------------------------------------------------
# forward market
# ---------------------------------------------------------------------------

@dataclass
class ForwardResult:
    p_hat: float                       # $/MWh
    chi: Dict[str, float]              # contract ratio chi_z in [0,1]
    Q: Dict[str, float]                # contracted MW  Q_z = chi_z K_z
    chi_interval: Dict[str, Tuple[float, float]]
    rho_star: Dict[str, float]         # per-MW risk-adjusted profit at the optimal chi, $/MW-yr
    supply: float                      # sum_z Q_z
    q_bar: float
    rationed: bool = False
    notes: List[str] = field(default_factory=list)


def _s_vec(p_hat: float, Lambda: float, Pbar: np.ndarray, a: float) -> np.ndarray:
    return Lambda * (p_hat - a) - Pbar


def forward_market(m: Dict[str, np.ndarray], I: Dict[str, float], K: Dict[str, float], Pbar: np.ndarray,
                   Lambda: float, gamma: Union[float, Dict[str, float]], q_bar: float, markdown: Dict[str, float],
                   contracts: bool, tol: float = 1e-7) -> ForwardResult:
    """Clear the forward market (eq. forward-clearing) given spot outcomes.

    m[z] : (Y,) spot margin per MW; I[z] : $/MW-yr; K[z] : MW; Pbar : (Y,) sum_h lambda_h p_hy;
    markdown[z] : a_z ($/MWh; 0 unless priced).  With ``contracts=False`` every chi_z = 0
    (merchant regimes R2, R6) and p_hat is reported as NaN.
    """
    names = list(m.keys())
    # gamma may be one value for every agent or {z: gamma_z} (params.gamma_scaling); each agent's
    # problem is separate, so a per-technology gamma needs nothing beyond using its own value.
    g = gamma if isinstance(gamma, dict) else {z: gamma for z in names}
    if not contracts or q_bar <= 0:
        rs = {z: rho(m[z] - I[z], g[z]) for z in names}
        return ForwardResult(p_hat=float("nan"), chi={z: 0.0 for z in names}, Q={z: 0.0 for z in names},
                             chi_interval={z: (0.0, 0.0) for z in names}, rho_star=rs, supply=0.0, q_bar=0.0)

    def intervals(p_hat: float):
        res = {}
        for z in names:
            s = _s_vec(p_hat, Lambda, Pbar, markdown.get(z, 0.0))
            res[z] = best_chi(m[z] - I[z], s, g[z])
        return res

    def supply_bounds(p_hat: float):
        iv = intervals(p_hat)
        lo = sum(iv[z][0] * K[z] for z in names)
        hi = sum(iv[z][1] * K[z] for z in names)
        return lo, hi, iv

    total_K = sum(K.values())
    notes = []
    if total_K <= 0:
        return ForwardResult(float("nan"), {z: 0.0 for z in names}, {z: 0.0 for z in names},
                             {z: (0.0, 0.0) for z in names}, {z: rho(m[z] - I[z], g[z]) for z in names},
                             0.0, q_bar, rationed=True, notes=["no capacity"])
    a_max = max(markdown.get(z, 0.0) for z in names)
    p_lo = float(Pbar.min() / Lambda) - 1.0
    p_hi = float(Pbar.max() / Lambda) + a_max + 1.0
    lo_s, hi_s, _ = supply_bounds(p_hi)
    if hi_s < q_bar - 1e-9:
        # even at chi=1 for everyone there is not enough capacity: rationed
        iv = intervals(p_hi)
        chi = {z: 1.0 for z in names}
        rs = {z: iv[z][2] for z in names}
        notes.append(f"forward market rationed: sum K = {total_K:.0f} < Q_bar = {q_bar:.0f}")
        return ForwardResult(p_hi, chi, {z: K[z] for z in names}, {z: (1.0, 1.0) for z in names}, rs,
                             total_K, q_bar, rationed=True, notes=notes)
    # bisection on p_hat: the supply interval [lo, hi] is monotone non-decreasing in p_hat.
    # Invariant: supply(p_lo) < q_bar <= supply(p_hi) in the set-valued sense.
    p_hat = None
    for _ in range(300):
        mid = 0.5 * (p_lo + p_hi)
        lo_s, hi_s, iv = supply_bounds(mid)
        if lo_s <= q_bar <= hi_s:
            p_hat = mid
            break
        if lo_s > q_bar:
            p_hi = mid          # too much supply -> lower price
        else:
            p_lo = mid          # too little supply -> raise price
        if p_hi - p_lo < tol * (1.0 + abs(mid)):
            p_hat = p_hi        # sup of prices with supply < q_bar: an indifference point up to tol
            break
    if p_hat is None:
        p_hat = 0.5 * (p_lo + p_hi)
    # At a price where some agent's optimal chi jumps, the exact indifference point is never
    # hit by the midpoints; the equilibrium chi of such an agent is any point of the union of
    # its optimal sets over the (collapsed) bracket.  Take [lo at p_lo, hi at p_hi].
    lo_s, hi_s, iv = supply_bounds(p_hat)
    if not (lo_s <= q_bar <= hi_s):
        iv_lo, iv_hi = intervals(p_lo), intervals(p_hi)
        iv = {z: (min(iv_lo[z][0], iv[z][0]), max(iv_hi[z][1], iv[z][1]), iv[z][2]) for z in names}
        lo_s = sum(iv[z][0] * K[z] for z in names)
        hi_s = sum(iv[z][1] * K[z] for z in names)
    if not (lo_s - 1e-6 * q_bar <= q_bar <= hi_s + 1e-6 * q_bar):
        notes.append(f"forward clearing tolerance: supply in [{lo_s:.1f},{hi_s:.1f}] vs Q_bar {q_bar:.1f}")
    # allocation: strict agents at their unique chi; indifferent agents share the residual pro rata
    chi = {z: iv[z][0] for z in names}
    residual = q_bar - lo_s
    slack = {z: (iv[z][1] - iv[z][0]) * K[z] for z in names}
    tot_slack = sum(slack.values())
    if tot_slack > 0 and residual > 0:
        frac = min(1.0, residual / tot_slack)
        for z in names:
            if K[z] > 0:
                chi[z] = iv[z][0] + frac * (iv[z][1] - iv[z][0])
    Q = {z: chi[z] * K[z] for z in names}
    return ForwardResult(p_hat=p_hat, chi=chi, Q=Q, chi_interval={z: (iv[z][0], iv[z][1]) for z in names},
                         rho_star={z: iv[z][2] for z in names}, supply=sum(Q.values()), q_bar=q_bar, notes=notes)


def agent_profit_paths(m: Dict[str, np.ndarray], I: Dict[str, float], K: Dict[str, float], chi: Dict[str, float],
                       p_hat: float, Pbar: np.ndarray, Lambda: float, markdown: Dict[str, float]) -> Dict[str, np.ndarray]:
    """Realised annual profit Pi_zy (total $, not per MW) for reporting."""
    out = {}
    for z in m:
        s = _s_vec(p_hat, Lambda, Pbar, markdown.get(z, 0.0)) if np.isfinite(p_hat) else np.zeros_like(Pbar)
        out[z] = (m[z] - I[z] + s * chi[z]) * K[z]
    return out
