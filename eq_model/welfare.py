"""
Welfare and summary statistics for a solved regime (eq. cost / eq. welfare):

    C_y = sum_h [ sum_m (c_m + e_m tau_w) q_mhy + VOLL l_hy ] + sum_z I_z K_z + sum_r Psi_ry
    C   = (1/Y) sum_y C_y + gamma * max_y C_y

``tau_w`` is the SCC for every regime when ``params.welfare_tau_always_scc`` (default), so
that unpriced emissions in tau=0 regimes still count as a social cost; otherwise it is the
regime's own tau.  The load-driven regulation cost (0.01 * D) is reported separately and is
identical across regimes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .equilibrium import RegimeResult
from .lp import psi_by_year
from .panel import HourlyPanel
from .params import ModelParams


@dataclass
class Welfare:
    regime: str
    gamma: float
    tau_market: float
    tau_welfare: float
    C_y: np.ndarray                   # (Y,)
    C_mean: float
    C_max: float
    C_risk_adjusted: float            # mean + gamma * max
    components_mean: Dict[str, float] # variable, carbon, voll, capex, psi (annual means, $)
    emissions_y: np.ndarray           # (Y,) tCO2
    lost_load_mwh_y: np.ndarray       # (Y,)
    lost_load_hours_y: np.ndarray     # (Y,) hours with l > 0
    psi_load_mean: float              # decision-independent regulation cost (not in C)
    energy_share: Dict[str, float]    # z -> share of generation (storage net counts negative)
    curtailment_share: Dict[str, float]  # r -> curtailed / available (energy)
    curtailment_mwh_y: Dict[str, np.ndarray]     # r -> (Y,) curtailed energy
    curtailment_hours_y: Dict[str, np.ndarray]   # r -> (Y,) hours with curtailment > threshold
    curtailment_hours_any_y: np.ndarray          # (Y,) hours with ANY VRE curtailed
    curtailment_share_vre: float                 # curtailed / available, VRE technologies pooled
    price_mean: float                 # simple average $/MWh
    price_load_weighted: float
    price_max: float
    hours_at_voll: int
    capacity: Dict[str, float]


def evaluate_welfare(panel: HourlyPanel, params: ModelParams, res: RegimeResult) -> Welfare:
    ev = res.evaluation
    Y, H = panel.Y, panel.H
    tau_w = params.tau_scc if params.welfare_tau_always_scc else res.tau
    mc_w = params.marginal_cost(tau_w)
    mc_var = params.marginal_cost(0.0)
    I = res.I
    K = res.K
    var = np.zeros(Y); carbon = np.zeros(Y); emis = np.zeros(Y)
    for t in params.thermal:
        q = np.stack([d.q[t.name] for d in ev.dispatch])
        e_y = q.sum(axis=1)
        var += mc_var[t.name] * e_y
        emis += t.e_rate * e_y
        carbon += t.e_rate * tau_w * e_y
    ll = np.stack([d.lost_load for d in ev.dispatch])
    voll_cost = params.voll * ll.sum(axis=1)
    capex = sum(I[z] * K[z] for z in K)
    psi = psi_by_year(panel, params, K)
    psi_y = sum(psi.values()) if psi else np.zeros(Y)
    C_y = var + carbon + voll_cost + capex + psi_y
    gamma = res.gamma
    comps = {"variable_om_fuel": float(var.mean()), "carbon_at_tau_welfare": float(carbon.mean()),
             "lost_load": float(voll_cost.mean()), "capex_annualized": float(capex), "reliability_psi": float(psi_y.mean())}
    psi_load = float((panel.D.sum(axis=1) * params.psi_load_per_mwh()).mean())
    # energy shares
    gen = {}
    for t in params.techs:
        gen[t.name] = float(np.stack([d.q[t.name] for d in ev.dispatch]).sum())
    total_load = float(panel.D.sum())
    shares = {z: v / total_load for z, v in gen.items()}
    shares["lost_load"] = float(ll.sum() / total_load)
    # curtailment: available theta*K that was not dispatched, by energy and by hour count
    curt, curt_mwh, curt_hrs = {}, {}, {}
    thr = params.curtailment_hour_threshold_mw
    any_curt = np.zeros((Y, H), dtype=bool)
    vre_avail = vre_spill = 0.0
    for t in params.vre:
        avail_hy = panel.theta[t.vre_key] * K[t.name]
        spill_hy = np.maximum(avail_hy - np.stack([d.q[t.name] for d in ev.dispatch]), 0.0)
        avail = float(avail_hy.sum())
        curt[t.name] = float(1 - gen[t.name] / avail) if avail > 0 else float("nan")
        curt_mwh[t.name] = spill_hy.sum(axis=1)
        curt_hrs[t.name] = (spill_hy > thr).sum(axis=1)
        any_curt |= spill_hy > thr
        vre_avail += avail
        vre_spill += float(spill_hy.sum())
    p = ev.price
    return Welfare(regime=res.regime.name, gamma=gamma, tau_market=res.tau, tau_welfare=tau_w, C_y=C_y,
                   C_mean=float(C_y.mean()), C_max=float(C_y.max()), C_risk_adjusted=float(C_y.mean() + gamma * C_y.max()),
                   components_mean=comps, emissions_y=emis, lost_load_mwh_y=ll.sum(axis=1),
                   lost_load_hours_y=(ll > 1e-6).sum(axis=1), psi_load_mean=psi_load, energy_share=shares,
                   curtailment_share=curt, curtailment_mwh_y=curt_mwh, curtailment_hours_y=curt_hrs,
                   curtailment_hours_any_y=any_curt.sum(axis=1),
                   curtailment_share_vre=(vre_spill / vre_avail) if vre_avail > 0 else float("nan"),
                   price_mean=float(p.mean()), price_load_weighted=float((p * panel.D).sum() / panel.D.sum()),
                   price_max=float(p.max()), hours_at_voll=int((p >= params.voll - 1e-6).sum()), capacity=dict(K))


def summary_row(w: Welfare, res: RegimeResult, params: ModelParams) -> dict:
    row = {"regime": w.regime, "gamma": w.gamma, "tau_market": w.tau_market, "converged": res.converged,
           "C_mean_$bn": w.C_mean / 1e9, "C_max_$bn": w.C_max / 1e9, "C_risk_adj_$bn": w.C_risk_adjusted / 1e9,
           "emissions_Mt_mean": float(w.emissions_y.mean()) / 1e6,
           "lost_load_GWh_mean": float(w.lost_load_mwh_y.mean()) / 1e3,
           "lost_load_hours_mean": float(w.lost_load_hours_y.mean()),
           "price_mean": w.price_mean, "price_load_wtd": w.price_load_weighted, "hours_at_VOLL": w.hours_at_voll,
           "p_hat": res.forward.p_hat if res.forward else float("nan"),
           "Q_bar_MW": res.q_bar, "runtime_s": res.runtime}
    for z in params.tech_names:
        row[f"K_{z}_MW"] = res.K[z]
    for z in params.tech_names:
        row[f"share_{z}"] = w.energy_share[z]
    row["share_lost_load"] = w.energy_share.get("lost_load", 0.0)
    row["curt_hours_any_mean"] = float(w.curtailment_hours_any_y.mean())
    row["curt_share_vre"] = w.curtailment_share_vre
    for t in params.vre:
        row[f"curt_hours_{t.name}_mean"] = float(w.curtailment_hours_y[t.name].mean())
        row[f"curt_share_{t.name}"] = w.curtailment_share[t.name]
        row[f"curt_GWh_{t.name}_mean"] = float(w.curtailment_mwh_y[t.name].mean()) / 1e3
    for z in params.tech_names:
        row[f"premium_pct_{z}"] = 100 * res.risk_premium[z] / res.I[z] if res.K[z] > 1.0 else float("nan")
    for z in params.tech_names:
        row[f"chi_{z}"] = res.forward.chi[z] if res.forward else 0.0
    if params.gamma_scaling != "none":
        # only when active, so default-scenario rows keep exactly their existing keys
        for z in params.tech_names:
            row[f"gamma_{z}"] = res.gamma_by_tech.get(z, float("nan"))
    for k, v in w.components_mean.items():
        row[f"cost_{k}_$bn"] = v / 1e9
    return row


def profit_table(res: RegimeResult, panel: HourlyPanel) -> pd.DataFrame:
    """Per-MW annual profit by technology and year (spot margin - I + contract payoff)."""
    from .agents import _s_vec
    rows = []
    for z in res.K:
        a_z = res.markdown.get(z, 0.0)
        s = (_s_vec(res.forward.p_hat, res.Lambda, res.Pbar, a_z) if (res.forward and np.isfinite(res.forward.p_hat)) else np.zeros(panel.Y))
        chi = res.forward.chi[z] if res.forward else 0.0
        for yi, y in enumerate(panel.years):
            rows.append({"tech": z, "year": int(y), "spot_margin": res.margins[z][yi], "I": res.I[z],
                         "contract_payoff_per_MW": s[yi] * chi, "profit_per_MW": res.margins[z][yi] - res.I[z] + s[yi] * chi,
                         "K_MW": res.K[z]})
    return pd.DataFrame(rows)
