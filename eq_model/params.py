"""
Model parameters for the PJM-calibrated long-run equilibrium model.

Every number here is transcribed from Appendix "Model Parameterization"
(parameters.tex).  Anything that is *not* pinned down in the appendix is
exposed as an explicit, documented default on ``ModelParams`` so that it can
be overridden from the command line or a config file.  See README.md,
section "Open parameters / inconsistencies" for the list.

Units
-----
* capacities K_z            MW  (AC nameplate for solar; see ``solar_capacity_basis``)
* energy q, D               MWh per hour (== MW over one hour)
* prices p, VOLL, c_m       $/MWh
* investment cost I_z       $/MW-yr
* reliability cost coefs    $/MW-h (per MW of reserve requirement per hour)
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional
import math

import numpy as np


# ---------------------------------------------------------------------------
# Technology table (2025$), parameters.tex Table "Technology parameters"
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Tech:
    name: str
    kind: str                 # "thermal" | "vre" | "storage"
    occ_per_kw: float         # overnight capital cost, $/kW  (storage: per kW of power)
    fom_per_kw_yr: float      # fixed O&M, $/kW-yr
    life_yr: int              # technical life
    c_var: float = 0.0        # variable cost c_m, $/MWh (VOM + heat rate x fuel)
    e_rate: float = 0.0       # emissions rate e_m, tCO2/MWh
    duration_h: float = 0.0   # storage duration d_s (hours)
    vre_key: str = ""         # "solar" | "wind" for VRE (links to theta columns)

    def annualized_cost(self, r: float) -> float:
        """I_z = OCC_z * CRF(r, L_z) + FOM_z, in $/MW-yr (eq. annualization)."""
        return 1000.0 * (self.occ_per_kw * crf(r, self.life_yr) + self.fom_per_kw_yr)

    def annualized_capital(self, r: float) -> float:
        """Capital-recovery part of I_z: OCC_z * CRF(r, L_z) in $/MW-yr, excluding FOM.  The basis
        for capital-scaled risk aversion - financing costs apply to capital, not O&M."""
        return 1000.0 * self.occ_per_kw * crf(r, self.life_yr)


def crf(r: float, life: float) -> float:
    """Capital recovery factor CRF(r, L) = r(1+r)^L / ((1+r)^L - 1)."""
    if r == 0:
        return 1.0 / life
    return r * (1 + r) ** life / ((1 + r) ** life - 1)


# c_z = VOM + heat_rate * fuel  (values shown in the table notes; recomputed here so the
# provenance is explicit; the table's rounded values are reproduced to 2 decimals)
_NUC_C = 3.05 + 10.497 * 0.54      # 8.72
_CCGT_C = 2.22 + 6.196 * 4.68      # 31.22
_CT_C = 7.56 + 9.717 * 4.68        # 53.04
_COAL_C = 9.80 + 8.490 * 2.47      # 30.77
_LB_PER_T = 2204.62
_CCGT_E = 118.6 * 6.196 / _LB_PER_T   # 0.333
_CT_E = 119.0 * 9.717 / _LB_PER_T     # 0.524
_COAL_E = 202.3 * 8.490 / _LB_PER_T   # 0.779

DEFAULT_TECHS: List[Tech] = [
    Tech("nuclear", "thermal", 6268, 191, 60, c_var=_NUC_C, e_rate=0.0),
    Tech("ccgt",    "thermal", 1288,  34, 30, c_var=_CCGT_C, e_rate=_CCGT_E),
    Tech("ct",      "thermal", 1134,  27, 30, c_var=_CT_C,   e_rate=_CT_E),
    Tech("coal",    "thermal", 3337,  91, 30, c_var=_COAL_C, e_rate=_COAL_E),
    Tech("solar",   "vre",     1138,  20, 30, vre_key="solar"),
    Tech("wind",    "vre",     1339,  32, 30, vre_key="wind"),
    Tech("storage4", "storage", 1417, 35, 15, duration_h=4.0),
    Tech("storage8", "storage", 2488, 62, 15, duration_h=8.0),
]


# ---------------------------------------------------------------------------
# Regimes (empirical_model_setup.tex, "Regimes" table)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Regime:
    name: str
    description: str
    planner: bool             # True: cost-minimising planner LP; False: decentralised market
    reliability: str          # "internalized" | "socialized" | "priced"  (priced = via a_r markdown)
    contracts: bool           # forward contracts available to agents (E-SFPFC)
    carbon_priced: bool       # tau > 0 in agents'/planner's cost


REGIMES: Dict[str, Regime] = {
    "P1": Regime("P1", "Reliability benchmark planner (tau=0)", True, "internalized", False, False),
    "P2": Regime("P2", "First-best benchmark planner (tau=SCC)", True, "internalized", False, True),
    "R2": Regime("R2", "Unreformed merchant market (chi_z=0)", False, "socialized", False, False),
    "R3": Regime("R3", "E-SFPFC contracted market", False, "socialized", True, False),
    "R4": Regime("R4", "E-SFPFC-RA (reliability priced via a_r)", False, "priced", True, False),
    "R5": Regime("R5", "E-SFPFC-RA + carbon price", False, "priced", True, True),
    "R6": Regime("R6", "Carbon price only, merchant", False, "socialized", False, True),
}


# ---------------------------------------------------------------------------
# All scalar parameters
# ---------------------------------------------------------------------------

@dataclass
class ModelParams:
    # --- Additional parameters table -------------------------------------
    r: float = 0.025                  # real discount rate
    voll: float = 10_000.0            # $/MWh
    tau_scc: float = 280.0            # social cost of carbon, $/t (2025$)
    gamma: float = 0.0                # risk aversion (CALIBRATED - no value in appendix)
    # Per-technology risk aversion of the market agents.  "none": every agent uses gamma.
    # "capital": gamma_z = gamma * capital cost of z / capital cost of the reference technology,
    # so capital-heavy technologies (nuclear) are more risk averse - encoding that financing costs
    # weigh most on them.  gamma keeps its meaning for the reference technology.  The welfare
    # metric keeps the scalar gamma (society's risk aversion has no per-technology meaning).
    gamma_scaling: str = "none"               # "none" | "capital"
    gamma_reference_tech: str = "ccgt"

    # --- Storage ---------------------------------------------------------
    storage_eff_oneway: float = math.sqrt(0.85)   # epsilon_s = 0.922

    # --- Reliability-cost block (parameters.tex "Reliability costs") ------
    p_ref: float = 50.73              # reference LMP, $/MWh (PJM 2025 RT load-weighted)
    flex_price_share: float = 0.03    # flexible/ramping reserve price = 3% of p_ref  -> 1.52 $/MW-h
    reg_price_share: float = 0.006    # regulation price = 0.6% of p_ref             -> 0.304 $/MW-h
    flex_wind_coef: float = 0.189     # 0.10 * 1.89   (per MWh of available wind)
    flex_solar_coef: float = 0.076    # 0.04 * 1.89   (per MW of AC solar capacity, daylight hours)
    reg_load_coef: float = 0.01       # regulation: 1% of load (decision-independent constant)
    reg_wind_coef: float = 0.005
    reg_solar_coef: float = 0.003
    # The appendix divides K_solar by 1.34 (ReEDS's DC->AC convention).  If K_solar is
    # already AC nameplate (EIA-860M "Nameplate Capacity (MW)"), no division is needed.
    # See README "AC vs DC".  Set to "DC" to reproduce the appendix formula literally.
    solar_capacity_basis: str = "AC"
    dc_ac_ratio: float = 1.34
    # daylight indicator rule for the solar reserve terms
    daylight_rule: str = "solar_threshold"   # "solar_threshold" | "fixed_hours"
    daylight_theta_threshold: float = 0.01   # hour counts as daylight if theta_solar > this
    daylight_fixed_hours: tuple = (6, 19)    # local hours [start, end) for "fixed_hours"

    # --- Contract (E-SFPFC) block -----------------------------------------
    # Fixed at 70% of the observed PJM RTO peak load in the supplied load CSVs:
    # 0.70 * 160154.429 MW = 112108.1003 MW (2025-06-23 21:00 UTC).
    q_bar_rule: str = "fixed"                # "mean_profile_peak" | "peak_load" | "mean_load" | "fixed"
    q_bar_coverage: float = 1.0              # multiplier on the rule's reference quantity
    q_bar_mw: Optional[float] = 112108.1003  # total contracted capacity Q_bar (MW)
    # a_r markdown ($/MWh of contracted energy). "regulation" calculates the value from
    # the active coefficients; "appendix" uses the printed values (0.0015, 0.0018) as a
    # legacy option; "full" also includes the flexible-reserve (psi_E) term. See README.
    markdown_mode: str = "regulation"        # "appendix" | "regulation" | "full"
    a_wind_appendix: float = 0.0015
    a_solar_appendix: float = 0.0018
    markdown_solar_cf: float = 0.25          # used by "regulation"/"full" to convert $/MW-yr -> $/MWh
    markdown_daylight_share: float = 0.5

    # An hour counts as curtailed for VRE r when available output theta_rh * K_r exceeds
    # dispatched output by more than this many MW (an absolute floor, so that LP round-off
    # in the many hours where VRE is fully absorbed does not register as curtailment).
    curtailment_hour_threshold_mw: float = 1.0

    # --- Welfare evaluation ----------------------------------------------
    # Carbon valued at the SCC in the welfare metric for ALL regimes (so that P2 is the
    # first-best benchmark).  Set False to value emissions only where tau is priced.
    welfare_tau_always_scc: bool = True

    # --- Technologies --------------------------------------------------------
    techs: List[Tech] = field(default_factory=lambda: list(DEFAULT_TECHS))

    # ------------------------------------------------------------------ derived
    @property
    def tech_names(self) -> List[str]:
        return [t.name for t in self.techs]

    def tech(self, name: str) -> Tech:
        for t in self.techs:
            if t.name == name:
                return t
        raise KeyError(name)

    @property
    def thermal(self) -> List[Tech]:
        return [t for t in self.techs if t.kind == "thermal"]

    @property
    def vre(self) -> List[Tech]:
        return [t for t in self.techs if t.kind == "vre"]

    @property
    def storage(self) -> List[Tech]:
        return [t for t in self.techs if t.kind == "storage"]

    def investment_cost(self, hours: int = 8760) -> Dict[str, float]:
        """I_z in $/MW-yr for every technology.  ``hours`` < 8760 prorates the annual cost
        (only for sub-year smoke tests; the model is defined on 8760-hour years)."""
        s = hours / 8760.0
        return {t.name: s * t.annualized_cost(self.r) for t in self.techs}

    def marginal_cost(self, tau: float) -> Dict[str, float]:
        """c_m + e_m * tau for thermal units ($/MWh); 0 for VRE/storage."""
        return {t.name: (t.c_var + t.e_rate * tau if t.kind == "thermal" else 0.0) for t in self.techs}

    # reliability-cost prices ($/MW-h of reserve)
    @property
    def flex_price(self) -> float:
        return self.flex_price_share * self.p_ref     # 1.5219 (appendix rounds to 1.52)

    @property
    def reg_price(self) -> float:
        return self.reg_price_share * self.p_ref      # 0.30438

    @property
    def solar_divisor(self) -> float:
        return self.dc_ac_ratio if self.solar_capacity_basis.upper() == "DC" else 1.0

    def psi_wind_per_mwh_available(self) -> float:
        """Reliability cost per MWh of *available* wind energy (theta*K), $/MWh.
        = flex_price*0.189 + reg_price*0.005."""
        return self.flex_price * self.flex_wind_coef + self.reg_price * self.reg_wind_coef

    def psi_solar_per_mw_daylight_hour(self) -> float:
        """Reliability cost per MW of solar capacity per daylight hour, $/MW-h.
        = (flex_price*0.076 + reg_price*0.003) / divisor."""
        return (self.flex_price * self.flex_solar_coef + self.reg_price * self.reg_solar_coef) / self.solar_divisor

    def psi_load_per_mwh(self) -> float:
        """Regulation cost attributable to load, $/MWh of load (decision independent)."""
        return self.reg_price * self.reg_load_coef

    def markdowns(self) -> Dict[str, float]:
        """a_r in $/MWh of contracted energy, for each VRE technology."""
        if self.markdown_mode == "appendix":
            return {"solar": self.a_solar_appendix, "wind": self.a_wind_appendix}
        # regulation-only, recomputed from the coefficients (reproduces the appendix numbers
        # when solar_capacity_basis == "AC")
        a_wind = self.reg_price * self.reg_wind_coef
        hours = 8760.0
        a_solar = (self.reg_price * self.reg_solar_coef / self.solar_divisor
                   * self.markdown_daylight_share * hours) / (self.markdown_solar_cf * hours)
        if self.markdown_mode == "regulation":
            return {"solar": a_solar, "wind": a_wind}
        if self.markdown_mode == "full":
            a_wind += self.flex_price * self.flex_wind_coef
            a_solar += (self.flex_price * self.flex_solar_coef / self.solar_divisor
                        * self.markdown_daylight_share * hours) / (self.markdown_solar_cf * hours)
            return {"solar": a_solar, "wind": a_wind}
        raise ValueError(f"unknown markdown_mode {self.markdown_mode!r}")

    def gamma_by_tech(self, gamma: Optional[float] = None) -> Dict[str, float]:
        """gamma_z used by each technology's agents (see ``gamma_scaling``)."""
        g = self.gamma if gamma is None else gamma
        if self.gamma_scaling == "none":
            return {t.name: g for t in self.techs}
        if self.gamma_scaling == "capital":
            ref = self.tech(self.gamma_reference_tech).annualized_capital(self.r)
            return {t.name: g * t.annualized_capital(self.r) / ref for t in self.techs}
        raise ValueError(f"unknown gamma_scaling {self.gamma_scaling!r}")

    def with_(self, **kw) -> "ModelParams":
        return replace(self, **kw)

    def summary(self) -> str:
        I = self.investment_cost()
        mc0 = self.marginal_cost(0.0)
        mc1 = self.marginal_cost(self.tau_scc)
        lines = [f"r={self.r}  VOLL={self.voll}  tau_SCC={self.tau_scc}  gamma={self.gamma}",
                 f"{'tech':10s} {'I ($/MW-yr)':>14s} {'c (tau=0)':>10s} {'c (tau=SCC)':>12s} {'e (t/MWh)':>10s}"]
        for t in self.techs:
            lines.append(f"{t.name:10s} {I[t.name]:14,.0f} {mc0[t.name]:10.2f} {mc1[t.name]:12.2f} {t.e_rate:10.3f}")
        lines.append(f"psi_wind = {self.psi_wind_per_mwh_available():.4f} $/MWh available wind; "
                     f"psi_solar = {self.psi_solar_per_mw_daylight_hour():.4f} $/MW-daylight-h "
                     f"(solar basis={self.solar_capacity_basis}, divisor={self.solar_divisor})")
        lines.append(f"markdowns a_r ({self.markdown_mode}): {self.markdowns()}")
        if self.gamma_scaling != "none":
            gz = self.gamma_by_tech()
            lines.append(f"gamma_scaling={self.gamma_scaling} (ref {self.gamma_reference_tech}): "
                         + ", ".join(f"{z}={v:.3g}" for z, v in gz.items()))
        return "\n".join(lines)


def implied_discount_rate(tech: Tech, target_I: float, lo: float = -0.5, hi: float = 2.0) -> float:
    """Solve OCC*CRF(r',L)+FOM = target_I for r' (bisection).  Used to translate an
    equilibrium risk premium (I_eff - I) into an equivalent cost-of-capital spread,
    the natural calibration target for gamma ("gamma and its financing-spread target")."""
    f = lambda rr: tech.annualized_cost(rr) - target_I
    if f(lo) > 0 or f(hi) < 0:
        return float("nan")
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if f(mid) > 0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


if __name__ == "__main__":
    print(ModelParams().summary())
