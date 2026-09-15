"""
pytest suite.  Runs in ~1-2 minutes on small synthetic panels.

    cd pjm_lre && python -m pytest -q tests
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eq_model.io_utils import DataError, parse_number
from eq_model.params import ModelParams, REGIMES, crf, implied_discount_rate
from eq_model.synthetic import make_synthetic_panel, write_synthetic_raw
from eq_model.lp import SystemLP, StorageArbitrageLP
from eq_model.dispatch import DispatchLP
from eq_model.agents import best_chi, rho, forward_market, MarginEvaluator
from eq_model.equilibrium import CapacityProblem, solve_regime
from eq_model.welfare import evaluate_welfare


# ----------------------------------------------------------------------------- data layer
def test_parse_number_handles_thousands_separators_and_refuses_silent_zeros():
    s = pd.Series(["1,281.0", " 12 ", "(3.5)", "", "NA", "0.25"])
    out = parse_number(s, "x")
    assert out.tolist()[:3] == [1281.0, 12.0, -3.5]
    assert np.isnan(out.iloc[3]) and np.isnan(out.iloc[4]) and out.iloc[5] == 0.25
    with pytest.raises(DataError):
        parse_number(pd.Series(["1,281.0", "abc"]), "y")          # never silently NaN/0
    assert np.isnan(parse_number(pd.Series(["abc"]), "z", strict=False).iloc[0])


@pytest.fixture(scope="module")
def raw(tmp_path_factory):
    d = tmp_path_factory.mktemp("raw")
    dirs = write_synthetic_raw(str(d), years=(2019, 2020), seed=3)     # includes leap year 2020
    return dirs


def test_area_resolution_uses_rto_row_and_cross_checks(raw):
    from eq_model.pjm_data import load_pjm_load
    s, res = load_pjm_load(raw["load"])
    assert res.used == "rto_row"
    assert res.rto_labels == ["RTO"] and set(res.region_labels) == {"MIDATL", "SOUTH", "WEST"}
    assert len(res.zonal_labels) == 20
    assert res.rel_diff_max < 0.01                       # zonal sum ~ RTO row
    assert len(s) == 24 * (365 + 366)                    # every hour of 2019-2020 (local years) present once
    s2, res2 = load_pjm_load(raw["load"], area_mode="zonal_sum")
    assert res2.used == "zonal_sum"
    # summing every row (the trap) would roughly triple the load
    df = pd.read_csv(os.path.join(raw["load"], sorted(os.listdir(raw["load"]))[0]))
    naive = df.groupby("datetime_beginning_utc")["mw"].sum().mean()
    assert 2.8 < naive / s.mean() < 3.2


def test_eia_capacity_and_coverage_match_truth(raw):
    from eq_model.eia import read_eia_generators, monthly_vre_capacity
    act = read_eia_generators(os.path.join(raw["capacity"], "generators_active_07_2026.csv"), "active")
    ret = read_eia_generators(os.path.join(raw["capacity"], "generators_retired_07_2026.csv"), "retired")
    assert (act["cap_ac_mw"] > 1000).any()               # the "1,970.0" rows parsed as numbers, not NaN
    cap = monthly_vre_capacity(act, ret, start="2019-01", end="2020-12")
    truth = pd.read_csv(os.path.join(raw["capacity"], "_synthetic_truth.csv"), parse_dates=["month"])
    m = cap.merge(truth, on="month", suffixes=("", "_t"))
    assert np.allclose(m.solar_mw, m.solar_mw_t, rtol=1e-3)
    assert np.allclose(m.solar_coverage, m.solar_coverage_t, atol=2e-3)
    assert (m.solar_mw_dc / m.solar_mw).between(1.29, 1.31).all()     # AC used, DC only reported


def test_panel_has_8760_hours_per_local_year_and_valid_shapes(raw):
    from eq_model.pjm_data import load_pjm_load, load_gen_by_fuel
    from eq_model.eia import read_eia_generators, monthly_vre_capacity
    from eq_model.panel import build_panel
    load, _ = load_pjm_load(raw["load"])
    gen = load_gen_by_fuel(raw["gen_by_fuel"])
    act = read_eia_generators(os.path.join(raw["capacity"], "generators_active_07_2026.csv"), "active")
    cap = monthly_vre_capacity(act, None, start="2019-01", end="2020-12")
    panel = build_panel(load, gen, cap, ModelParams())
    assert panel.D.shape == (2, 8760) and list(panel.years) == [2019, 2020]
    assert panel.lam.shape == (24,) and panel.lam.max() == 1.0 and panel.lam.min() > 0.5
    for th in panel.theta.values():
        assert th.min() >= 0 and th.max() <= 1
    assert 0.3 < panel.daylight.mean() < 0.6
    assert panel.hour_of_day[0, 0] == 0                  # local midnight starts the year


# ----------------------------------------------------------------------------- parameters
def test_parameter_table_matches_appendix():
    P = ModelParams()
    I = P.investment_cost()
    assert abs(crf(0.025, 30) - 0.04778) < 1e-4
    assert abs(I["ccgt"] - 95_538) < 5 and abs(I["nuclear"] - 393_791) < 5
    mc = P.marginal_cost(0.0)
    assert abs(mc["ccgt"] - 31.22) < 0.01 and abs(mc["ct"] - 53.04) < 0.01 and abs(mc["coal"] - 30.77) < 0.01
    assert abs(P.tech("ccgt").e_rate - 0.333) < 1e-3 and abs(P.tech("coal").e_rate - 0.779) < 1e-3
    assert abs(P.flex_price - 1.52) < 0.01 and abs(P.reg_price - 0.304) < 0.001
    a = P.markdowns()
    assert P.markdown_mode == "regulation"
    assert abs(a["wind"] - P.reg_price * P.reg_wind_coef) < 1e-12
    assert abs(a["solar"] - (P.reg_price * P.reg_solar_coef * P.markdown_daylight_share
                              / P.markdown_solar_cf)) < 1e-12
    r = implied_discount_rate(P.tech("ct"), I["ct"] * 1.10)
    assert 0.025 < r < 0.05


# ----------------------------------------------------------------------------- LP layer
@pytest.fixture(scope="module")
def small_panel():
    return make_synthetic_panel(years=(2017, 2018), H=24 * 21, seed=0)


def test_balance_dual_is_price(small_panel):
    from copy import deepcopy
    P = ModelParams()
    sol = SystemLP(small_panel, P, tau=P.tau_scc, include_psi=True).solve()
    y, h = 1, int(np.argmax((sol.price[1] > 50) & (sol.price[1] < P.voll - 1)))
    pan2 = deepcopy(small_panel); pan2.D[y, h] += 10.0
    sol2 = SystemLP(pan2, P, tau=P.tau_scc, include_psi=True).solve()
    assert abs((sol2.obj - sol.obj) - sol.price[y, h] * 10.0 / small_panel.Y) < 1e-3 * abs(sol.price[y, h] * 10.0 / small_panel.Y) + 1e-6


def test_margins_from_prices_equal_lp_duals_for_active_techs(small_panel):
    P = ModelParams()
    sol = SystemLP(small_panel, P, tau=P.tau_scc, include_psi=True).solve()
    me = MarginEvaluator(small_panel, P, P.tau_scc)
    m = me.margins(sol.price)
    for z, K in sol.K.items():
        if K > 1.0:
            assert np.allclose(m[z], sol.cap_margin[z], rtol=1e-6, atol=1e-3), z
            assert abs(m[z].mean() - sol.cost_K_used[z]) < 1e-6 * sol.cost_K_used[z]      # zero profit
        else:
            assert m[z].mean() <= sol.cost_K_used[z] * (1 + 1e-9)                          # no entry


def test_dispatch_lp_reduced_costs_match_closed_form(small_panel):
    P = ModelParams()
    d = DispatchLP(small_panel, 0, P, tau=0.0)
    K = {"nuclear": 0, "ccgt": 70000, "ct": 20000, "coal": 0, "solar": 5000, "wind": 30000, "storage4": 3000, "storage8": 0}
    r = d.solve(K)
    mc = P.marginal_cost(0.0)
    assert abs(r.margin["ct"] - np.maximum(r.price - mc["ct"], 0).sum()) < 1e-6 * max(1, r.margin["ct"])
    assert abs(r.margin["wind"] - (small_panel.theta["wind"][0] * np.maximum(r.price, 0)).sum()) < 1e-6 * max(1, r.margin["wind"])
    arb = StorageArbitrageLP(small_panel.H, 4.0, P.storage_eff_oneway)
    assert abs(r.margin["storage4"] - arb.value(r.price)) < 1e-6 * max(1, r.margin["storage4"])
    assert abs(r.price.max() - P.voll) < 1e-6 or r.lost_load.sum() < 1e-6


# ----------------------------------------------------------------------------- agents
def test_best_chi_matches_grid_search():
    rng = np.random.default_rng(1)
    for _ in range(100):
        Y = int(rng.integers(2, 8)); g = float(rng.uniform(0, 2))
        pi0 = rng.normal(0, 1e4, Y); s = rng.normal(0, 1e4, Y)
        lo, hi, v = best_chi(pi0, s, g)
        grid = np.linspace(0, 1, 5001)
        vg = max(rho(pi0 + s * x, g) for x in grid)
        assert v >= vg - 1e-6 * (1 + abs(v))
        assert abs(rho(pi0 + s * lo, g) - v) < 1e-6 * (1 + abs(v)) and abs(rho(pi0 + s * hi, g) - v) < 1e-6 * (1 + abs(v))


def test_forward_market_clears_and_is_monotone():
    Lambda = 7000.0
    m = {"a": np.array([9e4, 8e4, 12e4, 7e4]), "b": np.array([6e4, 9e4, 5e4, 8e4]), "c": np.array([10e4] * 4)}
    I = {"a": 9e4, "b": 7e4, "c": 10e4}; K = {"a": 30000.0, "b": 20000.0, "c": 5000.0}
    Pbar = np.array([40.0, 35.0, 50.0, 30.0]) * Lambda
    last = -np.inf
    for qb in [5000.0, 10000.0, 30000.0, 54000.0, 55000.0]:
        fr = forward_market(m, I, K, Pbar, Lambda, gamma=0.5, q_bar=qb, markdown={}, contracts=True)
        assert abs(fr.supply - qb) < 1e-6 * qb and not fr.rationed
        assert all(0 <= c <= 1 for c in fr.chi.values())
        assert fr.p_hat >= last - 1e-9; last = fr.p_hat        # price non-decreasing in Q_bar
    fr = forward_market(m, I, K, Pbar, Lambda, gamma=0.5, q_bar=60000.0, markdown={}, contracts=True)
    assert fr.rationed
    fr0 = forward_market(m, I, K, Pbar, Lambda, gamma=0.5, q_bar=1e4, markdown={}, contracts=False)
    assert all(v == 0 for v in fr0.chi.values()) and np.isnan(fr0.p_hat)


# ----------------------------------------------------------------------------- equilibrium
@pytest.fixture(scope="module")
def panel3():
    return make_synthetic_panel(years=(2017, 2018, 2019), H=24 * 28, seed=0)


def test_planner_cutting_planes_match_monolithic_lp(panel3):
    P = ModelParams()
    ref = SystemLP(panel3, P, tau=P.tau_scc, include_psi=True).solve()
    cap = CapacityProblem(panel3, P, tau=P.tau_scc)
    r = solve_regime(panel3, P, REGIMES["P2"], cap=cap, verbose=False)
    assert r.converged and r.capacity_result.completed
    for z in ref.K:
        assert abs(r.K[z] - ref.K[z]) <= 1e-4 * max(1.0, ref.K[z]) + 0.5, z
    assert abs(r.capacity_result.objective - ref.obj) <= 1e-7 * abs(ref.obj)
    assert r.capacity_result.foc_residual < 1e-6
    assert np.abs(r.evaluation.price - ref.price).max() < 1e-4        # exact KKT prices recovered


def test_market_at_gamma0_equals_planner_without_reliability_cost(panel3):
    P = ModelParams()
    cap = CapacityProblem(panel3, P, tau=0.0)
    ref = SystemLP(panel3, P, tau=0.0, include_psi=False).solve()
    r = solve_regime(panel3, P, REGIMES["R2"], gamma=0.0, cap=cap, verbose=False)
    assert r.converged and len(r.outer_history) == 1
    for z in ref.K:
        assert abs(r.K[z] - ref.K[z]) <= 1e-4 * max(1.0, ref.K[z]) + 0.5, z
    assert all(abs(v) < 1e-9 for v in r.risk_premium.values())


@pytest.mark.parametrize("regime", ["R2", "R3", "R4", "R6"])
def test_market_equilibrium_zero_risk_adjusted_profit(panel3, regime):
    P = ModelParams(gamma=0.4)
    r = solve_regime(panel3, P, REGIMES[regime], verbose=False, tol=1e-3)
    assert r.converged, r.outer_history[-1]
    for z, K in r.K.items():
        if K > 1.0:
            assert abs(r.rho_star[z]) <= 1e-3 * r.I[z], (z, r.rho_star[z])         # active: zero profit
        else:
            assert r.rho_star[z] <= 1e-3 * r.I[z], (z, r.rho_star[z])              # inactive: no entry
    if REGIMES[regime].contracts:
        assert abs(r.forward.supply - r.q_bar) <= 1e-6 * r.q_bar and np.isfinite(r.forward.p_hat)
    else:
        assert all(v == 0.0 for v in r.forward.chi.values())
    w = evaluate_welfare(panel3, P, r)
    assert w.C_risk_adjusted >= w.C_mean and np.isfinite(w.C_mean)
    # active technologies carry a positive risk premium in the merchant regime
    if regime == "R2":
        assert all(r.risk_premium[z] > 0 for z, K in r.K.items() if K > 1.0)


def test_first_best_planner_has_lowest_risk_neutral_welfare_cost(panel3):
    P = ModelParams(gamma=0.4)
    caps = {}
    costs = {}
    for name in ["P2", "R5", "R6"]:
        reg = REGIMES[name]
        tau = P.tau_scc if reg.carbon_priced else 0.0
        caps.setdefault(tau, CapacityProblem(panel3, P, tau))
        r = solve_regime(panel3, P, reg, cap=caps[tau], verbose=False)
        costs[name] = evaluate_welfare(panel3, P, r).C_mean
    assert costs["P2"] <= min(costs.values()) * (1 + 1e-6)

# ----------------------------------------------------------------------------- capital-scaled gamma
def test_gamma_by_tech():
    P = ModelParams()
    assert set(P.gamma_by_tech(0.3).values()) == {0.3}                  # default: uniform
    Pc = ModelParams(gamma=0.3, gamma_scaling="capital")
    g = Pc.gamma_by_tech()
    assert g == Pc.gamma_by_tech(0.3)
    assert abs(g["ccgt"] - 0.3) < 1e-12                                  # reference keeps gamma
    assert 3.2 < g["nuclear"] / g["ccgt"] < 3.4                          # capital recovery ratio
    assert g["ct"] < g["ccgt"] < g["nuclear"]
    with pytest.raises(ValueError):
        ModelParams(gamma_scaling="bogus").gamma_by_tech(0.3)


def test_forward_market_uniform_dict_equals_scalar():
    Lambda = 7000.0
    m = {"a": np.array([9e4, 8e4, 12e4, 7e4]), "b": np.array([6e4, 9e4, 5e4, 8e4]), "c": np.array([10e4] * 4)}
    I = {"a": 9e4, "b": 7e4, "c": 10e4}; K = {"a": 30000.0, "b": 20000.0, "c": 5000.0}
    Pbar = np.array([40.0, 35.0, 50.0, 30.0]) * Lambda
    for contracts in (True, False):
        a = forward_market(m, I, K, Pbar, Lambda, gamma=0.5, q_bar=30000.0, markdown={}, contracts=contracts)
        b = forward_market(m, I, K, Pbar, Lambda, gamma={z: 0.5 for z in m}, q_bar=30000.0,
                           markdown={}, contracts=contracts)
        assert a.chi == b.chi and a.rho_star == b.rho_star
        assert (np.isnan(a.p_hat) and np.isnan(b.p_hat)) or a.p_hat == b.p_hat


@pytest.mark.parametrize("regime", ["R2", "R3"])
def test_market_equilibrium_capital_scaled_gamma(panel3, regime):
    P = ModelParams(gamma=0.4, gamma_scaling="capital")
    r = solve_regime(panel3, P, REGIMES[regime], verbose=False, tol=1e-3)
    assert r.converged, r.outer_history[-1]
    assert r.gamma_by_tech == P.gamma_by_tech(0.4)
    for z, K in r.K.items():
        if K > 1.0:
            assert abs(r.rho_star[z]) <= 1e-3 * r.I[z], (z, r.rho_star[z])         # active: zero profit
        else:
            assert r.rho_star[z] <= 1e-3 * r.I[z], (z, r.rho_star[z])              # inactive: no entry
    u = solve_regime(panel3, ModelParams(gamma=0.4), REGIMES[regime], verbose=False, tol=1e-3)
    if r.K["nuclear"] > 1.0 and u.K["nuclear"] > 1.0:
        assert r.risk_premium["nuclear"] >= u.risk_premium["nuclear"]
