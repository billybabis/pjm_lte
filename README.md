# eq_model — PJM-calibrated long-run equilibrium model

Python implementation of the model in `empirical_model_setup.tex` with the parameters of
`parameters.tex`: a greenfield, single-node, 8760 h × Y weather-year capacity-expansion
model of PJM with two risk-neutral planner benchmarks (P1, P2) and five decentralised
market regimes (R2–R6) in which risk-averse generators/storage evaluate annual profits with
ρ(Π) = mean_y Π_y + γ·min_y Π_y and can hedge with an E-SFPFC forward contract.

Everything is LP-based (HiGHS via `highspy`). No commercial solver is needed.

```
pip install numpy scipy pandas highspy pytest
cd pjm_lre
python -m pytest -q tests                       # ~1-2 min, 17 tests on synthetic data
python -m eq_model params                        # parameter table as implemented
```

**I could not access your data files from this session** (no computer link), so the loaders
were written against the documented Data Miner 2 / EIA-860M layouts and tested on synthetic
files written in those layouts (`eq_model/synthetic.py`). The first thing to do with the real
files is run `check-load` and `build-panel` and read the diagnostics they print (section 2).

---

## 1. Layout

| file | what |
|---|---|
| `eq_model/params.py` | technology table, CRF, I_z, c_m + e_m τ, reliability-cost coefficients, a_r markdowns, regimes, all open parameters as documented defaults |
| `eq_model/io_utils.py` | `parse_number` (thousands separators; refuses to turn a non-blank value into NaN/0), timestamp parsing, column lookup |
| `eq_model/pjm_data.py` | `hrl_load_metered` loader with explicit area resolution + cross-check; `gen_by_fuel` loader |
| `eq_model/eia.py` | EIA-860M → monthly PJM solar/wind AC nameplate and the >10 MW plant coverage share; loader for a derived `vre_capacity.csv` |
| `eq_model/panel.py` | hourly panel D_hy, θ_rhy, λ_h, daylight_hy; 8760 h per local year; Q̄ rule |
| `eq_model/lp.py` | monolithic LP (HiGHS), storage-arbitrage LP, Ψ coefficients |
| `eq_model/dispatch.py` | one-year dispatch LP with K fixed as bounds (the workhorse) |
| `eq_model/agents.py` | risk measure, contract choice (χ), forward-market clearing, spot margins |
| `eq_model/equilibrium.py` | trust-region cutting planes over K, exact price completion, market fixed point, `solve_regime` |
| `eq_model/welfare.py` | C_y, risk-adjusted 𝒞, cost components, emissions, summary rows |
| `eq_model/cli.py` | `check-load`, `build-panel`, `run`, `sweep`, `params` |
| `eq_model/synthetic.py` | synthetic raw files / panels for tests |
| `tests/test_model.py` | pytest suite |

## 2. Data pipeline

```
# 1) look at how the load export is classified (zones vs. aggregates) before anything else
python -m eq_model check-load --load data/raw/load

# 2) build the hourly panel (recomputes monthly capacity + coverage from EIA-860M and
#    cross-checks it against your vre_capacity.csv)
python -m eq_model build-panel --load data/raw/load --gen data/raw/gen_by_fuel \
    --capacity data/raw/capacity/vre_capacity.csv \
    --eia-active data/raw/capacity/generators_active_07_2026.csv \
    --eia-retired data/raw/capacity/generators_retired_07_2026.csv \
    --save-capacity data/vre_capacity_with_coverage.csv --out data/panel.npz
```

What the loaders do, and what to check in their output:

* **Load area resolution.** Each row's `load_area` (falling back to `zone` when blank) is
  classified as RTO total (`RTO`, `PJM RTO`, …), market-region aggregate (`MIDATL`, `WEST`,
  `SOUTH`, …) or zone. D_hy is the RTO row when present and consistent with the zonal sum
  (`--area-mode auto`; force `rto` or `zonal_sum`). The report prints the label lists and the
  mean/max relative difference between the zonal sum and the RTO row. If the mean difference
  is not tiny (≲0.5 %), something is misclassified — send me the printed label lists and I
  will adjust `RTO_LABELS` / `REGION_LABELS` (or pass `extra_rto_labels` / `extra_region_labels`).
  Summing every row of the export naively triples the load (test `test_area_resolution…`).
* **Numbers.** All files are read as strings and converted with `parse_number`, so
  `"1,281.0"` → 1281.0 and any non-blank unparsable value raises instead of becoming NaN/0.
  It logs how many values had thousands separators.
* **AC basis.** `Nameplate Capacity (MW)` is used for capacity; `DC Net Capacity (MW)` is read
  and reported for solar only. See §5 for the one place where the appendix is not on an AC basis.
* **Coverage.** `monthly_vre_capacity` sums generators to *plant* level (Plant ID × technology)
  month by month (operating date ≤ month < retirement date) and computes
  coverage = capacity in plants > `--threshold-mw` / total. θ = gen / (cap × coverage), clipped to
  [0, 1] with counts of negative/over-1 hours reported. If your `vre_capacity.csv` has no
  coverage columns and you do not pass the EIA files, coverage = 1 and θ is biased low (warning).
* **Hours.** Years are local (America/New_York) calendar years; hours are chronological in UTC
  (DST creates no gaps/duplicates); Feb 29 is dropped so every year has exactly 8760 hours;
  small gaps are interpolated and reported. λ_h = mean intraday load profile by local hour /
  its max. Daylight = θ_solar > 0.01 (or fixed hours; `--param daylight_rule=fixed_hours`).

## 3. Running the model

```
python -m eq_model run   --panel data/panel.npz --regimes P1,P2,R2,R3,R4,R5,R6 --gamma 0.3 --out results/g0.3
python -m eq_model sweep --panel data/panel.npz --regime R2 --gammas 0,0.1,0.25,0.5,1 --out results/sweep
python -m eq_model run   --panel data/panel.npz --years 2017-2020,2022-2025 ...   # leave-one-year-out
python -m eq_model run   ... --param voll=20000 --param q_bar_rule=peak_load --param markdown_mode=full
python -m eq_model run   ... --workers 4                                          # parallel yearly dispatch
```

Outputs per regime: `result_<R>.json` (K, χ, p̂, effective costs, risk premia, implied
discount rates, ρ*, convergence history, welfare block), `profits_<R>.csv` (per-MW annual
profit by technology and year), `prices_<R>.npz` (hourly prices and lost load), and
`summary.csv` across regimes.

Runtime on the 2-core sandbox used here, at full size (9 × 8760 h, synthetic data, γ = 0.3):
P1 2.8 min, R2 6.2 min, R3 6.9 min, R4 8.0 min (4–5 outer iterations each, one exact price
completion per outer iteration); P2 4.4 min; the τ = SCC market regimes (7 active
technologies) take ≈ 5 min per outer iteration, i.e. 20–40 min each. Memory ≈ 3 GB. Cuts are
shared between regimes with the same τ. `--workers N` solves the yearly dispatch LPs in N
processes (results are identical; the completion step stays single-process), which should
cut the cutting-plane part roughly N-fold on a multi-core machine.

## 4. Model → code mapping and the solution method

**Planner (P1/P2)** is exactly eq. (planner): the LP in `lp.py::SystemLP`. It is *solved*
by decomposition, because the monolithic LP has 8 dense columns (each K_z touches every
hour) which makes interior point slow (215 s for one year here). `dispatch.py::DispatchLP`
solves one year with K fixed as variable bounds in ~0.2–0.8 s; `equilibrium.py::CapacityProblem`
runs a trust-region cutting-plane (Benders) method over the 8-dimensional K using the yearly
dispatch values V_y(K) and their subgradients (the capacity duals = per-MW margins).
`test_planner_cutting_planes_match_monolithic_lp` checks that the result coincides with the
monolithic LP (same K, same objective, same prices).

**Price completion.** At the optimal K the dispatch LPs are dual-degenerate in the few hours
where load exactly equals installed capacity (any price between the marginal unit's cost and
VOLL is a valid dual there). Those few hours matter — in one test they moved the CT's
break-even by 22 % — and the KKT conditions of the capacity problem (zero profit) pin them.
`CapacityProblem.complete` recovers the exact prices by re-solving the monolithic LP warm-started
from the dispatch solution with K boxed around the cutting-plane optimum (≈ 1.5 min at full
size instead of a cold solve). Prices used everywhere are these KKT-consistent prices.

**Market regimes (R2–R6).** The χ "tweak" you asked about is implemented: with
Q_z := χ_z K_z (0 ≤ Q_z ≤ K_z) every agent's problem is an LP for given prices, and the
forward clearing is Σ_z Q_z = Q̄ (`agents.py`). But the *equilibrium* of risk-averse agents
with a single, incomplete hedging instrument is not itself a single LP/QP: each agent
weights years with its own risk-adjusted probability (1/Y + γ on its own worst year), so
there is no common objective (Ehrenmann–Smeers 2011; Ralph–Smeers 2015). What *is* true is
that, given prices, every agent's problem is homogeneous of degree 1 in (K_z, Q_z), so
equilibrium ⇔ zero per-MW risk-adjusted profit for active technologies and ≤ 0 for inactive
ones, with the spot dispatch being the cost-minimising dispatch given K. Writing the active
condition as Σ_y f_y m_zy(K) = I_eff_z with I_eff_z := Σ_y f_y m_zy − ρ*_z, the equilibrium
K solves the planner-form problem with *effective* (risk-premium loaded) capacity costs.
`solve_regime` iterates on I_eff (safeguarded per-technology secant steps; exact one-step
update for inactive technologies), re-solving the capacity problem with the shared cut pool
and clearing the forward market each time:

* spot margins per MW: thermal Σ_h (p − c − eτ)⁺, VRE Σ_h θ p⁺, storage = arbitrage LP with
  K_s = 1 (exact also when K_s = 0, where LP duals are degenerate);
* contract choice: χ_z maximises the concave piecewise-linear ρ(π₀ + s χ) on [0, 1] — the
  maximiser can be an interval; p̂ is found by bisection on the (set-valued, monotone)
  contract supply, indifferent agents share the residual pro rata (`forward_market`);
* markdown a_r (R4/R5) enters as s_y = Λ(p̂ − a_z) − P̄_y with Λ = Σ_h λ_h, P̄_y = Σ_h λ_h p_hy;
* merchant regimes: χ = 0.

Convergence criterion: max_z |ρ*_z|/I_z ≤ `--tol` (1e-3) over active technologies and
ρ*_z ≤ tol·I_z for inactive ones, plus FOC of the capacity problem. `result_<R>.json` records
the full history. At γ = 0 the market regimes reproduce the planner without Ψ in one step
(`test_market_at_gamma0…`).

**Welfare** (`welfare.py`): C_y of eq. (cost) with Σ_r Ψ_r evaluated from the regime's K and
hourly θ; 𝒞 = mean + γ·max. Emissions are valued at the SCC in *every* regime by default
(`welfare_tau_always_scc=True`), otherwise P1 would look better than P2; switch it off to
value them at the regime's own τ. The load-driven regulation term 0.01·D (identical across
regimes) is reported separately as `psi_load_mean`, not included in C.

**Calibrating γ.** `result_<R>.json` reports per active technology the risk premium
I_eff − I and the implied real discount rate r' solving OCC·CRF(r', L) + FOM = I_eff, i.e. the
financing spread over r = 2.5 % that the merchant regime implies. `sweep` tabulates this
against γ.

## 5. Inconsistencies found in the .tex files, and open parameters

Things I would like you to confirm; each has a switch in `ModelParams`.

1. **Solar AC/DC divisor (you asked me to flag this).** The reserve formulas divide K_solar by
   1.34 (ReEDS's DC→AC conversion). With EIA `Nameplate Capacity (MW)` — AC — used for K_solar
   and θ_solar, that division should *not* be applied: the ReEDS coefficients (0.04, 0.003) are
   per MW-AC. Evidence inside the appendix itself: a_solar = $0.0018/MWh is what one gets
   *without* the 1.34 (0.006·50.73·0.003·0.5·8760/(0.25·8760) = 0.00183); with it, a_solar would
   be 0.00136. So the printed markdown is on an AC basis while the printed reserve formula is
   not. Default here: `solar_capacity_basis="AC"` → divisor 1 (consistent with the printed a_r).
   `--param solar_capacity_basis=DC` reproduces the formula literally (divisor 1.34).
2. **Scope of a_r.** The default `markdown_mode=regulation` calculates a_r from the active
  regulation coefficients. The R4/R5 appendix markdowns (0.0015, 0.0018 $/MWh) price only the *regulation*
   term ψ_A. The flexibility-reserve term ψ_E (1.52 × 0.189 = $0.287/MWh of wind; ≈ $0.23/MWh of
   solar energy) is ~200× larger and is not in a_r, so R4 ≈ R3 numerically. If "priced via a_r"
   is meant to internalise Ψ, use `--param markdown_mode=full` (a_wind ≈ 0.289, a_solar ≈ 0.233
  $/MWh). Use `markdown_mode=appendix` to reproduce the printed appendix values.
   Even then Ψ is small relative to VRE costs (≈ 1 % of I_r per MWh-equivalent).
3. **Q̄ is not specified.** The forward clearing needs the total contracted capacity. Default:
   Q̄ = max_h D̄_h (peak of the mean intraday profile) so that λ_h·Q̄ equals the mean load in
   hour-of-day h; `q_bar_rule ∈ {mean_profile_peak, peak_load, mean_load, fixed}` and
   `q_bar_coverage` scale it. D^max is defined in the variable table but unused in the text I
   have — it may be the intended reference (`q_bar_rule=peak_load`).
4. **γ is "calibrated"** — no value. Default 0 in `ModelParams`; pass `--gamma`.
5. **τ in the welfare metric** for τ = 0 regimes — see §4 (default: SCC everywhere).
6. **Ψ_r(K_r, θ̄_r)** in eq. (cost) is written with θ̄; I evaluate it per year with hourly θ_hy,
   which is what the hourly formulas in the appendix imply.
7. **Reserve price "1.52"** is 0.03 × 50.73 = 1.5219; I use the product (the appendix rounds).
8. **Daylight hours 𝒯^day** are not defined; default θ_solar > 0.01 (data-driven).
9. **Leap years**: the text says 8760 h; Feb 29 is dropped (`--keep-feb29` keeps 8784-h years,
   which then breaks the equal-length assumption — not supported by the (Y, H) arrays).
10. **τ = $280/t.** EPA's $230/t (2020$, 2030 emissions, 2 % near-term Ramsey) is confirmed
    ([EPA values summary](https://costofcarbon.org/epa-values-for-the-social-cost-of-greenhouse-gases),
    [EPA report](https://www.epa.gov/system/files/documents/2023-12/epa_scghg_2023_report_final.pdf)).
    261/230 = 1.135, but the GDP implicit price deflator is 118.0 (2022) → 129.0 (2025) on FRED's
    2017 = 100 index ([FRED A191RD3A086NBEA](https://fred.stlouisfed.org/series/A191RD3A086NBEA)),
    consistent with your 1.09 for 2022→2025, and 2020→2025 is ≈ 1.22 (2020 ≈ 105.4, from memory —
    please verify), which would give ≈ $280/t. I left 261 as the default; override with
    `--param tau_scc=...`.
11. Typos: storage firms maximise ρ(Π_m) (should be Π_s); `C_r` vs `c_m`; the "Technology
    parameters" subsection is duplicated; "gamma" without `$`.
12. **Storage OCC** is taken as $/kW of *power* (ATB 4 h / 8 h systems), so I_s is per MW of
    power with d_s hours of energy — consistent with the constraints.

Everything else in the tables is reproduced exactly (`test_parameter_table_matches_appendix`:
c_z, e_z, I_z, CRF, 1.52/0.304, a_r).

## 6. Modelling caveats you should know about

* **Price indeterminacy is a property of the model, not only of the solver.** With inelastic
  demand and a finite sample of hours, the equilibrium K sits at an LP vertex where, in a few
  hours per year, the price is not pinned by dispatch. The zero-profit conditions pin
  Σ_y f_y·(those prices), not their split across years — but the split affects which year is an
  agent's worst year and hence ρ. The code uses the KKT-consistent prices of the planner-form
  problem (a definite, reproducible selection), but other equilibria with the same K and
  slightly different scarcity-hour prices exist. If this matters for your results, a small
  price-responsive demand block (an operating-reserve-demand-curve-like tiering of VOLL) would
  make prices unique; I have not implemented that.
* **Existence/uniqueness of the risk-averse equilibrium** is not guaranteed in general
  (incomplete risk markets). The solver reports non-convergence rather than pretending;
  in all synthetic tests it converged in 3–6 outer iterations.
* **Homogeneity.** Each agent's problem is linear-homogeneous in (K_z, Q_z), so "χ_z ∈ [0,1]
  chosen by the firm" is only determinate up to indifference intervals at the clearing p̂;
  the pro-rata allocation among indifferent agents is a convention.
* Sub-year panels (`--hours N`, tests) prorate the annual fixed costs by N/8760 — only for
  smoke tests.
