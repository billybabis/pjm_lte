# eq_model — PJM-calibrated long-run equilibrium model

A greenfield, single-node, 8760 h × Y weather-year capacity-expansion model of PJM with two
risk-neutral planner benchmarks (P1, P2) and five decentralised market regimes (R2–R6) in which
risk-averse generators and storage evaluate annual profits with

    ρ(Π) = mean_y Π_y + γ · min_y Π_y

and can hedge with an E-SFPFC forward contract. It implements the model of
`empirical_model_setup.tex` with the parameters of `parameters.tex`.

Everything is LP-based (HiGHS via `highspy`); no commercial solver is required.

```
pip install -r requirements.txt
python -m pytest -q tests        # ~1-2 min, 17 tests on synthetic data
python -m eq_model params        # the parameter table as implemented
```

The loaders are written against the documented PJM Data Miner 2 and EIA-860M layouts and are
tested against synthetic files in those layouts (`eq_model/synthetic.py`). With a new data
vintage, run `check-load` and `build-panel` first and read the diagnostics they print (§2).

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
| `eq_model/cli.py` | `check-load`, `build-panel`, `run`, `sweep`, `combine`, `plot`, `params` |
| `eq_model/plots.py` | capacity/curtailment figure (stacked bars by regime, one panel per γ) |
| `eq_model/synthetic.py` | synthetic raw files / panels for tests |
| `tests/test_model.py` | pytest suite |
| `.github/workflows/run-model.yml` | one CI job per regime, results merged by `combine` |
| `docs/github-actions.md` | how to dispatch a run, collect artifacts, and build the multi-γ figure |
| `scripts/fetch_run.py` | download the latest CI run into `results/raw/<scenario>/g<γ>/`, check it, plot it |
| `scripts/check_results.py` | verify a results tree: every regime present, every run converged |

## 2. Data pipeline

```
# 1) inspect how the load export classifies its rows (zones vs. aggregates) before anything else
python -m eq_model check-load --load data/raw/load

# 2) build the hourly panel (recomputes monthly capacity + coverage from EIA-860M and
#    cross-checks it against vre_capacity.csv)
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
  mean/max relative difference between the zonal sum and the RTO row. A mean difference that is
  not tiny (≳0.5 %) indicates a misclassified label; extend `RTO_LABELS` / `REGION_LABELS` in
  `pjm_data.py`, or pass `extra_rto_labels` / `extra_region_labels`. Summing every row of the
  export naively triples the load (test `test_area_resolution…`).
* **Numbers.** All files are read as strings and converted with `parse_number`, so
  `"1,281.0"` → 1281.0 and any non-blank unparsable value raises instead of becoming NaN/0.
  The loader logs how many values carried thousands separators.
* **AC basis.** `Nameplate Capacity (MW)` is used for capacity; `DC Net Capacity (MW)` is read
  and reported for solar only. See §5.1 for the one place where the appendix is not on an AC basis.
* **Coverage.** `monthly_vre_capacity` sums generators to *plant* level (Plant ID × technology)
  month by month (operating date ≤ month < retirement date) and computes
  coverage = capacity in plants > `--threshold-mw` / total. θ = gen / (cap × coverage), clipped to
  [0, 1] with counts of negative/over-1 hours reported. If `vre_capacity.csv` has no coverage
  columns and the EIA files are not passed, coverage = 1 and θ is biased low (warning).
* **Hours.** Years are local (America/New_York) calendar years; hours are chronological in UTC
  (DST creates no gaps/duplicates); Feb 29 is dropped so every year has exactly 8760 hours;
  small gaps are interpolated and reported. λ_h = mean intraday load profile by local hour /
  its max. Daylight = θ_solar > 0.01 (or fixed hours; `--param daylight_rule=fixed_hours`).

## 3. Running the model

```
python -m eq_model run   --panel data/panel.npz --regimes P1,P2,R2,R3,R4,R5,R6 --gamma 0.3 --out results/raw/default/g0.3
python -m eq_model sweep --panel data/panel.npz --regime R2 --gammas 0,0.1,0.25,0.5,1 --out results/raw/sweep
python -m eq_model run   --panel data/panel.npz --years 2017-2020,2022-2025 ...   # leave-one-year-out
python -m eq_model run   ... --param voll=20000 --param q_bar_rule=peak_load --param markdown_mode=full
python -m eq_model run   ... --workers 4                                          # parallel yearly dispatch
python -m eq_model run   ... --jobs 4                                             # parallel regimes
python -m eq_model combine results/ --out results/summary.csv                     # stitch split runs
python -m eq_model plot results/raw/SCEN --kind all --out results/figs/SCEN/fig.png  # figures
```

Outputs per regime: `result_<R>.json` (K, χ, p̂, effective costs, risk premia, implied
discount rates, ρ*, convergence history, welfare block), `profits_<R>.csv` (per-MW annual
profit by technology and year), `prices_<R>.npz` (hourly prices and lost load), `row_<R>.json`
(that regime's summary row on its own), and `summary.csv` across regimes.

Curtailment is reported two ways. `curt_share_<r>` is the fraction of available VRE energy
θ_rh·K_r that went undispatched; `curt_hours_<r>_mean` and `curt_hours_any_mean` count hours per
year in which curtailment exceeded `curtailment_hour_threshold_mw` (default 1 MW — an absolute
floor, so that LP round-off in hours where VRE is fully absorbed does not register). The threshold
is a reporting convention only and does not affect the solution.

### Parallelism

There are two independent axes, and they multiply:

* `--workers M` splits the *yearly dispatch LPs* of one solve over M processes (M ≤ number of
  years). Results are identical; the price-completion step stays single-process.
* `--jobs N` splits the *regimes* (`run`) or the *gammas* (`sweep`) over N processes. Regimes
  sharing a τ stay in one process while there are no spare jobs, because they share the cut pool
  and each warm-starts K from the previous one — so `--jobs 2` on the default regime list costs
  nothing, while `--jobs 7` buys wall-clock at the price of re-deriving cuts and cold K starts in
  every process. Each process holds its own cut pool and monolithic LP (≈ 3 GB at full size), so
  N × 3 GB is the memory bound.

### Figures

```
python -m eq_model plot results/raw/default --kind all --gammas 0,0.3,0.6,1 --out results/figs/default/fig.png
```

`--kind` selects the figure: `capacity`, `energy`, `contract`, `both` (capacity + energy) or
`all`. With more than one the kind is appended to the output stem.

* **capacity** — installed K_z stacked by technology, mean annual curtailment *hours* on a
  secondary axis (`--no-curtailment` drops the overlay).
* **energy** — share of load served, stacked by technology; columns sum to 100 %. Storage is net
  of round-trip losses so its segment is drawn below the axis; unserved energy is its own segment.
  Curtailed VRE (% of available) on the secondary axis.
* **contract** — contracted capacity Q_z = χ_z·K_z stacked by technology, with the forward
  clearing price p̂ on the secondary axis and a dashed line at Q̄. Every contracting regime clears
  at Σ_z Q_z = Q̄, so the bars are all the same height and the figure is about the *mix*. Only
  R3/R4/R5 have forward markets; the other regimes are omitted (`--keep-empty` keeps their slots
  labelled, so the regime axis lines up with the other two figures).

All three share the γ-panel layout: Technology colours are fixed in
`plots.py::TECH_COLORS` (Okabe–Ito, colour-blind safe) so a technology keeps its colour across
every figure. Output format follows the extension — `.png` by default, `.pdf` for LaTeX —
and `--unit MW`, `--panel-width`, `--height`, `--dpi` control the rest.

The planner regimes P1 and P2 are risk-neutral by construction and solve identically at every γ.
They are drawn in every panel as the fixed benchmark, which is why the summary rows carry both
`gamma` (what the regime used — always 0 for a planner) and `gamma_requested` (what the run was
launched with). Panels group on the latter.

### Scenarios

A scenario is a set of `--param` overrides. Give each its own output directory and point `plot`
and `combine` at one directory at a time:

```
python -m eq_model run ... --param voll=20000 --out results/raw/high_voll/g0.3
python -m eq_model plot results/raw/high_voll --kind all --gammas 0,0.3,0.6,1 --out results/figs/high_voll/fig.png
```

Results are keyed by `(regime, gamma_requested)`, which does not include the scenario, so two
scenarios cannot share one table or figure. Pointing a command at a directory spanning two of them
is refused, naming both files and the first field that differs — it is never silently merged.
Results that genuinely reached the command twice with identical values (one regime downloaded in
two CI artifact directories, say) are deduplicated quietly.

Every `result_<R>.json` carries a `params` block with the full `ModelParams` used, so a results
directory records the settings that produced it rather than relying on the directory name.

`combine` stitches the `row_*.json` files back into one `summary.csv` when the work is split
across *machines* rather than processes — as in the `run-model` GitHub Actions workflow
(`.github/workflows/run-model.yml`), which builds the panel once and then runs one matrix job per
regime. It scans its directory arguments recursively, so pointing it at a directory of downloaded
artifacts is enough. See [docs/github-actions.md](docs/github-actions.md) for the dispatch,
artifact-collection and figure-building steps.

### Runtime

At full size (9 × 8760 h, synthetic data, γ = 0.3) on two cores: P1 2.8 min, P2 4.4 min, R2
6.2 min, R3 6.9 min, R4 8.0 min — 4–5 outer iterations each, one exact price completion per outer
iteration. The τ = SCC market regimes carry 7 active technologies and take ≈ 5 min per outer
iteration, i.e. 20–40 min each. Memory ≈ 3 GB per process.

## 4. Model → code mapping and the solution method

**Planner (P1/P2)** is exactly eq. (planner): the LP in `lp.py::SystemLP`. It is *solved* by
decomposition, because the monolithic LP has 8 dense columns (each K_z touches every hour), which
makes interior point slow (215 s for one year at full size). `dispatch.py::DispatchLP` solves one
year with K fixed as variable bounds in ~0.2–0.8 s; `equilibrium.py::CapacityProblem` runs a
trust-region cutting-plane (Benders) method over the 8-dimensional K using the yearly dispatch
values V_y(K) and their subgradients (the capacity duals = per-MW margins).
`test_planner_cutting_planes_match_monolithic_lp` checks that the result coincides with the
monolithic LP (same K, same objective, same prices).

**Price completion.** At the optimal K the dispatch LPs are dual-degenerate in the few hours where
load exactly equals installed capacity: any price between the marginal unit's cost and VOLL is a
valid dual there. Those few hours matter — in one test they moved the CT's break-even by 22 % —
and the KKT conditions of the capacity problem (zero profit) pin them. `CapacityProblem.complete`
recovers the exact prices by re-solving the monolithic LP warm-started from the dispatch solution
with K boxed around the cutting-plane optimum (≈ 1.5 min at full size instead of a cold solve).
The prices used everywhere are these KKT-consistent prices.

**Market regimes (R2–R6).** With Q_z := χ_z K_z (0 ≤ Q_z ≤ K_z) every agent's problem is an LP for
given prices, and the forward clearing is Σ_z Q_z = Q̄ (`agents.py`). The *equilibrium* of
risk-averse agents with a single, incomplete hedging instrument is not itself a single LP or QP:
each agent weights years with its own risk-adjusted probability (1/Y + γ on its own worst year),
so there is no common objective (Ehrenmann–Smeers 2011; Ralph–Smeers 2015). What does hold is that,
given prices, every agent's problem is homogeneous of degree 1 in (K_z, Q_z), so equilibrium ⇔ zero
per-MW risk-adjusted profit for active technologies and ≤ 0 for inactive ones, with the spot
dispatch being the cost-minimising dispatch given K. Writing the active condition as
Σ_y f_y m_zy(K) = I_eff_z with I_eff_z := Σ_y f_y m_zy − ρ*_z, the equilibrium K solves the
planner-form problem with *effective* (risk-premium loaded) capacity costs. `solve_regime` iterates
on I_eff (safeguarded per-technology secant steps; exact one-step update for inactive
technologies), re-solving the capacity problem with the shared cut pool and clearing the forward
market each time:

* spot margins per MW: thermal Σ_h (p − c − eτ)⁺, VRE Σ_h θ p⁺, storage = arbitrage LP with
  K_s = 1 (exact also when K_s = 0, where LP duals are degenerate);
* contract choice: χ_z maximises the concave piecewise-linear ρ(π₀ + s χ) on [0, 1] — the
  maximiser can be an interval; p̂ is found by bisection on the (set-valued, monotone) contract
  supply, and indifferent agents share the residual pro rata (`forward_market`);
* markdown a_r (R4/R5) enters as s_y = Λ(p̂ − a_z) − P̄_y with Λ = Σ_h λ_h, P̄_y = Σ_h λ_h p_hy;
* merchant regimes: χ = 0.

Convergence criterion: max_z |ρ*_z|/I_z ≤ `--tol` (1e-3) over active technologies and
ρ*_z ≤ tol·I_z for inactive ones, plus the FOC of the capacity problem. `result_<R>.json` records
the full history. At γ = 0 the market regimes reproduce the planner without Ψ in one step
(`test_market_at_gamma0…`).

**Welfare** (`welfare.py`): C_y of eq. (cost) with Σ_r Ψ_r evaluated from the regime's K and
hourly θ; 𝒞 = mean + γ·max. Emissions are valued at the SCC in *every* regime by default
(`welfare_tau_always_scc=True`), since otherwise P1 scores better than P2; set it to False to value
them at the regime's own τ. The load-driven regulation term 0.01·D (identical across regimes) is
reported separately as `psi_load_mean` and is not included in C.

**Calibrating γ.** `result_<R>.json` reports, per active technology, the risk premium I_eff − I and
the implied real discount rate r′ solving OCC·CRF(r′, L) + FOM = I_eff — i.e. the financing spread
over r = 2.5 % that the merchant regime implies. `sweep` tabulates this against γ.

## 5. Open parameters and documented deviations from the .tex files

Each item below is a point where the appendix is silent or internally inconsistent. Each has an
explicit switch in `ModelParams`, and the default is stated.

### 5.1 Deviations

1. **Solar AC/DC divisor.** The reserve formulas divide K_solar by 1.34 (ReEDS's DC→AC
   convention). With EIA `Nameplate Capacity (MW)` — AC — used for K_solar and θ_solar, that
   division should not be applied: the ReEDS coefficients (0.04, 0.003) are per MW-AC. The
   appendix is internally inconsistent on this point: a_solar = $0.0018/MWh is what one obtains
   *without* the 1.34 (0.006·50.73·0.003·0.5·8760/(0.25·8760) = 0.00183); with it, a_solar would be
   0.00136. The printed markdown is therefore on an AC basis while the printed reserve formula is
   not. Default: `solar_capacity_basis="AC"` → divisor 1, consistent with the printed a_r.
   `--param solar_capacity_basis=DC` reproduces the formula literally (divisor 1.34).
2. **Scope of a_r.** The R4/R5 appendix markdowns (0.0015, 0.0018 $/MWh) price only the
   *regulation* term ψ_A. The flexibility-reserve term ψ_E (1.52 × 0.189 = $0.287/MWh of wind;
   ≈ $0.23/MWh of solar energy) is ~200× larger and is not in a_r, which makes R4 ≈ R3
   numerically. Default `markdown_mode="regulation"` computes a_r from the active regulation
   coefficients. If "priced via a_r" is meant to internalise Ψ in full, use
   `--param markdown_mode=full` (a_wind ≈ 0.289, a_solar ≈ 0.233 $/MWh); `markdown_mode=appendix`
   reproduces the printed values. Even under `full`, Ψ is small relative to VRE costs (≈ 1 % of
   I_r per MWh-equivalent).
3. **Ψ_r(K_r, θ̄_r)** in eq. (cost) is written with θ̄; it is evaluated here per year with hourly
   θ_hy, which is what the hourly formulas in the appendix imply.
4. **Reserve price "1.52"** is 0.03 × 50.73 = 1.5219; the product is used (the appendix rounds).
5. **Leap years.** The text specifies 8760 h; Feb 29 is dropped. `--keep-feb29` retains 8784-h
   years, which breaks the equal-length assumption of the (Y, H) arrays and is not supported.
6. **Storage OCC** is read as $/kW of *power* (ATB 4 h / 8 h systems), so I_s is per MW of power
   with d_s hours of energy — consistent with the constraints.

### 5.2 Parameters the appendix does not pin down

| parameter | default | switch |
|---|---|---|
| γ (risk aversion) | 0 | `--gamma`, `--param gamma=` |
| τ_SCC | $280/t (2025$) | `--param tau_scc=` |
| Q̄ (total contracted capacity) | 112,108.1 MW = 70 % of the observed PJM RTO peak | `q_bar_rule ∈ {fixed, mean_profile_peak, peak_load, mean_load}`, `q_bar_mw`, `q_bar_coverage` |
| τ in the welfare metric for τ = 0 regimes | SCC everywhere | `welfare_tau_always_scc` |
| daylight hours 𝒯^day | θ_solar > 0.01 | `daylight_rule`, `daylight_theta_threshold`, `daylight_fixed_hours` |

`D^max` is defined in the appendix's variable table but unused in the text; if it is the intended
reference quantity for Q̄, use `--param q_bar_rule=peak_load`.

Everything else in the parameter tables is reproduced exactly
(`test_parameter_table_matches_appendix`: c_z, e_z, I_z, CRF, 1.52/0.304, a_r).

## 6. Modelling caveats

* **Price indeterminacy is a property of the model, not only of the solver.** With inelastic
  demand and a finite sample of hours, the equilibrium K sits at an LP vertex where, in a few hours
  per year, the price is not pinned by dispatch. The zero-profit conditions pin
  Σ_y f_y·(those prices), not their split across years — but the split determines which year is an
  agent's worst year, and hence ρ. The code uses the KKT-consistent prices of the planner-form
  problem, a definite and reproducible selection, but other equilibria with the same K and slightly
  different scarcity-hour prices exist. A small price-responsive demand block (an
  operating-reserve-demand-curve-like tiering of VOLL) would make prices unique; it is not
  implemented.
* **Existence and uniqueness of the risk-averse equilibrium** are not guaranteed in general
  (incomplete risk markets). The solver reports non-convergence rather than returning a spurious
  answer; in all synthetic tests it converged in 3–6 outer iterations.
* **Homogeneity.** Each agent's problem is linear-homogeneous in (K_z, Q_z), so "χ_z ∈ [0,1] chosen
  by the firm" is determinate only up to indifference intervals at the clearing p̂; the pro-rata
  allocation among indifferent agents is a convention.
* **Sub-year panels** (`--hours N`, used in tests) prorate the annual fixed costs by N/8760. They
  are for smoke tests only.
