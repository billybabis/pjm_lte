# Model overview: classification, software, solution method

Companion to `docs/model_choices.md` (formulas) and `docs/output_columns.md` (outputs).

Notation: z = technology, y = weather year (Y = 9, equally likely), h = hour (H = 8760),
K_z = capacity (MW), I_z = annualised capital + fixed O&M cost ($/MW-yr), c_z = marginal operating
cost ($/MWh), p_hy = spot price, m_zy(K) = spot margin per MW, γ = risk aversion.

---

## 1. What kind of model this is

**A static, long-run, partial-equilibrium model of an energy-only wholesale electricity market
with endogenous investment, under uncertainty, with risk-averse investors and incomplete markets.**

Each part of that description:

- **Partial equilibrium.** Only the electricity sector is modelled. Fuel prices, capital costs and
  the discount rate are exogenous. Demand is perfectly inelastic up to the value of lost load (VOLL).
- **Long-run, static ("steady-state") equilibrium.** Capacity is fully adjustable: no legacy fleet,
  no build-out path. Costs are annualised (I_z = OCC_z·CRF(r, L_z) + FOM_z), and the model solves for
  the capacity mix that a representative year sustains in equilibrium. This is the
  **peak-load pricing / screening-curve** tradition (Boiteux 1949; Steiner 1957;
  Crew, Fernando & Kleindorfer 1995), extended to chronological hourly dispatch with storage and
  variable renewables.
- **Energy-only market with scarcity pricing.** Generators earn only spot-energy rents. Fixed costs
  are recovered from inframarginal rents and from scarcity rents in hours where the price rises to
  VOLL. There is no capacity market.
- **Perfect competition.** Firms are atomistic price-takers with constant returns to scale and
  free entry, so equilibrium is characterised by **zero (risk-adjusted) economic profit** for every
  technology that is built.
- **Uncertainty.** The state of the world is the weather/load year. Uncertainty is represented by
  the **empirical distribution** of 9 historical years (2017–2025), each with probability 1/Y:
  a two-stage structure in which investment (first stage) precedes the realisation of the year and
  dispatch (second stage) responds to it.
- **Risk-averse agents (mean–variance).** Each firm evaluates its stochastic per-MW profit π_zy with
  the **mean–variance** criterion ρ(π) = E[π] − γ·Var(π) (Markowitz 1952), with profit measured in
  $M per MW-yr, so γ is a coefficient of absolute risk aversion per $M. Applying it per MW keeps each
  firm's problem homogeneous of degree one in capacity (constant returns), which is what makes the
  free-entry (zero-profit) characterisation valid. In dual form, each firm prices profits with its
  own **risk-adjusted probability measure**, weight (1/Y)·(1 − 2γ(π_y − E[π])), which puts more
  weight on its own bad years. Mean–variance hedging in electricity forward markets follows
  Bessembinder & Lemmon (2002).
- **Incomplete markets.** The only risk-sharing instrument is one standardised financial forward
  contract (the shaped fixed-price forward contract, SFPFC): a swap on the load-shaped spot price,
  traded at a single price p̂ in a forward market where total contracted capacity equals an
  exogenous quantity Q̄. Because one instrument cannot span the 9 states, firms' risk-adjusted
  measures remain heterogeneous in equilibrium.
- **Equilibrium concept: a risk-averse stochastic (competitive) equilibrium.** The equilibrium is a
  set of capacities, spot prices, a forward price and contract positions such that (i) spot markets
  clear hourly, (ii) the forward market clears, (iii) each firm's contract choice maximises ρ, and
  (iv) the zero-profit / free-entry complementarity conditions hold. With incomplete markets and
  heterogeneous risk measures this is a **mixed complementarity problem (MCP)** that is **not**
  equivalent to any single optimisation problem (Ehrenmann & Smeers 2011, *Oper. Res.*;
  Ralph & Smeers 2015, *SIAM J. Optim.*). Existence holds under standard conditions; uniqueness is
  not guaranteed.
- **Benchmark.** SP is the **first-best social planner** (central-planning, cost-minimising)
  problem: risk-neutral, with carbon priced at the social cost of carbon and the reliability cost of
  renewables internalised. By the **first welfare theorem**, the risk-neutral competitive
  market (γ = 0) reproduces the planner without the reliability cost. With γ > 0 the market departs
  from the planner, and the difference measures the welfare cost of risk aversion with incomplete
  hedging.
- **Policy regimes** differ in instruments: DM (merchant, no contracts), DM-FC (SFPFC contracts),
  DM-FC-tau (SFPFC contracts and a carbon price) and DM-tau (carbon price, no contracts). In all four
  the reliability cost of renewables is socialised (paid outside the market).

## 2. Main Python packages

| package | role in this model |
|---|---|
| **highspy** | Python interface to **HiGHS**, the open-source LP solver that does almost all the computation: the yearly dispatch LPs (dual simplex, warm-started when K changes), the monolithic system LP used for price completion (interior point + crossover, or warm-started simplex), and the storage-arbitrage LPs used to value 1 MW of storage. Wrapped in `eq_model/lp.py` (`HighsModel`). |
| **scipy** | `scipy.optimize.linprog` (HiGHS backend) solves the small Benders master LP (`equilibrium.py:_master`). `scipy.sparse` builds the sparse constraint matrices of the dispatch and system LPs. |
| **numpy** | All array computation: hourly prices (Y × H), margins, risk measure, contract payoffs. |
| **pandas** | Loading and cleaning the PJM load and EIA-860M capacity data, building the hourly panel, and writing the output tables (`summary.csv`, `profits_*.csv`). |
| **matplotlib** | Figures (`eq_model/plots.py`). |
| **tabulate** | Text tables in console output. |
| **multiprocessing** (standard library) | Solves the 9 yearly dispatch LPs in parallel. |
| **pytest** | Test suite (`tests/test_model.py`), e.g. that the decomposition reproduces the monolithic LP and that the market at γ = 0 reproduces the planner. |

## 3. How the model is solved

The algorithm is a set of nested loops. From the inside out:

### 3.1 Spot market given capacity: LP and LP duality
For fixed K, each year's competitive spot market is computed as the **economic dispatch LP**
(`dispatch.py`): minimise fuel, variable O&M, carbon and VOLL cost subject to hourly energy
balance, capacity limits, renewable availability θ_zhy·K_z, and storage dynamics. Spot prices
p_hy are the **dual variables (shadow prices)** of the hourly balance constraints. With
price-taking firms, the cost-minimising dispatch and these dual prices *are* the competitive spot
equilibrium (LP duality: the dual prices support the dispatch as each firm's profit-maximising
output).

Two outputs feed the outer loops:
- the year's operating cost V_y(K), and
- the margins m_zy(K), which by the envelope theorem are the subgradient: ∂V_y/∂K_z = −m_zy.

### 3.2 Capacity given a target: Benders decomposition
The outer loops need a function that maps a vector of target average margins κ ($/MW-yr, one per
technology) to the capacity mix K at which each built technology's average margin equals κ_z
(and unbuilt ones earn less). That K solves the two-stage stochastic LP

    min_K  Σ_z κ_z K_z + (1/Y) Σ_y V_y(K),

whose first-order (KKT) conditions are exactly (1/Y) Σ_y m_zy(K) = κ_z for K_z > 0 and ≤ κ_z for
K_z = 0. It is solved by **Benders decomposition (L-shaped method) with a trust region**
(`equilibrium.py: CapacityProblem.solve`):
1. evaluate the 9 dispatch LPs at the current K → V_y(K) and m_zy(K) → one **optimality cut** per
   year, θ_y ≥ V_y(K_j) − m_y(K_j)·(K − K_j);
2. solve the master LP over K and the epigraph variables θ_y using all cuts so far, restricted to a
   box (trust region) around the incumbent;
3. accept or reject the step and resize the trust region; stop when the upper bound (true cost) and
   lower bound (master value) agree.
Cuts depend only on the data and τ, so they are pooled and reused across iterations, regimes and γ
values.

**Price completion** (`CapacityProblem.complete`). At the optimal K the dispatch LPs are
**dual-degenerate** in hours where load exactly equals capacity: any price between the marginal
unit's cost and VOLL is a valid dual there. These scarcity prices determine fixed-cost recovery,
so the code re-solves the monolithic (all-years) LP with K boxed tightly around the Benders
solution to select the duals consistent with the KKT conditions of the capacity problem.

For SP, κ_z is set to the true capacity cost (I_z plus the reliability cost ψ_z), and
this step alone is the planner's solution.

### 3.3 Forward market given spot outcomes: bisection
Given margins and prices (`agents.py: forward_market`):
- each technology's optimal contract share χ_z maximises the **concave quadratic**
  χ ↦ ρ(π_z(χ)) on [0, 1], in closed form: the **minimum-variance hedge ratio**
  −Cov(π_z, s_z)/Var(s_z) plus a speculative term mean(s_z)/(2γ′Var(s_z)), clipped to [0, 1]
  (Johnson 1960; Stein 1961). Only at γ = 0 can the maximiser be an interval (indifference);
- aggregate contract supply Σ_z χ_z K_z is a **monotone** function of p̂ (continuous for γ > 0), so
  p̂ is found by **bisection** until supply equals Q̄; indifferent firms share the remainder
  pro rata.
This yields ρ*_z, each technology's maximal risk-adjusted profit per MW, using the true I_z and γ.

### 3.4 Market equilibrium: solving the complementarity conditions
The market equilibrium conditions are

    ρ*_z(K) = 0  for built technologies,   ρ*_z(K) ≤ 0  for unbuilt ones.

The code solves them by searching over κ, which is a re-parameterisation of K: each κ corresponds
to the capacity mix whose average margins equal κ (§3.2). κ is endogenous (an average revenue
of the guessed mix), **not** a model parameter; I_z, c_z and γ are never changed. In the code
it is named `I_eff`.

Loop (`equilibrium.py: solve_regime`):
1. start from κ = I;
2. §3.2: compute K(κ), then prices and margins;
3. §3.3: clear the forward market and compute ρ*_z with the true costs and γ;
4. update κ_z for every built technology by a **safeguarded secant (diagonal quasi-Newton) step**
   on the residual ρ*_z: if ρ*_z > 0 the technology is too profitable, so κ_z is lowered and the
   next K contains more of it; unbuilt technologies take an exact one-step update. Step limits
   and backtracking to the best iterate handle the small jumps in margins caused by re-priced
   scarcity hours;
5. stop when max_z |ρ*_z| / I_z ≤ tolerance: default 10⁻² (`--tol`), i.e. every built
   technology's risk-adjusted profit is within 1% of its capital cost.

At the solution κ_z equals technology z's equilibrium average spot margin, so
(κ_z − I_z)/I_z is reported as `premium_pct_z`: the risk premium investors require, also
expressed as an implied discount rate in `result_<R>.json`.

**Why this approach.** The MCP in §3.4 has a very large inner structure (9 × 8,760 hourly
markets). Solving it directly as one complementarity problem (e.g. with PATH) would mean handling
the full hourly system together with the non-smooth risk measure. The decomposition keeps all the
heavy computation in LPs that HiGHS solves fast and that the planner already needs, and adds only
an 8-dimensional outer search on top.

## 4. Key references

- Boiteux, M. (1949/1960). Peak-load pricing. *Journal of Business* 33(2) (English translation).
- Steiner, P. O. (1957). Peak loads and efficient pricing. *Quarterly Journal of Economics* 71(4).
- Benders, J. F. (1962). Partitioning procedures for solving mixed-variables programming problems.
  *Numerische Mathematik* 4.
- Van Slyke, R. & Wets, R. (1969). L-shaped linear programs. *SIAM J. Appl. Math.* 17(4).
- Markowitz, H. (1952). Portfolio selection. *Journal of Finance* 7(1).
- Johnson, L. L. (1960). The theory of hedging and speculation in commodity futures.
  *Review of Economic Studies* 27(3).
- Stein, J. L. (1961). The simultaneous determination of spot and futures prices.
  *American Economic Review* 51(5).
- Bessembinder, H. & Lemmon, M. L. (2002). Equilibrium pricing and optimal hedging in electricity
  forward markets. *Journal of Finance* 57(3).
- Ehrenmann, A. & Smeers, Y. (2011). Generation capacity expansion in a risky environment: a
  stochastic equilibrium analysis. *Operations Research* 59(6).
- Ralph, D. & Smeers, Y. (2015). Risk trading and endogenous probabilities in investment equilibria.
  *SIAM Journal on Optimization* 25(4).
- Huangfu, Q. & Hall, J. A. J. (2018). Parallelizing the dual revised simplex method.
  *Mathematical Programming Computation* 10 (the HiGHS solver).

Citation details are from memory; verify before use in the paper.
