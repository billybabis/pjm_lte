# How the model with risk is solved

What the code computes, written as formulas. Everything here comes from the current code;
file references are given so each statement can be checked.

## 0. Two terms I used in conversation

- **"Column"**: a column of the output table `summary.csv` (one row per regime; e.g. `premium_pct_ccgt`,
  `chi_ccgt`, `K_ccgt_MW` are columns). Written in `eq_model/welfare.py:134`.
- **"Readout"**: an output the program *reports* after solving. It does not feed back into any
  calculation. `premium_pct_*` is one of these.

## 1. Notation

| symbol | meaning | units |
|---|---|---|
| y = 1..Y | weather/load year (Y = 9, 2017–2025) | |
| h = 1..H | hour of the year (H = 8760) | |
| z | technology (nuclear, ccgt, ct, coal, solar, wind, storage4, storage8) | |
| K_z | installed capacity | MW |
| I_z | annualised capital + fixed O&M cost = OCC·CRF(r, L) + FOM | $/MW-yr |
| p_hy | spot price | $/MWh |
| m_zy | spot margin (operating profit) of 1 MW of z in year y | $/MW-yr |
| λ_h | contract shape: mean intraday load profile, max = 1 | – |
| Λ = Σ_h λ_h | contracted MWh per MW per year (≈ 7,814) | MWh/MW |
| P̄_y = Σ_h λ_h p_hy | spot value of the contract shape in year y | $/MW-yr |
| p̂ | SFPFC (forward contract) price | $/MWh |
| a_z | markdown on the contract price (VRE only; no current regime uses it, so 0) | $/MWh |
| χ_z ∈ [0,1] | share of z's capacity that is contracted | – |
| Q̄ | total contracted capacity required (fixed, 112,108 MW) | MW |
| γ | risk aversion | – |

## 2. Spot market (all regimes)

For given capacities K, each year's spot market is the cost-minimising dispatch
(`eq_model/dispatch.py:1-12`):

    V_y(K) = min Σ_h [ Σ_thermal (c_m + e_m τ) q_mh + VOLL · l_h ]
             s.t. supply + lost load = demand in every hour   (dual = price p_hy)
                  0 ≤ q_mh ≤ K_m,  0 ≤ q_rh ≤ θ_rhy K_r,  storage limits.

This is exact for a competitive spot market: with price-taking firms, each hour's price is the
marginal cost of the last unit running, or VOLL when load is shed, and that is what this LP produces.

The spot margin per MW (`eq_model/agents.py:86-109`):

- thermal: m_zy = Σ_h max(p_hy − c_z − e_z τ, 0)
- VRE: m_zy = Σ_h θ_zhy · max(p_hy, 0)
- storage: m_zy = value of the optimal arbitrage of 1 MW given the prices

Fact used below (envelope theorem): ∂V_y/∂K_z = −m_zy. One more MW of z lowers year-y system
cost by exactly the margin that MW earns.

## 3. The social planner (SP)

    min_K  Σ_z c_z K_z + (1/Y) Σ_y V_y(K)          (c_z = I_z + ψ_z, where ψ_z = reliability cost)

**Optimality condition**, from the envelope fact:

    (1/Y) Σ_y m_zy(K) = c_z    for every technology that is built (K_z > 0)
    (1/Y) Σ_y m_zy(K) ≤ c_z    for every technology that is not built          (★)

In words: build each technology until its average margin just covers its cost.
Code: `eq_model/equilibrium.py` `CapacityProblem.solve`; the condition is checked in `_foc` (line 260).

## 4. The decentralised market (DM, DM-FC, DM-FC-tau, DM-tau)

### 4.1 One firm's profit per MW

A firm of technology z with contract share χ earns, per MW, in year y (`agents.py:18-21`):

    π_zy(χ) = m_zy − I_z + χ · s_zy
    s_zy    = Λ (p̂ − a_z) − P̄_y          (SFPFC payoff per MW: receive p̂, pay spot, on shape λ_h)

Merchant regimes (DM, DM-tau): χ = 0.

### 4.2 The firm's objective, which is where γ enters

    ρ(π) = (1/Y) Σ_y π_y − γ · Var_y(π)                 (agents.py:50-52)
    Var_y(π) = (1/Y) Σ_y (π_y − mean_y π)²,  with π measured in $M per MW-yr

so γ has units 1/($M per MW-yr); γ ≈ 1.25 gives CCGT a ≈ 25 % premium in the merchant regime.
The firm chooses χ to maximise ρ. Write the best value as

    ρ*_z = max_{χ∈[0,1]} ρ(π_z(χ))                      (agents.py:59-83)

ρ(π_z(χ)) is a concave quadratic in χ, so the best χ has a closed form:

    χ*_z = clip( −Cov_y(π_z(0), s_z) / Var_y(s_z)  +  mean_y(s_z) / (2γ′ Var_y(s_z)),  0, 1 ),   γ′ = γ/10⁶
           (minimum-variance hedge ratio)            (speculative term: buy/sell the
                                                      expected contract gain)

γ appears in exactly one place in the code: here.

### 4.3 Equilibrium conditions

Each firm's profit is (capacity) × (per-MW profit), so any size is equally good when ρ* = 0,
any positive size is unprofitable when ρ* < 0, and the firm wants infinite size when ρ* > 0.
Free entry therefore gives:

    ρ*_z(K, p̂) = 0   for every built technology
    ρ*_z(K, p̂) ≤ 0   for every unbuilt technology                                  (★★)
    Σ_z χ_z K_z = Q̄   (forward market clears; p̂ adjusts to make this hold)
    spot prices = the dispatch prices of §2 given K

These are the equations that define the market outcome. Nothing else is imposed.

## 5. Why the market is solved "like a planner", and what that does and does not mean

### 5.1 When γ = 0, the market is literally the planner

With γ = 0, ρ* = mean_y(m_zy) − I_z (the contract changes only the mean, and at γ = 0 the
forward price settles where contracting is costless). So (★★) becomes
mean_y m_zy = I_z, which is exactly the planner condition (★) with c_z = I_z, i.e. a planner
without the reliability cost ψ. This is the textbook result that a competitive market with
risk-neutral firms reaches the cost-minimising capacity. It is tested in
`test_market_at_gamma0…`.

### 5.2 When γ > 0, there is no planner, only a search procedure that reuses the planner code

With γ > 0, (★★) is not the optimality condition of any single cost-minimisation problem. Each
technology's ρ puts extra weight on *its own* bad years (CT's bad years differ from solar's),
so no single objective function reproduces all the firms' decisions at once
(Ehrenmann & Smeers 2011). The market outcome must be found by solving the equations (★★) directly.

To find K satisfying (★★), the code relies on one observation. For any K, define the number

    E_z(K) := mean_y m_zy(K) − ρ*_z(K)

Then

    ρ*_z(K) = 0   ⇔   mean_y m_zy(K) = E_z(K).

Compare this with the planner condition (★): mean_y m_zy(K) = c_z. So **if a cost vector c is
found such that (i) the planner facing costs c builds K, and (ii) c_z = E_z(K), then that K
satisfies (★★)**, and K is the market equilibrium.

The solver (`equilibrium.py:440-515`):

1. Guess a cost vector c (the code calls it `I_eff`; it starts at c = I).
2. Run the planner code with costs c to get a candidate K.
3. At that K, compute spot prices, clear the forward market, and compute every firm's true ρ*_z,
   using the **true** I_z and the true γ.
4. If ρ*_z > 0, z is too profitable at this K, so raise c_z (the planner then builds less z),
   and vice versa.
5. Stop when |ρ*_z| / I_z ≤ tolerance for every built z (`equilibrium.py:452-455, 469`).

So c (`I_eff`) is a **search variable**, the way a trial price is a search variable when you look
for the price where supply equals demand. It is not an assumption about costs, and it is not
added to anyone's profit. The test for being done is (★★), stated in terms of ρ alone. The
"planner" in step 2 is a fictitious calculation device. It does not imply that the market
outcome is efficient; at γ > 0 it generally is not.

**Why do it this way?** The planner code (Benders decomposition over yearly dispatch LPs, plus
price completion) is the expensive, carefully tested part of the model. This trick finds the
market equilibrium using only that code plus the cheap per-firm ρ calculation.

## 6. What `premium_pct_*` is

At the solution, the final search value is c_z = E_z(K*) = mean_y m_zy(K*) (because ρ*_z = 0).
The code reports

    risk_premium_z   = c_z − I_z = mean_y m_zy(K*) − I_z                       (equilibrium.py:522)
    premium_pct_z    = 100 · risk_premium_z / I_z                              (welfare.py:134)

**Meaning:** how far, in equilibrium, a technology's *average* spot margin exceeds its cost.
Under risk neutrality this is 0 (entry continues until the average margin just covers cost). Under
risk aversion firms stop entering earlier, so capacity is lower and average margins sit above cost.
`premium_pct` measures that gap. It is computed after solving and is not used in any equation.
The README's "Calibrating γ" note (§4) converts it into an implied discount rate, also only for
reporting.

**Check against the output** (merchant regime, then called R2, ccgt, from the earlier run with the min measure at γ = 0.3; the
definition is the same under variance): mean margin = $119,971/MW-yr,
I = $95,538 → 119,971 / 95,538 − 1 = 25.6 % = `premium_pct_ccgt` in `summary.csv`.

**Where it comes from in terms of ρ.** Setting ρ*_z = 0 and rearranging:

    mean_y m_zy − I_z  =  − χ_z · mean_y s_zy   +   γ′ · Var_y(π_zy)
       (premium)            (average cost of        (compensation for the year-to-year
                             the contract)           variability that remains after hedging)

- Merchant (χ = 0): the premium is entirely the second term.
- With an SFPFC: hedging shrinks the second term; the first term is what the firm pays for the
  hedge when p̂ is below the fair value mean_y P̄_y / Λ.

**Why the risk measure was changed from mean + γ·min.** With the earlier ρ = mean + γ·min, the
value of hedging was roughly linear in χ. Because Q̄ (112 GW) is below the hedging supply of
thermal plants (~135 GW), competition drove p̂ down until the last unit of hedge was worth
nothing, which with linear value meant the whole hedge was worth nothing: in the γ = 0.3 run,
CCGT paid ≈ $21k/MW-yr for a hedge that removed ≈ $24k of risk, and the SFPFC market (then R3) ≈ the merchant market (then R2). Under variance,
each extra unit of hedge is worth less than the one before, so competition only pushes p̂ down
to the (small) value of the last unit, and the earlier units keep their value.

## 7. The welfare score

    C_y = system cost in year y;     C = mean_y C_y                                 (welfare.py)

Consumers, who buy the contracts, are risk-neutral, so expected cost is the welfare measure
(`C_mean_$bn`; `C_max_$bn` is reported alongside). It is computed after solving and does not
affect K, p̂ or χ.

## 8. Code map

| item | file:line |
|---|---|
| ρ and its use in the contract choice | `eq_model/agents.py:47-83` |
| contract payoff s_zy | `eq_model/agents.py:129-130` |
| forward market clearing (bisection on p̂) | `eq_model/agents.py:133-223` |
| spot margins m_zy | `eq_model/agents.py:86-109` |
| dispatch LP V_y(K) | `eq_model/dispatch.py` |
| planner problem / condition (★) | `eq_model/equilibrium.py:237-330` |
| market search loop over c (`I_eff`) | `eq_model/equilibrium.py:409-525` |
| `risk_premium` | `eq_model/equilibrium.py:522` |
| `premium_pct_*` in summary.csv | `eq_model/welfare.py:134` |
| welfare score | `eq_model/welfare.py:62-68, 101` |
| existing (terser) write-up | `README.md` §4 |
