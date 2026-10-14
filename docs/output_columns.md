# What each column of `summary.csv` means

Each row of `summary.csv` is one solved regime (SP, DM, DM-FC, DM-FC-tau, DM-tau) at one γ. The row is built in
`eq_model/welfare.py:113-145` (`summary_row`); the last three columns are added in
`eq_model/cli.py:203-208`.

Notation follows `the-functional-form-of-groovy-pixel.md` (the model write-up):
y = year (Y = 9), h = hour, z = technology, K_z = capacity, I_z = annualised capital + fixed O&M
cost, p_hy = spot price, D_hy = load, l_hy = lost load.

`{z}` below stands for each technology: nuclear, ccgt, ct, coal, solar, wind, storage4, storage8.
"Mean" always means the simple average over the 9 years.

---

## 1. Identification

| column | meaning |
|---|---|
| `regime` | SP = social planner (carbon priced, reliability cost internalised); DM = market; DM-FC = market with SFPFCs; DM-FC-tau = SFPFCs + carbon price; DM-tau = carbon price, no SFPFCs. Older results use P1, P2, R2–R6. |
| `gamma` | γ the regime was **actually solved with**. Always 0 for SP (the planner is risk-neutral). |
| `gamma_requested` | γ the run was launched with. Same as `gamma` for the market regimes. For SP it only says which run the row belongs to; it was not used. |
| `tau_market` | carbon price faced by firms in the market, $/tCO₂ (0 or the SCC, $280) |
| `converged` | True if the solver met its tolerance. For R-regimes: every built technology has \|ρ*\| ≤ tol·I. False = the row is the best point found, not a confirmed equilibrium. |

## 2. System cost (the welfare measure)

For each year, `C_y` = fuel & variable O&M + carbon cost + VOLL·lost load + Σ_z I_z K_z + reliability cost Ψ
(`welfare.py:62-68`). All cost columns are in **$ billion per year**.

| column | formula / meaning |
|---|---|
| `C_mean_$bn` | mean_y C_y. **The main efficiency comparison across regimes.** |
| `C_max_$bn` | max_y C_y: cost in the most expensive year |
| `cost_variable_om_fuel_$bn` | mean_y Σ_h Σ_thermal c_z q_zhy |
| `cost_carbon_at_tau_welfare_$bn` | mean_y Σ_h Σ_thermal e_z τ_w q_zhy, where τ_w = SCC in **every** regime by default (`welfare_tau_always_scc=True`), so unpriced emissions in DM/DM-FC still count as a social cost |
| `cost_lost_load_$bn` | mean_y VOLL · Σ_h l_hy |
| `cost_capex_annualized_$bn` | Σ_z I_z K_z, using the **true** I_z (not the solver's search variable `I_eff`) |
| `cost_reliability_psi_$bn` | mean_y Σ Ψ_ry: reserve cost caused by VRE (wind, solar) |

Consumers (the contract buyers) are risk-neutral in the model, so `C_mean_$bn` is the welfare
measure. Results produced before the switch to mean–variance also contain `C_risk_adj_$bn`
(mean_y C_y + γ·max_y C_y); that column has been removed.

## 3. Emissions, reliability, prices

| column | meaning | units |
|---|---|---|
| `emissions_Mt_mean` | mean_y Σ_h Σ_thermal e_z q_zhy | MtCO₂/yr |
| `lost_load_GWh_mean` | mean_y Σ_h l_hy | GWh/yr |
| `lost_load_hours_mean` | mean_y (number of hours with l_hy > 0) | hours/yr |
| `hours_at_VOLL` | number of hours, **summed over all 9 years**, where the price equals VOLL (not a per-year mean) | hours |
| `price_mean` | simple mean of p_hy over all hours and years | $/MWh |
| `price_load_wtd` | Σ p_hy D_hy / Σ D_hy: average price paid per MWh consumed | $/MWh |

## 4. Forward market (SFPFC)

| column | meaning | units |
|---|---|---|
| `p_hat` | SFPFC price p̂ that clears the forward market. Empty for regimes without contracts (SP, DM, DM-tau). For comparison, the "fair" price is mean_y P̄_y / Λ (≈ $46 in the earlier τ=280, γ=0.3 run with the min risk measure, where `p_hat` = 43.2). | $/MWh |
| `Q_bar_MW` | Q̄: total contracted capacity the market must clear. 0 when there are no contracts. | MW |
| `chi_{z}` | χ_z: share of z's capacity that is contracted, in [0, 1]. 0 in regimes without contracts. | – |

## 5. Capacity and generation mix

| column | meaning | units |
|---|---|---|
| `K_{z}_MW` | installed capacity K_z (tiny negatives like −0.0 are solver round-off; read as 0) | MW |
| `share_{z}` | z's total generation / total load, over all hours and years. Storage shares are slightly **negative**: storage loses energy, so its net output is below zero. | – |
| `share_lost_load` | total lost load / total load | – |

## 6. Curtailment (wind and solar only)

Curtailed energy in an hour = available VRE output (θ_zhy K_z) minus what was dispatched. An hour
"counts" as curtailed if more than 1 MW is spilled (`curtailment_hour_threshold_mw`).

| column | meaning | units |
|---|---|---|
| `curt_hours_any_mean` | mean_y number of hours where solar **or** wind is curtailed | hours/yr |
| `curt_share_vre` | curtailed / available energy, solar and wind pooled | – |
| `curt_hours_{solar,wind}_mean` | mean_y number of curtailed hours for that technology | hours/yr |
| `curt_share_{solar,wind}` | 1 − dispatched / available energy for that technology | – |
| `curt_GWh_{solar,wind}_mean` | mean_y curtailed energy | GWh/yr |

## 7. Risk diagnostics

| column | meaning | units |
|---|---|---|
| `premium_pct_{z}` | 100 · (mean_y m_zy − I_z) / I_z: how far z's **average spot margin** exceeds its cost in equilibrium. Computed after solving; not used in any calculation. 0 in planner rows by construction; empty if K_z ≤ 1 MW. See §6 of the model write-up for how it decomposes into contract cost and risk compensation. | % |
| `gamma_{z}` | γ_z used by technology z. **Only present** when `gamma_scaling` ≠ "none" (e.g. "capital", where γ_z = γ × overnight capital cost of z / that of CCGT). | – |

## 8. Run bookkeeping

| column | meaning | units |
|---|---|---|
| `runtime_s` | wall-clock time to solve this regime | s |
| `completion_s` | part of the runtime spent on the single-process "price completion" LP (recovers exact scarcity prices; extra cores don't speed it up). 0 for SP. | s |
| `n_evaluations` | number of times the 9 yearly dispatch LPs were solved, summed over the market search. 0 for SP. | count |
