# Risk aversion (γ): specification, calibration check, and limitations

Companion to `docs/model_choices.md` (formulas) and `docs/model_overview.md` (method).
All numbers below are from the full-data runs at γ = 1.25 with capital scaling
(`results/raw/variance/g1.25/`; regimes DM = merchant market, DM-FC = market with SFPFCs, run
under their former names R2 and R3). Carbon is not priced in either. Those runs used the earlier
annualised-capital scaling; switching to overnight capital changes γ only for nuclear and storage,
which are unprofitable and unbuilt in both regimes under either scaling, so the equilibria are
unchanged. Their hypothetical rows below use the new γ values.

---

## 1. The specification

Each firm evaluates its per-MW annual profit across the Y = 9 weather years with a
**mean–variance** criterion:

    ρ_z = mean_y π_zy − γ_z · Var_y(π_zy)
    π_zy = spot margin_zy − I_z + χ_z · (SFPFC payoff)_zy

- **What varies:** the 9 annual profit totals. Hourly variation enters only through each year's
  total. Investors and lenders care about annual cash flow, and much hourly variation (day/night,
  seasons) is predictable rather than risky.
- **Units:** profit in **$M per MW-yr** inside the variance (`agents._RISK_UNIT`), so γ is of
  order 1.
- **Level:** γ = **1.25** for CCGT, the reference technology. Chosen because it reproduces the ≈25%
  CCGT merchant premium of the earlier specification (mean + 0.3·min). **It is not yet calibrated
  to an external target** (see §5).
- **Capital scaling** (`gamma_scaling=capital`): each technology's γ is scaled by its overnight
  capital cost per MW relative to CCGT, i.e. risk aversion proportional to the dollars at stake,

      γ_z = γ × OCC_z / OCC_ccgt

  | | nuclear | coal | storage8 | storage4 | wind | CCGT | CT | solar |
  |---|---|---|---|---|---|---|---|---|
  | γ_z | 6.08 | 3.24 | 2.41 | 1.38 | 1.30 | 1.25 | 1.10 | 1.10 |

  Overnight rather than annualised capital is used so that a long asset life does not dilute the
  scaling: with annualised capital, nuclear's 60-year life cut its γ to 3.3× CCGT's instead of
  4.9×, and its financing spread fell below CCGT's despite four to five times the capital at stake.

  **Rationale:** revenue risk raises financing costs mainly by limiting how much cheap debt a
  project can raise. Lenders size debt on downside revenue (debt-service coverage). That penalty
  is felt in proportion to the capital being financed, so the same revenue volatility costs a
  capital-intensive plant more. This is the channel through which long-term contracts are argued
  to matter most for nuclear.

## 2. How a premium becomes a financing cost

In equilibrium a built technology's average spot margin must cover I_z plus a **risk charge**
γ′_z·Var(π) (γ′ = γ/10⁶) plus, under SFPFCs, the **average cost of its contracts** −χ·mean(s). The
model converts that requirement into an **implied real discount rate** r′:

    OCC_z · CRF(r′, life_z) + FOM_z = I_z + risk charge + contract cost

r′ is the cost of capital at which a risk-neutral investor would make the same decision. Compare it
with the model's risk-free real rate r = 2.5%. The gap is the **financing spread**.

## 3. What the model implies

Per MW, at each regime's equilibrium prices. Nuclear, coal and storage are **not built** in either
regime; their rows show what 1 MW would face at those prices.

| | γ_z | SD of annual profit, merchant ($k/MW) | **merchant (DM)**: risk charge, $k/MW-yr (% of I) | implied real rate | spread | **with SFPFC (DM-FC)**: χ | risk charge + contract cost, $k | implied real rate | spread |
|---|---|---|---|---|---|---|---|---|---|
| CCGT | 1.25 | 148 | 27.4 (28.7%) | 5.53% | 303 bp | 0.82 | 0.7 + 5.8 | 3.27% | 77 bp |
| CT | 1.10 | 145 | 23.2 (28.5%) | 5.42% | 292 bp | 0.78 | 0.8 + 5.4 | 3.34% | 84 bp |
| solar | 1.10 | 45 | 2.2 (2.9%) | 2.80% | 30 bp | 0.14 | 0.9 + 1.0 | 2.77% | 27 bp |
| wind | 1.30 | 12 | 0.2 (0.2%) | 2.52% | 2 bp | 0.00 | 0.2 + 0.0 | 2.52% | 2 bp |
| nuclear* | 6.08 | 148 | 133.3 (33.8%) | 5.09% | 259 bp | 0.98 | 0.1 + 6.9 | 2.65% | 15 bp |
| coal* | 3.24 | 148 | 71.0 (28.3%) | 5.53% | 303 bp | 0.95 | 0.3 + 6.6 | 2.82% | 32 bp |
| storage4* | 1.38 | 104 | 14.9 (9.9%) | 4.21% | 171 bp | 0.49 | 1.1 + 3.4 | 3.04% | 54 bp |
| storage8* | 2.41 | 139 | 46.9 (17.8%) | 5.50% | 300 bp | 0.88 | 0.4 + 6.1 | 2.94% | 44 bp |

\* not built; hypothetical 1 MW at the regime's prices.

Reading the table:
- **Thermal and storage revenue is risky and almost perfectly hedgeable.** Their margins come from
  the same scarcity hours that drive the SFPFC's settlement, so the hedge nearly eliminates the
  risk charge. What remains under SFPFCs is mostly the **contract cost**: p̂ = $43.1 vs a fair
  $44.0/MWh, because Q̄ is below what generators want to hedge.
- **SFPFCs lower thermal financing costs by about 220 bp** (CCGT 5.5% → 3.3% real).
- **Nuclear's saving is the largest, in dollars and in basis points.** SFPFCs cut its required
  revenue by ≈ $126k/MW-yr, about 6× CCGT's ≈ $21k, and its financing cost by ≈ 244 bp vs
  CCGT's ≈ 226 bp. This is the capital-scaling mechanism at work: the same revenue risk weighs
  more on the technology with the most capital at stake, and SFPFCs remove it.
- **Wind and solar carry almost no risk charge**, merchant or not (see §5).

## 4. Comparison with observed financing costs

The model's rates are **real, pre-tax, whole-project (WACC)** rates. Quoted figures are usually
nominal, often after-tax, and sometimes equity returns, which are higher than WACC because debt is
cheaper. To compare, add expected inflation of ≈ 2–2.5 percentage points.

Benchmarks below are approximate and from memory; **verify against the sources before citing**
(e.g. the latest PJM / Brattle Cost of New Entry review for merchant gas; NREL ATB and Lazard LCOE
for renewables; UK CfD / RAB documentation for nuclear).

| | model, real | ≈ nominal | rough real-world nominal WACC | assessment |
|---|---|---|---|---|
| merchant CCGT / CT | 5.4–5.5% | ≈ 7.5–8% | ≈ 7–8% (PJM CONE-type assumptions) | **consistent** |
| contracted gas | 3.3% | ≈ 5.5% | ≈ 5–6% for contracted / regulated generation | consistent |
| merchant wind / solar | 2.5–2.8% | ≈ 5% | ≈ 7–9% merchant | **too low by ≈ 250–400 bp** |
| contracted wind / solar | 2.5–2.8% | ≈ 5% | ≈ 5–6% with a PPA | consistent |
| merchant nuclear | 5.1% (if built) | ≈ 7.5% | widely considered unfinanceable without support; high single to low double digits | **too low** |
| contracted nuclear | 2.65% | ≈ 5% | substantially lower under CfD / RAB-type support | direction right, level uncertain |

## 5. Shortcomings

1. **Only one source of risk: weather-year variation at a known capacity mix.** The model omits
   fuel-price (gas) risk, demand growth, policy risk, construction-cost risk and uncertainty about
   others' future entry (cannibalisation risk). Real merchant spreads reflect mostly these.
2. **Renewable risk is understated.** Wind's annual revenue barely varies across the 9 weather
   years (SD ≈ $12k/MW vs ≈ $148k for thermal), so no reasonable γ gives it a material premium.
   Real merchant renewables face price-level and cannibalisation risk the model doesn't simulate.
3. **Nuclear's merchant risk is understated.** Its dominant real risks (construction cost and
   schedule, long-horizon price risk) are absent, so its merchant rate (≈ 5.1% real) is well below
   what merchant nuclear would actually face.
4. **Nine observations, two of which dominate.** The contract-shaped average price P̄_y/Λ is
   $33–48/MWh in most years but $65 and $88 in two scarcity-heavy years. Much of every thermal
   plant's variance, and so its premium, comes from those two years.
5. **Capital scaling is a reduced form.** The model doesn't simulate debt. Scaling γ with capital
   intensity stands in for the debt-sizing channel; it is a modelling assumption, not derived.
6. **γ is not yet calibrated to an external target.** 1.25 was chosen for continuity with the
   earlier specification. It happens to give a merchant CCGT spread (≈ 300 bp real) in the
   plausible range, but that should be pinned to a cited number.
7. **Basis differences.** Real vs nominal, no taxes, WACC vs equity: comparisons with quoted
   figures need the conversions in §4.

## 6. Implications for the results

- **Reliability and thermal investment results are on the firmest ground.** Thermal premiums and
  their reduction under SFPFCs are in a plausible range, and lost load is driven by peaking
  capacity (DM 33.6 → DM-FC 26.6 GWh/yr vs a planner's ≈ 22).
- **The merchant-market renewable build-out is overstated.** With renewables nearly riskless and
  thermal carrying a ≈ 29% premium, DM builds ≈ 53 GW of wind (vs ≈ 27–34 GW elsewhere). With no
  carbon price, that lowers emissions and makes DM look better than DM-FC on SCC-valued cost
  (`C_mean` 95.0 vs 100.2 $bn): risk aversion acts as an accidental carbon tax. With realistic
  renewable risk this effect would shrink. Treat it as an upper bound.
- **Who benefits from SFPFCs is skewed toward thermal and storage.** In the model SFPFCs do little
  for wind and solar because there is little renewable risk to remove. In reality, cheaper
  renewable financing is a main argument for long-term contracts. The model likely understates
  SFPFCs' value for renewables.
- **The nuclear mechanism depends on capital scaling.** Under capital scaling SFPFCs save nuclear
  ≈ 6× what they save CCGT per MW. Under a uniform γ (1.25 for nuclear instead of 6.08), nuclear's
  merchant risk charge would be ≈ 7% of cost instead of ≈ 34%, and the disproportionate benefit
  would largely disappear.
  Nuclear is built only when carbon is priced, so the DM-tau vs DM-FC-tau comparison is where this
  shows up.
- **Results may be sensitive to which weather years are included**, given that two years drive
  most of the variance.

## 7. Recommended robustness checks

1. **Calibrate γ** to a cited merchant gas spread with `scripts/calibrate_gamma.py`.
2. **γ sweep** at about ½×, 1× and 2× the calibrated value, to show how SFPFC effects scale with
   risk aversion.
3. **Uniform γ** (`gamma_scaling=none`) to show how much the nuclear result depends on capital
   scaling.
4. **Leave-one-year-out** (`--years`), especially dropping each of the two scarcity-heavy years.
5. **Optional extension:** gas-price scenarios as additional states of the world, which would give
   renewables a realistic, hedgeable price-level risk under the same uniform γ.
