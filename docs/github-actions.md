# Running the model on GitHub Actions

The `run-model` workflow (`.github/workflows/run-model.yml`) builds the panel once, then solves
**one regime per parallel job**. A full five-regime run takes about as long as its slowest single
regime instead of the sum of all five.

One dispatch = one γ. The four-panel paper figure needs four dispatches, then one local command.

---

## 1. Start a run

**Actions** tab → **run-model** in the left sidebar → **Run workflow** (top right). A form appears:

Every field except `gamma_scaling` is a free-text box. The values below are only what GitHub
prefills — overwrite any of them before clicking Run (e.g. type `0.6` into `gamma`).

| field | prefilled with | what it does |
|---|---|---|
| `run_name` | *(required)* | name for the run: its title in the Actions list and its results folder (`results/raw/<run_name>/`) |
| `regimes` | `SP,DM,DM-FC,DM-FC-tau,DM-tau` | one matrix job per entry |
| `gamma` | `1.25` | risk aversion for this dispatch |
| `gamma_scaling` | `capital` | drop-down: `capital` scales each technology's γ by its overnight capital cost relative to CCGT; `none` gives every technology the same γ |
| `workers` | `3` | parallel yearly-dispatch processes inside each job (runner has 4 vCPU) |
| `tau_scc` | `280` | carbon price $/tCO2 (2025$) |
| `params` | *(empty)* | any other overrides, space separated: `voll=20000 markdown_mode=full` |
| `sweep_gammas` | *(empty)* | if set, also sweeps one regime across these γ; leave empty to skip |
| `sweep_regime` | `DM` | which regime the sweep uses |

No commit or YAML edit is needed — the form exists because the workflow declares
`workflow_dispatch`, and the values apply to that dispatch only. To produce the four-panel figure,
dispatch four times, typing `0`, `0.3`, `0.6` and `1` into `gamma`.

The same thing from a terminal, if you prefer it to the browser — identical in every respect:

```
gh workflow run run-model.yml -f gamma=0.6
gh workflow run run-model.yml -f gamma=0.3 -f params="voll=20000"
```

The Run workflow button appears only once the workflow file is on the default branch.

## 2. Watch it

Actions → the run. Jobs are listed as `regime (SP)`, `regime (DM)`, … Each streams its solver log
and prints a one-row summary at the end. `gh run watch` does the same from a terminal.

`fail-fast` is off, so one diverging regime does not cancel the others, and the `combine` job still
produces a summary from whatever finished.

## 3. Collect the results

Each run's **Artifacts** box holds:

| artifact | contents | retention |
|---|---|---|
| `summary` | `summary.csv`, `capacity.pdf`/`.png` (single-γ), `sweep.csv` if swept | 90 days |
| `results-<REGIME>` | that regime's `result_*.json`, `profits_*.csv`, `prices_*.npz`, `row_*.json` | 30 days |
| `panel` | `panel.npz`, `vre_capacity_with_coverage.csv` | 7 days |

The merged table is also rendered on the run's summary page, so a quick look needs no download.

Retention is a real deadline: download anything you intend to cite.

## 4. Collect a run and build its figures

`scripts/fetch_run.py` does the whole download-check-plot step. Run it once per dispatch:

```
python scripts/fetch_run.py                      # most recent completed run
python scripts/fetch_run.py --run-id 12345678    # a specific run
```

It downloads the run, reads the γ from the results themselves, and files everything under the
`run_name` typed into the form:

```
results/raw/baseline/g1.25/      <- the artifacts
results/figs/baseline/g1.25_capacity.png
results/figs/baseline/g1.25_energy.png
results/figs/baseline/g1.25_contract.png
```

**Nothing is overwritten.** If `results/raw/<run_name>/g<gamma>` already exists, the run is filed
as `<run_name>1` (then `<run_name>2`, …) and the script says so. Pass `--overwrite` to replace the
existing folder instead, or `--scenario <name>` to file it under a different name. Runs from before
the `run_name` field (and push-triggered runs) have no name, so the script asks for one.

Figures are PNG by default, which is what you want for looking at them. Pass `--format pdf`
(or use a `.pdf` extension on `--out` for the plot command) for the vector version to drop
into LaTeX.

It refuses to build figures from a run with a missing or non-converged regime (`--force-figures`
overrides). `--run-id <id>` fetches a specific run, `--no-figures` downloads and checks only.

Once every γ is collected, the multi-panel figure comes from one command:

```
python -m eq_model plot results/raw/default --kind all --gammas 0,0.3,0.6,1     --panel-width 4.5 --out results/figs/default/all.png
```

That writes `all_capacity.png`, `all_energy.png` and `all_contract.png` under
`results/figs/default/`. The single-γ figure inside each `summary` artifact is not the same
thing — the multi-panel version only comes from this step.

`--panel-width 4.5` because five regimes per panel is cramped at the 3.4-inch default.

To do it by hand instead (no `gh`), download the artifacts from the run page and unzip them under
`results/raw/<scenario>/g<gamma>/`, then run `python scripts/check_results.py` on that directory
before plotting.

`tau_scc` enters the agents' own costs only in the carbon-priced regimes (SP, DM-FC-tau, DM-tau). It also
values emissions in *every* regime's welfare metric — that is deliberate, and is what makes the
regimes comparable — so editing it moves the welfare numbers for all five regimes, not just three.

Risk aversion scaled by capital cost (nuclear ≈ 3.3× the γ you enter, CCGT exactly γ) is a
`params` entry: type `gamma_scaling=capital`. Fetch it under its own name, e.g.
`python scripts/fetch_run.py --scenario gcap`.

To set the γ level, dispatch that scenario at several γ (e.g. 0, 0.3, 0.6, 0.9), fetch each, then:

```
python scripts/calibrate_gamma.py results/raw/gcap --target-spread-bp 250
```

It prints each technology's implied financing spread over the risk-free rate, in bp, by γ and regime.
It then interpolates the γ at which merchant CCGT (DM) matches your cited spread, and refuses if the
target lies outside the grid. Dispatch that γ as the headline run.

Changing `tau_scc` (or anything in `params`) makes it a different **scenario** — give it its own
scenario name when you fetch it, or results at the same (regime, gamma) will collide. See below.

## Scenarios

A scenario is a set of `params` overrides. Give each its own name under `results/raw/` and run
`plot` once per scenario — results are keyed by `(regime, γ)`, which does not include the scenario, so two of them
cannot share one figure. A command pointed at a directory spanning two scenarios is refused, naming
both files and the first differing field. Every `result_<R>.json` records the full parameter set it
was produced with.

## Notes

* **Cost.** Standard runners on a public repository are free and unmetered. Nothing in this
  workflow uses paid infrastructure, and no payment method is required.
* **The 6-hour job limit** is the binding constraint. Jobs are capped at 350 minutes so they fail
  cleanly just under GitHub's 360-minute kill. The τ = SCC market regimes (DM-FC-tau, DM-tau) are the ones to watch.
* **The `push:` trigger** starts a run at the default γ whenever `eq_model/**` changes on `main`.
  Delete that block if every run should be deliberate.
* **Several runs at once.** Each manual dispatch runs independently, so you can launch several
  named runs back to back. Only push-triggered runs cancel an earlier push run on the same branch.
  The free tier allows about 20 concurrent jobs per account; with 5 regimes per run, a fifth
  simultaneous run waits in the queue (each job's own run time is unaffected).
* **`--jobs` is for local runs.** Inside a CI job leave it at 1: each process holds ~3 GB and the
  runner has 16 GB. Parallelism in CI comes from the matrix, not from `--jobs`.
