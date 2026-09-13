# Running the model on GitHub Actions

The `run-model` workflow (`.github/workflows/run-model.yml`) builds the panel once, then solves
**one regime per parallel job**. A full seven-regime run takes about as long as its slowest single
regime instead of the sum of all seven.

One dispatch = one γ. The four-panel paper figure needs four dispatches, then one local command.

---

## 1. Start a run

**Actions** tab → **run-model** in the left sidebar → **Run workflow** (top right). A form appears:

Every field is a free-text box. The values below are only what GitHub prefills — overwrite any of
them before clicking Run (e.g. type `0.6` into `gamma`).

| field | prefilled with | what it does |
|---|---|---|
| `regimes` | `P1,P2,R2,R3,R4,R5,R6` | one matrix job per entry |
| `gamma` | `0.3` | risk aversion for this dispatch |
| `workers` | `3` | parallel yearly-dispatch processes inside each job (runner has 4 vCPU) |
| `tau_scc` | `280` | carbon price $/tCO2 (2025$) |
| `params` | *(empty)* | any other overrides, space separated: `voll=20000 markdown_mode=full` |
| `sweep_gammas` | *(empty)* | if set, also sweeps one regime across these γ; leave empty to skip |
| `sweep_regime` | `R2` | which regime the sweep uses |

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

Actions → the run. Jobs are listed as `regime (P1)`, `regime (P2)`, … Each streams its solver log
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
python scripts/fetch_run.py --scenario default
```

It takes the most recent completed run, downloads it, reads the γ from the results themselves,
and files everything as:

```
results/raw/default/g0.3/      <- the artifacts
results/figs/default/g0.3_capacity.png
results/figs/default/g0.3_energy.png
results/figs/default/g0.3_contract.png
```

Only the **scenario** name is yours to choose; the `g<gamma>` level is derived. Artifact names
repeat across dispatches, which is exactly why each run needs its own directory — the script
handles that for you.

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

`--panel-width 4.5` because seven regimes per panel is cramped at the 3.4-inch default.

To do it by hand instead (no `gh`), download the artifacts from the run page and unzip them under
`results/raw/<scenario>/g<gamma>/`, then run `python scripts/check_results.py` on that directory
before plotting.

`tau_scc` enters the agents' own costs only in the carbon-priced regimes (P2, R5, R6). It also
values emissions in *every* regime's welfare metric — that is deliberate, and is what makes P1 and
P2 comparable — so editing it moves the welfare numbers for all seven regimes, not just three.

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
  cleanly just under GitHub's 360-minute kill. The τ = SCC regimes (R5, R6) are the ones to watch.
* **The `push:` trigger** starts a run at the default γ whenever `eq_model/**` changes on `main`.
  Delete that block if every run should be deliberate.
* **`concurrency: cancel-in-progress`** means a push during a long run cancels it. Set to `false`
  to let long runs finish.
* **`--jobs` is for local runs.** Inside a CI job leave it at 1: each process holds ~3 GB and the
  runner has 16 GB. Parallelism in CI comes from the matrix, not from `--jobs`.
