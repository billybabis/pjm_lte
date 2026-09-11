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
| `params` | *(empty)* | scenario overrides, space separated: `voll=20000 markdown_mode=full` |
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

## 4. Build the multi-γ figure

Artifact names repeat across dispatches, so **download each run into its own directory** or the
later ones overwrite the earlier:

```
gh run list --workflow=run-model.yml
gh run download <id-for-0>   -D results/g0
gh run download <id-for-0.3> -D results/g0.3
gh run download <id-for-0.6> -D results/g0.6
gh run download <id-for-1>   -D results/g1

python -m eq_model plot results/ --kind both --gammas 0,0.3,0.6,1 \
    --panel-width 4.5 --out figs/fig.png
```

That writes `figs/fig_capacity.png` and `figs/fig_energy.png` (use `.pdf` for LaTeX). The
single-γ figure inside each `summary` artifact is not the same thing — the multi-panel version
only comes from this step.

`--panel-width 4.5` because seven regimes per panel is cramped at the 3.4-inch default.

## Scenarios

A scenario is a set of `params` overrides. Give each its own directory tree and run `plot` once per
scenario — results are keyed by `(regime, γ)`, which does not include the scenario, so two of them
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
