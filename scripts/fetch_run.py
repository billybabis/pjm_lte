"""Fetch a finished GitHub Actions run, check it, and build its figures.

    python scripts/fetch_run.py                      # latest run, prompts for the scenario name
    python scripts/fetch_run.py --scenario default   # no prompt
    python scripts/fetch_run.py --run-id 12345678 --scenario high_voll

Downloads land in ``results/raw/<scenario>/g<gamma>/`` and figures in
``results/figs/<scenario>/g<gamma>_{capacity,energy,contract}.png`` (``--format pdf`` for
the paper).  The gamma is read from the
downloaded results themselves (``gamma_requested``), not guessed from the run, so the folder
name always matches what was actually solved.

A scenario is a set of --param overrides; give each one its own name.  Results are keyed by
(regime, gamma) with no scenario field, so two scenarios must never share a directory tree -
that is what the per-scenario level is for.

Requires the GitHub CLI (`winget install --id GitHub.cli`, then `gh auth login`).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import check_results  # noqa: E402

WORKFLOW = "run-model.yml"
RAW_DIR = os.path.join("results", "raw")     # downloaded artifacts, one dir per scenario/gamma
FIG_DIR = os.path.join("results", "figs")    # figures, one dir per scenario


def _run(cmd, **kw):
    return subprocess.run(cmd, check=True, text=True, cwd=ROOT, **kw)


def _gh_json(args):
    out = subprocess.run(["gh", *args], check=True, text=True, capture_output=True, cwd=ROOT).stdout
    return json.loads(out)


def _replace_dir(dest):
    """Clear ``dest``, tolerating Windows/OneDrive file locks.

    OneDrive holds handles on files while it syncs, so a recursive delete fails intermittently
    with PermissionError.  Retry briefly, then fall back to renaming the old directory aside
    rather than failing the fetch - nothing is ever lost, and the move can be cleaned up later.
    """
    def _clear_readonly(func, path, _exc):
        os.chmod(path, stat.S_IWRITE)
        func(path)

    for attempt in range(4):
        try:
            shutil.rmtree(dest, onerror=_clear_readonly)
            return
        except (PermissionError, OSError):
            time.sleep(0.5 * (attempt + 1))
    aside = f"{dest}.old-{time.strftime('%Y%m%d-%H%M%S')}"
    try:
        os.rename(dest, aside)
        print(f"  could not delete {os.path.basename(dest)} (file lock - OneDrive?); moved it to "
              f"{os.path.basename(aside)}")
    except OSError as e:
        raise SystemExit(
            f"{dest} exists and cannot be removed or renamed ({e}). "
            "Close anything using it (Explorer, an editor, OneDrive sync) and retry, "
            "or delete it by hand.")


def require_gh():
    if shutil.which("gh") is None:
        raise SystemExit("the GitHub CLI is not installed or not on PATH.\n"
                         "  winget install --id GitHub.cli\n"
                         "then open a NEW terminal and run:  gh auth login")


def latest_run(workflow=WORKFLOW):
    runs = _gh_json(["run", "list", "--workflow", workflow, "--limit", "10",
                     "--json", "databaseId,status,conclusion,createdAt,displayTitle"])
    if not runs:
        raise SystemExit(f"no runs found for {workflow}")
    done = [r for r in runs if r["status"] == "completed"]
    if not done:
        raise SystemExit(f"the most recent {workflow} run is still {runs[0]['status']}; wait for it "
                         "to finish (gh run watch)")
    return done[0]


def gamma_of(rows):
    gs = sorted({r.get("gamma_requested", r.get("gamma")) for r in rows})
    if len(gs) != 1:
        raise SystemExit(f"this run contains several gammas ({', '.join(format(g, 'g') for g in gs)}); "
                         "fetch_run expects one dispatch = one gamma. Sort it manually, or fetch a "
                         "run that did not also sweep.")
    return gs[0]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", help="name under results/raw/ and results/figs/ (e.g. default, high_voll)")
    ap.add_argument("--run-id", help="which run to fetch (default: the most recent completed one)")
    ap.add_argument("--workflow", default=WORKFLOW)
    ap.add_argument("--panel-width", type=float, default=4.5, help="inches per gamma panel")
    ap.add_argument("--format", default="png", choices=["png", "pdf", "svg"],
                    help="png by default (quick to eyeball); use pdf for the paper")
    ap.add_argument("--no-figures", action="store_true", help="download and check only")
    ap.add_argument("--force-figures", action="store_true",
                    help="build figures even if the check found problems")
    ap.add_argument("--overwrite", action="store_true", help="replace an existing target directory")
    args = ap.parse_args(argv)

    require_gh()

    if args.run_id:
        run = _gh_json(["run", "view", args.run_id, "--json",
                        "databaseId,status,conclusion,createdAt,displayTitle"])
    else:
        run = latest_run(args.workflow)
    print(f"run {run['databaseId']}  {run['createdAt']}  {run['conclusion']}  {run['displayTitle']}")
    if run["conclusion"] not in ("success", "neutral"):
        print(f"  NOTE: this run's conclusion is {run['conclusion']!r} - some matrix jobs may be "
              "missing. The check below will say which.")

    scenario = args.scenario
    while not scenario:
        scenario = input("scenario folder name (e.g. default, high_voll): ").strip()
    scenario = scenario.replace(" ", "_")

    # Download to a staging directory first: the gamma, and therefore the final path, is only
    # known once the results are on disk.
    stage = tempfile.mkdtemp(prefix="eqrun_")
    try:
        print(f"downloading artifacts for run {run['databaseId']} ...")
        _run(["gh", "run", "download", str(run["databaseId"]), "-D", stage])
        rows = check_results.load_rows([stage])
        if not rows:
            raise SystemExit("downloaded, but no row_*.json found - did the regime jobs fail? "
                             f"(staged at {stage})")
        gamma = gamma_of(rows)

        dest = os.path.join(ROOT, RAW_DIR, scenario, f"g{gamma:g}")
        if os.path.exists(dest):
            if not args.overwrite:
                ans = input(f"{os.path.relpath(dest, ROOT)} already exists - overwrite? [y/N] ")
                if ans.strip().lower() not in ("y", "yes"):
                    raise SystemExit(f"stopping; the download is staged at {stage}")
            _replace_dir(dest)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.move(stage, dest)
        stage = None
        print(f"-> {os.path.relpath(dest, ROOT)}")
    finally:
        if stage and os.path.isdir(stage):
            shutil.rmtree(stage, ignore_errors=True)

    print()
    _, problems = check_results.check([dest])
    if problems:
        print("PROBLEMS")
        for p in problems:
            print("  *", p)
    else:
        print("OK: every expected regime present and converged.")

    if args.no_figures:
        return 1 if problems else 0
    if problems and not args.force_figures:
        print("\nnot building figures from an incomplete or non-converged run.\n"
              "  re-dispatch the failed regimes, or pass --force-figures to plot anyway.")
        return 1

    figdir = os.path.join(ROOT, FIG_DIR, scenario)
    os.makedirs(figdir, exist_ok=True)
    out = os.path.join(figdir, f"g{gamma:g}.{args.format}")
    print()
    _run([sys.executable, "-m", "eq_model", "plot", dest, "--kind", "all",
          "--panel-width", str(args.panel_width), "--out", out])

    print(f"\nfigures in {os.path.relpath(figdir, ROOT)}/")
    print("\nonce every gamma is fetched, the multi-panel figure is:")
    raw = RAW_DIR.replace(os.sep, "/")
    figs = FIG_DIR.replace(os.sep, "/")
    print(f"  python -m eq_model plot {raw}/{scenario} --kind all "
          f"--panel-width {args.panel_width:g} --out {figs}/{scenario}/all.{args.format}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
