"""Sanity-check a downloaded results directory before using it.

    python scripts/check_results.py results/raw/default/g0.3
    python scripts/check_results.py results/raw/default        # every gamma at once

Scans recursively for row_*.json (so it works on a `gh run download` tree, a local run
directory, or a parent holding several), and reports what would silently ruin a figure:
a missing regime, a run that hit the iteration limit, or an unexpected gamma.
"""
from __future__ import annotations

import glob
import json
import os
import sys

EXPECTED = ["P1", "P2", "R2", "R3", "R4", "R5", "R6"]


def load_rows(dirs):
    rows = []
    for d in dirs:
        for fn in sorted(glob.glob(os.path.join(d, "**", "row_*.json"), recursive=True)):
            with open(fn) as fh:
                rows.append(json.load(fh))
    return rows


def check(dirs, verbose=True):
    """Report on a results tree.  Returns (rows, problems); problems empty means usable."""
    rows = load_rows(dirs)
    if not rows:
        return [], [f"no row_*.json found under {list(dirs)} - did the download finish?"]
    return rows, _report(rows, verbose)


def main(argv):
    if not argv:
        print(__doc__)
        return 2
    rows, problems = check(argv)
    if problems:
        print("PROBLEMS")
        for p in problems:
            print("  *", p)
        return 1
    print("OK: every expected regime present and converged at every gamma.")
    return 0


def _report(rows, verbose=True):
    gammas = sorted({r.get("gamma_requested", r.get("gamma")) for r in rows})
    if verbose:
        print(f"{len(rows)} result(s) across gamma = {', '.join(format(g, 'g') for g in gammas)}\n")

    hdr = f"{'gamma':>6} {'regime':>7} {'conv':>5} {'outer':>6} {'runtime_s':>10} {'completion':>11} {'C_mean_bn':>10}"
    if verbose:
        print(hdr)
        print("-" * len(hdr))
    bad = []
    for r in sorted(rows, key=lambda r: (r.get("gamma_requested", r.get("gamma")), r["regime"])):
        g = r.get("gamma_requested", r.get("gamma"))
        conv = bool(r.get("converged", False))
        cm = next((v for k, v in r.items() if k.startswith("C_mean")), float("nan"))
        comp = r.get("completion_s")
        comp_s = f"{comp:11.0f}" if isinstance(comp, (int, float)) else f"{'-':>11}"
        if verbose:
            print(f"{g:6g} {r['regime']:>7} {str(conv):>5} {r.get('n_evaluations', '-'):>6} "
                  f"{r.get('runtime_s', float('nan')):10.0f} {comp_s} {cm:10.3f}")
        if not conv:
            bad.append((g, r["regime"]))
    if verbose:
        print()

    problems = []
    for g in gammas:
        have = {r["regime"] for r in rows if r.get("gamma_requested", r.get("gamma")) == g}
        missing = [x for x in EXPECTED if x not in have]
        extra = sorted(have - set(EXPECTED))
        if missing:
            problems.append(f"gamma={g:g}: MISSING {', '.join(missing)} - a matrix job failed or "
                            "was not downloaded")
        if extra:
            problems.append(f"gamma={g:g}: unexpected regime(s) {', '.join(extra)}")
    if bad:
        problems.append("NOT CONVERGED: " + ", ".join(f"{r} at gamma={g:g}" for g, r in bad) +
                        " - these hit --max-outer and are not usable results")
    return problems


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
