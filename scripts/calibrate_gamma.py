"""Tabulate implied financing spreads by gamma, and calibrate the gamma level to a cited spread.

    python scripts/calibrate_gamma.py results/raw/gcap
    python scripts/calibrate_gamma.py results/raw/gcap --target-spread-bp 250

The model turns each technology's equilibrium risk premium into an equivalent real discount rate
(``implied_discount_rate`` in result_<R>.json): the rate r' at which OCC*CRF(r', L) + FOM equals the
risk-loaded capital cost.  Its excess over the model's risk-free rate r is the financing spread that
the risk-averse market implies.  This script lays those spreads out as a gamma x technology table per
market regime, and, given a cited target spread, interpolates the gamma at which the calibration
technology (default: CCGT in the unreformed merchant market R2) reproduces it.

gamma is the value each run was launched with (``gamma_requested``).  Under gamma_scaling=capital it
is the reference technology's gamma, from which every other technology's follows.  Read-only.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys

TECH_ORDER = ["nuclear", "coal", "ccgt", "ct", "solar", "wind", "storage4", "storage8"]


def load(dirs):
    """{(regime, gamma): {"spread_bp": {tech: bp}, "converged": bool, "scaling": str, "path": str}}
    for every market-regime result under ``dirs``.  Planners are risk-neutral and carry no spread."""
    out = {}
    for d in dirs:
        for row_fn in sorted(glob.glob(os.path.join(d, "**", "row_*.json"), recursive=True)):
            res_fn = os.path.join(os.path.dirname(row_fn), os.path.basename(row_fn).replace("row_", "result_", 1))
            if not os.path.exists(res_fn):
                continue
            with open(row_fn) as fh:
                row = json.load(fh)
            with open(res_fn) as fh:
                res = json.load(fh)
            if res.get("regime", {}).get("planner"):
                continue
            params = res.get("params") or {}
            r = params.get("r", 0.025)
            g = float(row.get("gamma_requested", row.get("gamma")))
            spreads = {}
            for z, w in (res.get("implied_discount_rate") or {}).items():
                if w is None or res["K"].get(z, 0.0) <= 1.0 or (isinstance(w, float) and math.isnan(w)):
                    continue
                spreads[z] = (w - r) * 1e4
            key = (row["regime"], g)
            if key in out:
                print(f"warning: {key[0]} at gamma={g:g} found twice; keeping {out[key]['path']}", file=sys.stderr)
                continue
            out[key] = {"spread_bp": spreads, "converged": bool(row.get("converged")),
                        "scaling": params.get("gamma_scaling", "none"), "path": res_fn}
    return out


def print_tables(data):
    scalings = sorted({e["scaling"] for e in data.values()})
    print(f"gamma_scaling in these results: {', '.join(scalings)}")
    if len(scalings) > 1:
        print("warning: results mix scaling settings - point this at one scenario", file=sys.stderr)
    print("Implied real financing spread over the risk-free rate, in basis points (active technologies)\n")
    for reg in sorted({k[0] for k in data}):
        gs = sorted(g for (r, g) in data if r == reg)
        techs = [z for z in TECH_ORDER if any(z in data[(reg, g)]["spread_bp"] for g in gs)]
        print(f"--- {reg} ---")
        print(f"{'gamma':>6} " + " ".join(f"{z:>9}" for z in techs))
        for g in gs:
            e = data[(reg, g)]
            cells = " ".join(f"{e['spread_bp'][z]:9.0f}" if z in e["spread_bp"] else f"{'-':>9}" for z in techs)
            print(f"{g:6g} {cells}{'' if e['converged'] else '   (not converged)'}")
        print()


def calibrate(data, regime, tech, target_bp):
    """Linear interpolation of gamma where ``tech``'s spread in ``regime`` equals ``target_bp``."""
    pts = sorted((g, e["spread_bp"][tech], e["converged"]) for (r, g), e in data.items()
                 if r == regime and tech in e["spread_bp"])
    if len(pts) < 2:
        raise SystemExit(f"need {tech} active in {regime} at two or more gammas to calibrate; found {len(pts)}")
    spreads = [s for _, s, _ in pts]
    if any(b <= a for a, b in zip(spreads, spreads[1:])):
        raise SystemExit(f"{tech} spread in {regime} is not strictly increasing in gamma "
                         f"({[round(s) for s in spreads]}); cannot interpolate")
    if not spreads[0] <= target_bp <= spreads[-1]:
        raise SystemExit(f"target {target_bp:g} bp is outside the grid's range "
                         f"[{spreads[0]:.0f}, {spreads[-1]:.0f}] bp - add gammas that bracket it")
    for (g0, s0, c0), (g1, s1, c1) in zip(pts, pts[1:]):
        if s0 <= target_bp <= s1:
            if not (c0 and c1):
                print("warning: a bracketing run did not converge; treat gamma* as approximate", file=sys.stderr)
            return g0 + (target_bp - s0) * (g1 - g0) / (s1 - s0), (g0, g1), (s0, s1)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="+", help="one scenario's results tree, e.g. results/raw/gcap")
    ap.add_argument("--target-spread-bp", type=float, help="cited financing spread to match, in basis points")
    ap.add_argument("--calib-regime", default="R2", help="regime whose spread is matched (default R2, merchant)")
    ap.add_argument("--calib-tech", default="ccgt", help="technology whose spread is matched (default ccgt)")
    args = ap.parse_args(argv)

    data = load(args.dirs)
    if not data:
        raise SystemExit(f"no market-regime results under {args.dirs}")
    print_tables(data)
    if args.target_spread_bp is not None:
        g, (g0, g1), (s0, s1) = calibrate(data, args.calib_regime, args.calib_tech, args.target_spread_bp)
        print(f"gamma* = {g:.3f}: {args.calib_tech} in {args.calib_regime} reaches {args.target_spread_bp:g} bp "
              f"(interpolated between gamma={g0:g} at {s0:.0f} bp and gamma={g1:g} at {s1:.0f} bp)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
