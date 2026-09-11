"""
Command line interface.

    python -m eq_model params                      # print the parameter table
    python -m eq_model check-load  --load DIR      # area-resolution report only
    python -m eq_model build-panel --load DIR --gen DIR --capacity FILE|--eia-active F --eia-retired F --out panel.npz
    python -m eq_model run   --panel panel.npz --regimes P1,P2,R2,R3,R4,R5,R6 --gamma 0.3 --out results/
    python -m eq_model sweep --panel panel.npz --regime R2 --gammas 0,0.1,0.25,0.5,1 --out results/

Any ModelParams field can be overridden with --param name=value (repeatable), e.g.
  --param voll=20000 --param q_bar_rule=peak_load --param markdown_mode=full --param solar_capacity_basis=DC
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
import sys
import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .params import ModelParams, REGIMES, implied_discount_rate

log = logging.getLogger("eq_model")


def _parse_years(s: Optional[str]) -> Optional[List[int]]:
    if not s:
        return None
    out = []
    for part in s.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def _apply_param_overrides(params: ModelParams, items: Optional[List[str]]) -> ModelParams:
    if not items:
        return params
    kw = {}
    fields = {f.name: f for f in dataclasses.fields(ModelParams)}
    for it in items:
        if "=" not in it:
            raise SystemExit(f"--param expects name=value, got {it!r}")
        k, v = it.split("=", 1)
        k = k.strip()
        if k not in fields:
            raise SystemExit(f"unknown parameter {k!r}; known: {sorted(fields)}")
        cur = getattr(params, k)
        if isinstance(cur, bool):
            val = v.strip().lower() in ("1", "true", "yes", "y")
        elif isinstance(cur, (int, float)) or cur is None:
            try:
                val = float(v) if ("." in v or "e" in v.lower() or cur is None or isinstance(cur, float)) else int(v)
            except ValueError:
                val = v
        elif isinstance(cur, tuple):
            val = tuple(int(x) for x in v.split(","))
        else:
            val = v
        kw[k] = val
    return params.with_(**kw)


def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if dataclasses.is_dataclass(o):
        return dataclasses.asdict(o)
    return str(o)


def cmd_params(args):
    P = _apply_param_overrides(ModelParams(), args.param)
    print(P.summary())


def cmd_check_load(args):
    from .pjm_data import load_pjm_load
    _, res = load_pjm_load(args.load, area_mode=args.area_mode)
    print(res.report())


def cmd_build_panel(args):
    from .pjm_data import load_pjm_load, load_gen_by_fuel
    from .eia import read_eia_generators, monthly_vre_capacity, load_vre_capacity_csv
    from .panel import build_panel
    P = _apply_param_overrides(ModelParams(), args.param)
    load, res = load_pjm_load(args.load, area_mode=args.area_mode)
    print(res.report())
    gen = load_gen_by_fuel(args.gen)
    cap = None
    if args.eia_active:
        act = read_eia_generators(args.eia_active, "active")
        ret = read_eia_generators(args.eia_retired, "retired") if args.eia_retired else None
        years = _parse_years(args.years) or sorted(set(load.index.tz_convert("America/New_York").year))
        cap = monthly_vre_capacity(act, ret, ba=args.ba, start=f"{min(years)}-01", end=f"{max(years)}-12",
                                   threshold_mw=args.threshold_mw)
        if args.capacity:
            user_cap = load_vre_capacity_csv(args.capacity)
            m = cap.merge(user_cap, on="month", suffixes=("", "_user"))
            if len(m):
                for r in ("solar", "wind"):
                    d = (m[f"{r}_mw"] / m[f"{r}_mw_user"] - 1).abs()
                    print(f"cross-check EIA-built vs {os.path.basename(args.capacity)}: {r} capacity max |rel diff| = {d.max():.3%} "
                          f"(mean {d.mean():.3%}) over {len(m)} months")
                    if f"{r}_coverage_user" in m and m[f"{r}_coverage_user"].notna().any():
                        dc = (m[f"{r}_coverage"] - m[f"{r}_coverage_user"]).abs()
                        print(f"   coverage max |diff| = {dc.max():.4f}")
        if args.save_capacity:
            cap.to_csv(args.save_capacity, index=False)
            print("monthly capacity + coverage written to", args.save_capacity)
    elif args.capacity:
        cap = load_vre_capacity_csv(args.capacity)
        if cap["solar_coverage"].isna().all():
            print("WARNING: the capacity file has no coverage columns; pass --eia-active/--eia-retired to compute the "
                  ">%.0f MW coverage adjustment, otherwise coverage=1 is used and theta is biased low." % args.threshold_mw)
    else:
        raise SystemExit("need --capacity and/or --eia-active")
    panel = build_panel(load, gen, cap, P, years=_parse_years(args.years), drop_feb29=not args.keep_feb29)
    print(panel.summary())
    panel.save(args.out)
    print("panel written to", args.out)


def _load_panel(args):
    from .panel import HourlyPanel
    panel = HourlyPanel.load(args.panel)
    yrs = _parse_years(args.years)
    if yrs:
        panel = panel.subset(yrs)
    if getattr(args, "hours", None):
        panel = panel.__class__(years=panel.years, utc=panel.utc[:, :args.hours], hour_of_day=panel.hour_of_day[:, :args.hours],
                                D=panel.D[:, :args.hours], theta={k: v[:, :args.hours] for k, v in panel.theta.items()},
                                lam=panel.lam, daylight=panel.daylight[:, :args.hours], meta=dict(panel.meta))
    return panel


def run_regimes(panel, P: ModelParams, regimes: List[str], gamma: float, out: str, tol: float = 1e-3,
                max_outer: int = 60, verbose: bool = True, tag: str = "", workers: int = 1) -> pd.DataFrame:
    from .equilibrium import CapacityProblem, solve_regime
    from .welfare import evaluate_welfare, summary_row, profit_table
    os.makedirs(out, exist_ok=True)
    rows = []
    # Group regimes by tau so that one CapacityProblem (cut pool + one ~1M-column monolithic
    # model for price completion) is alive at a time; order within a group is preserved.
    taus = []
    for name in regimes:
        tau = P.tau_scc if REGIMES[name].carbon_priced else 0.0
        if tau not in taus:
            taus.append(tau)
    ordered = [n for tau in taus for n in regimes if (P.tau_scc if REGIMES[n].carbon_priced else 0.0) == tau]
    cap = None
    cap_tau = None
    K_start = None
    for name in ordered:
        reg = REGIMES[name]
        tau = P.tau_scc if reg.carbon_priced else 0.0
        if cap is None or cap_tau != tau:
            if cap is not None:
                cap.close()
                del cap                                        # free the previous monolithic model
                import gc; gc.collect()
            cap = CapacityProblem(panel, P, tau, workers=workers)
            cap_tau = tau
            K_start = None
        t0 = time.time()
        res = solve_regime(panel, P, reg, gamma=gamma, cap=cap, K0=K_start, tol=tol, max_outer=max_outer, verbose=verbose)
        K_start = res.K                                        # next regime (same tau) starts here
        w = evaluate_welfare(panel, P, res)
        row = summary_row(w, res, P)
        rows.append(row)
        suffix = f"{name}{tag}"
        with open(os.path.join(out, f"result_{suffix}.json"), "w") as fh:
            json.dump({"regime": dataclasses.asdict(reg), "gamma": res.gamma, "tau": res.tau, "K": res.K, "I": res.I,
                       "psi_coef": res.psi_coef, "cost_K_effective": res.cost_K, "risk_premium": res.risk_premium,
                       "implied_discount_rate": {z: implied_discount_rate(P.tech(z), res.cost_K[z]) if res.K[z] > 1 else None for z in res.K},
                       "rho_star": res.rho_star, "q_bar": res.q_bar, "Lambda": res.Lambda, "Pbar_by_year": res.Pbar,
                       "forward": dataclasses.asdict(res.forward) if res.forward else None, "markdown": res.markdown,
                       "converged": res.converged, "outer_history": res.outer_history,
                       "capacity_iterations": res.capacity_result.iterations, "foc": res.capacity_result.foc,
                       "welfare": {"C_y": w.C_y, "C_mean": w.C_mean, "C_max": w.C_max, "C_risk_adjusted": w.C_risk_adjusted,
                                   "tau_welfare": w.tau_welfare, "components_mean": w.components_mean, "emissions_y": w.emissions_y,
                                   "lost_load_mwh_y": w.lost_load_mwh_y, "psi_load_mean": w.psi_load_mean,
                                   "energy_share": w.energy_share, "curtailment_share": w.curtailment_share},
                       "runtime_s": time.time() - t0}, fh, indent=1, default=_json_default)
        profit_table(res, panel).to_csv(os.path.join(out, f"profits_{suffix}.csv"), index=False)
        np.savez_compressed(os.path.join(out, f"prices_{suffix}.npz"), price=res.evaluation.price, years=panel.years,
                            lost_load=np.stack([d.lost_load for d in res.evaluation.dispatch]))
        log.info("%s done in %.0fs: K=%s  C_mean=%.3f $bn  conv=%s", name, time.time() - t0,
                 {z: round(v) for z, v in res.K.items()}, w.C_mean / 1e9, res.converged)
    df = pd.DataFrame(rows)
    df["regime"] = pd.Categorical(df["regime"], categories=regimes, ordered=True)   # restore requested order
    df = df.sort_values("regime").reset_index(drop=True)
    df["regime"] = df["regime"].astype(str)
    df.to_csv(os.path.join(out, f"summary{tag}.csv"), index=False)
    if cap is not None:
        cap.close()
    return df


def cmd_run(args):
    P = _apply_param_overrides(ModelParams(), args.param)
    panel = _load_panel(args)
    print(panel.summary())
    print(P.summary())
    regimes = [r.strip() for r in args.regimes.split(",") if r.strip()]
    df = run_regimes(panel, P, regimes, args.gamma, args.out, tol=args.tol, max_outer=args.max_outer, verbose=not args.quiet,
                     workers=args.workers)
    cols = ["regime", "gamma", "converged", "C_mean_$bn", "C_risk_adj_$bn", "emissions_Mt_mean", "lost_load_hours_mean",
            "price_load_wtd", "p_hat"] + [c for c in df.columns if c.startswith("K_")]
    with pd.option_context("display.width", 250, "display.max_columns", 50, "display.float_format", "{:,.3f}".format):
        print(df[cols].to_string(index=False))
    print("written to", args.out)


def cmd_sweep(args):
    P = _apply_param_overrides(ModelParams(), args.param)
    panel = _load_panel(args)
    gammas = [float(g) for g in args.gammas.split(",")]
    frames = []
    for g in gammas:
        df = run_regimes(panel, P, [args.regime], g, args.out, tol=args.tol, max_outer=args.max_outer,
                         verbose=not args.quiet, tag=f"_gamma{g:g}", workers=args.workers)
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(os.path.join(args.out, f"sweep_{args.regime}.csv"), index=False)
    cols = ["gamma", "converged", "C_mean_$bn", "price_load_wtd", "p_hat"] + [c for c in df.columns if c.startswith("K_") or c.startswith("premium_pct_")]
    with pd.option_context("display.width", 250, "display.max_columns", 60, "display.float_format", "{:,.3f}".format):
        print(df[cols].to_string(index=False))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="eq_model", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("params"); s.add_argument("--param", action="append"); s.set_defaults(fn=cmd_params)

    s = sub.add_parser("check-load"); s.add_argument("--load", required=True)
    s.add_argument("--area-mode", default="auto", choices=["auto", "rto", "zonal_sum"]); s.set_defaults(fn=cmd_check_load)

    s = sub.add_parser("build-panel")
    s.add_argument("--load", required=True, help="directory of hrl_load_metered CSVs")
    s.add_argument("--gen", required=True, help="directory of gen_by_fuel CSVs")
    s.add_argument("--capacity", help="derived monthly capacity CSV (e.g. vre_capacity.csv)")
    s.add_argument("--eia-active"); s.add_argument("--eia-retired")
    s.add_argument("--ba", default="PJM"); s.add_argument("--threshold-mw", type=float, default=10.0)
    s.add_argument("--save-capacity", help="write the EIA-built monthly capacity+coverage table here")
    s.add_argument("--years", help="e.g. 2017-2025 or 2017,2019,2021")
    s.add_argument("--keep-feb29", action="store_true")
    s.add_argument("--area-mode", default="auto", choices=["auto", "rto", "zonal_sum"])
    s.add_argument("--param", action="append")
    s.add_argument("--out", required=True); s.set_defaults(fn=cmd_build_panel)

    for name, fn in (("run", cmd_run), ("sweep", cmd_sweep)):
        s = sub.add_parser(name)
        s.add_argument("--panel", required=True); s.add_argument("--years")
        s.add_argument("--hours", type=int, help="TEST ONLY: keep the first N hours of each year")
        s.add_argument("--out", required=True); s.add_argument("--param", action="append")
        s.add_argument("--tol", type=float, default=1e-3); s.add_argument("--max-outer", type=int, default=60)
        s.add_argument("--workers", type=int, default=1, help="parallel processes for the yearly dispatch LPs (<= number of years)")
        s.add_argument("--quiet", action="store_true")
        if name == "run":
            s.add_argument("--regimes", default="P1,P2,R2,R3,R4,R5,R6"); s.add_argument("--gamma", type=float, default=0.0)
        else:
            s.add_argument("--regime", default="R2"); s.add_argument("--gammas", default="0,0.1,0.25,0.5,1.0")
        s.set_defaults(fn=fn)

    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    logging.getLogger("eq_model").setLevel(logging.DEBUG if args.verbose else logging.INFO)
    args.fn(args)


if __name__ == "__main__":
    main()
