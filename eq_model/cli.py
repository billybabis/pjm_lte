"""
Command line interface.

    python -m eq_model params                      # print the parameter table
    python -m eq_model check-load  --load DIR      # area-resolution report only
    python -m eq_model build-panel --load DIR --gen DIR --capacity FILE|--eia-active F --eia-retired F --out panel.npz
    python -m eq_model run   --panel panel.npz --regimes P1,P2,R2,R3,R4,R5,R6 --gamma 0.3 --out results/
    python -m eq_model sweep --panel panel.npz --regime R2 --gammas 0,0.1,0.25,0.5,1 --out results/
    python -m eq_model combine results/ --out results/summary.csv
    python -m eq_model plot results/ --gammas 0,0.3,1 --out figs/capacity.pdf

``run``/``sweep`` take --jobs N to solve regimes (resp. gammas) in N parallel processes, and
--workers M to solve the yearly dispatch LPs of each of those in M processes.  When the work is
split across machines instead (one regime per GitHub Actions matrix job), each job writes a
``row_<regime>.json`` next to its other outputs and ``combine`` stitches them into one summary.

Any ModelParams field can be overridden with --param name=value (repeatable), e.g.
  --param voll=20000 --param q_bar_rule=peak_load --param markdown_mode=full --param solar_capacity_basis=DC
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import dataclasses
import gc
import glob
import json
import logging
import multiprocessing as mp
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


def _regime_tau(P: ModelParams, name: str) -> float:
    return P.tau_scc if REGIMES[name].carbon_priced else 0.0


def _tau_groups(P: ModelParams, regimes: List[str]) -> List[List[str]]:
    """Regimes grouped by tau, order preserved.  One CapacityProblem (cut pool + one ~1M-column
    monolithic model for price completion) serves a whole group, and each regime warm-starts K from
    the previous one, so a group is the cheapest unit of work."""
    taus = []
    for name in regimes:
        tau = _regime_tau(P, name)
        if tau not in taus:
            taus.append(tau)
    return [[n for n in regimes if _regime_tau(P, n) == tau] for tau in taus]


def _split_units(groups: List[List[str]], jobs: int) -> List[List[str]]:
    """Split the tau-groups into at most ``jobs`` independent units.  Same-tau regimes stay together
    while there are no spare jobs; beyond that the largest group is halved, trading the shared cuts
    and the warm start for wall-clock."""
    units = [list(g) for g in groups]
    while len(units) < jobs and any(len(u) > 1 for u in units):
        i = max(range(len(units)), key=lambda j: len(units[j]))
        u = units.pop(i)
        h = len(u) // 2
        units[i:i] = [u[:h], u[h:]]
    return units


def _solve_and_write(panel, P: ModelParams, name: str, gamma: float, out: str, cap, K0, tol: float,
                     max_outer: int, verbose: bool, tag: str):
    """Solve one regime on an existing CapacityProblem and write its output files."""
    from .equilibrium import solve_regime
    from .welfare import evaluate_welfare, summary_row, profit_table
    reg = REGIMES[name]
    t0 = time.time()
    res = solve_regime(panel, P, reg, gamma=gamma, cap=cap, K0=K0, tol=tol, max_outer=max_outer, verbose=verbose)
    w = evaluate_welfare(panel, P, res)
    row = summary_row(w, res, P)
    # The planner regimes are risk-neutral by construction and always report gamma=0, so the
    # gamma the run was launched with is recorded separately; without it the P1/P2 rows of
    # different gamma runs are indistinguishable and collapse into one.
    row["gamma_requested"] = float(gamma)
    # How much of the runtime is the single-process completion LP: the part extra cores cannot help.
    row["completion_s"] = float(sum(h.get("completion_s", 0.0) for h in res.outer_history))
    row["n_evaluations"] = int(sum(h.get("evals", 0) for h in res.outer_history))
    suffix = f"{name}{tag}"
    with open(os.path.join(out, f"result_{suffix}.json"), "w") as fh:
        json.dump({"regime": dataclasses.asdict(reg), "gamma": res.gamma, "tau": res.tau, "K": res.K, "I": res.I,
                   "psi_coef": res.psi_coef, "cost_K_effective": res.cost_K, "risk_premium": res.risk_premium,
                   "implied_discount_rate": {z: implied_discount_rate(P.tech(z), res.cost_K[z]) if res.K[z] > 1 else None for z in res.K},
                   "rho_star": res.rho_star, "q_bar": res.q_bar, "Lambda": res.Lambda, "Pbar_by_year": res.Pbar,
                   "forward": dataclasses.asdict(res.forward) if res.forward else None, "markdown": res.markdown,
                   "converged": res.converged, "outer_history": res.outer_history,
                   "capacity_iterations": res.capacity_result.iterations, "foc": res.capacity_result.foc,
                   "n_evaluations": res.capacity_result.n_evaluations,
                   # Provenance: the full parameter set this result was produced with, so a
                   # results directory is a record rather than an assertion about which
                   # --param flags were typed.
                   "params": dataclasses.asdict(P),
                   "welfare": {"C_y": w.C_y, "C_mean": w.C_mean, "C_max": w.C_max, "C_risk_adjusted": w.C_risk_adjusted,
                               "tau_welfare": w.tau_welfare, "components_mean": w.components_mean, "emissions_y": w.emissions_y,
                               "lost_load_mwh_y": w.lost_load_mwh_y, "psi_load_mean": w.psi_load_mean,
                               "energy_share": w.energy_share, "curtailment_share": w.curtailment_share,
                               "curtailment_mwh_y": w.curtailment_mwh_y,
                               "curtailment_hours_y": w.curtailment_hours_y,
                               "curtailment_hours_any_y": w.curtailment_hours_any_y},
                   "runtime_s": time.time() - t0}, fh, indent=1, default=_json_default)
    profit_table(res, panel).to_csv(os.path.join(out, f"profits_{suffix}.csv"), index=False)
    np.savez_compressed(os.path.join(out, f"prices_{suffix}.npz"), price=res.evaluation.price, years=panel.years,
                        lost_load=np.stack([d.lost_load for d in res.evaluation.dispatch]))
    # One summary row per regime on disk, so a run split over several processes or machines can be
    # stitched back together with `python -m eq_model combine`.
    with open(os.path.join(out, f"row_{suffix}.json"), "w") as fh:
        json.dump(row, fh, indent=1, default=_json_default)
    log.info("%s done in %.0fs: K=%s  C_mean=%.3f $bn  conv=%s", suffix, time.time() - t0,
             {z: round(v) for z, v in res.K.items()}, w.C_mean / 1e9, res.converged)
    return row, res.K


def _run_unit(panel, P: ModelParams, unit, out: str, tol: float, max_outer: int, verbose: bool,
              workers: int) -> List[dict]:
    """Solve one unit -- (gamma, tag, [regimes]) at a single tau -- in this process."""
    from .equilibrium import CapacityProblem
    gamma, tag, names = unit
    cap = CapacityProblem(panel, P, _regime_tau(P, names[0]), workers=workers)
    try:
        rows, K = [], None
        for name in names:
            row, K = _solve_and_write(panel, P, name, gamma, out, cap, K, tol, max_outer, verbose, tag)
            rows.append(row)
    finally:
        cap.close()
    return rows


def _execute(panel, P: ModelParams, units, out: str, tol: float, max_outer: int, verbose: bool,
             workers: int, jobs: int) -> List[dict]:
    os.makedirs(out, exist_ok=True)
    n_par = min(max(1, int(jobs)), len(units))
    if n_par == 1:
        rows = []
        for u in units:
            rows += _run_unit(panel, P, u, out, tol, max_outer, verbose, workers)
            gc.collect()                                   # free the monolithic model before the next tau
        return rows
    log.warning("running %d units in %d processes; each keeps its own cut pool and monolithic LP (~3 GB at "
                "full size), so peak memory is ~%d x that, with %d x --workers %d = %d dispatch processes on top",
                len(units), n_par, n_par, n_par, workers, n_par * workers)
    ctx = mp.get_context("fork") if "fork" in mp.get_all_start_methods() else mp.get_context("spawn")
    rows: List[dict] = []
    with cf.ProcessPoolExecutor(max_workers=n_par, mp_context=ctx) as ex:
        futs = [ex.submit(_run_unit, panel, P, u, out, tol, max_outer, verbose, workers) for u in units]
        for f in futs:
            rows += f.result()
    return rows


def _gamma_col(df: pd.DataFrame) -> str:
    """The gamma to group/sort by: what the run asked for, not what a risk-neutral planner used."""
    return "gamma_requested" if "gamma_requested" in df.columns else "gamma"


def _order_rows(rows: List[dict], regimes: List[str]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["regime"] = pd.Categorical(df["regime"], categories=regimes, ordered=True)   # restore requested order
    df = df.sort_values(["regime", _gamma_col(df)]).reset_index(drop=True)
    df["regime"] = df["regime"].astype(str)
    return df


def run_regimes(panel, P: ModelParams, regimes: List[str], gamma: float, out: str, tol: float = 1e-3,
                max_outer: int = 60, verbose: bool = True, tag: str = "", workers: int = 1,
                jobs: int = 1) -> pd.DataFrame:
    """Solve ``regimes`` at one gamma.  ``jobs`` > 1 runs independent units of regimes in parallel
    processes; ``workers`` is the nested parallelism of the yearly dispatch LPs inside each unit."""
    units = [(gamma, tag, u) for u in _split_units(_tau_groups(P, regimes), jobs)]
    rows = _execute(panel, P, units, out, tol, max_outer, verbose, workers, jobs)
    df = _order_rows(rows, regimes)
    df.to_csv(os.path.join(out, f"summary{tag}.csv"), index=False)
    return df


_RUN_COLS = ["regime", "gamma", "converged", "C_mean_$bn", "C_risk_adj_$bn", "emissions_Mt_mean",
             "lost_load_hours_mean", "price_load_wtd", "p_hat"]


def _print_table(df: pd.DataFrame, cols: List[str]) -> None:
    cols = [c for c in cols if c in df.columns]
    with pd.option_context("display.width", 250, "display.max_columns", 60, "display.float_format", "{:,.3f}".format):
        print(df[cols].to_string(index=False))


def cmd_run(args):
    P = _apply_param_overrides(ModelParams(), args.param)
    panel = _load_panel(args)
    print(panel.summary())
    print(P.summary())
    regimes = [r.strip() for r in args.regimes.split(",") if r.strip()]
    unknown = [r for r in regimes if r not in REGIMES]
    if unknown:
        raise SystemExit(f"unknown regime(s) {unknown}; known: {sorted(REGIMES)}")
    df = run_regimes(panel, P, regimes, args.gamma, args.out, tol=args.tol, max_outer=args.max_outer,
                     verbose=not args.quiet, workers=args.workers, jobs=args.jobs)
    _print_table(df, _RUN_COLS + [c for c in df.columns if c.startswith("K_")])
    print("written to", args.out)


def cmd_sweep(args):
    P = _apply_param_overrides(ModelParams(), args.param)
    panel = _load_panel(args)
    gammas = [float(g) for g in args.gammas.split(",")]
    # One gamma per unit: each gamma re-solves the whole fixed point, so gammas parallelise cleanly.
    units = [(g, f"_gamma{g:g}", [args.regime]) for g in gammas]
    rows = _execute(panel, P, units, args.out, args.tol, args.max_outer, not args.quiet, args.workers, args.jobs)
    df = _order_rows(rows, [args.regime]).sort_values(_gamma_col(pd.DataFrame(rows))).reset_index(drop=True)
    df.to_csv(os.path.join(args.out, f"sweep_{args.regime}.csv"), index=False)
    _print_table(df, ["gamma", "converged", "C_mean_$bn", "price_load_wtd", "p_hat"] +
                 [c for c in df.columns if c.startswith("K_") or c.startswith("premium_pct_")])
    print("written to", args.out)


def _rows_agree(a: dict, b: dict, rtol: float = 1e-9) -> bool:
    """Same solve, allowing for wall-clock noise and float round-trips through JSON."""
    skip = {"runtime_s"}
    if set(a) - skip != set(b) - skip:
        return False
    for k in set(a) - skip:
        x, y = a[k], b[k]
        if isinstance(x, (int, float)) and isinstance(y, (int, float)) and not isinstance(x, bool):
            if x != y and not (np.isnan(x) and np.isnan(y)) and abs(x - y) > rtol * max(1.0, abs(x), abs(y)):
                return False
        elif x != y:
            return False
    return True


def _first_difference(a: dict, b: dict) -> str:
    for k in sorted(set(a) | set(b)):
        if k == "runtime_s":
            continue
        if a.get(k) != b.get(k):
            return f"{k}: {a.get(k)!r} vs {b.get(k)!r}"
    return "(none found outside runtime_s)"


def _collect_rows(dirs: List[str], regimes: Optional[str] = None) -> pd.DataFrame:
    """Read every ``row_*.json`` under ``dirs`` (recursively) into one ordered table.  This is what
    lets results produced by separate processes, or by separate CI matrix jobs, be merged."""
    rows, seen = [], {}
    for d in dirs:
        for fn in sorted(glob.glob(os.path.join(d, "**", "row_*.json"), recursive=True)):
            with open(fn) as fh:
                row = json.load(fh)
            key = (row.get("regime"), row.get("gamma_requested", row.get("gamma")))
            if key in seen:
                prev_fn, prev = seen[key]
                if _rows_agree(prev, row):
                    # The same solve reached us twice - e.g. one regime present in two
                    # downloaded CI artifact directories.  Harmless: keep either.
                    log.info("%s at gamma=%s seen twice with identical values (%s); keeping the first",
                             key[0], key[1], fn)
                    continue
                raise SystemExit(
                    f"{key[0]} at gamma={key[1]} appears twice with DIFFERENT results:\n"
                    f"    {prev_fn}\n    {fn}\n"
                    f"  first differing field: {_first_difference(prev, row)}\n"
                    "These are different runs (different --param settings?) that cannot share "
                    "one table or figure. Point the command at one of them, or plot them "
                    "separately.")
            seen[key] = (fn, row)
            rows.append(row)
    if not rows:
        raise SystemExit(f"no row_*.json found under {dirs}")
    order = [r.strip() for r in regimes.split(",")] if regimes else list(REGIMES)
    for r in rows:                                     # append-in-place: a regime present in the
        if r["regime"] not in order:                   # data but not requested must be added once,
            order.append(r["regime"])                  # not once per gamma
    return _order_rows(rows, order)


def cmd_combine(args):
    """Stitch the per-regime ``row_*.json`` files written by separate processes or machines (one
    GitHub Actions matrix job per regime) into a single summary table."""
    df = _collect_rows(args.dirs, args.regimes)
    if args.by_gamma:
        df = df.sort_values([_gamma_col(df), "regime"]).reset_index(drop=True)
    d = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(d, exist_ok=True)
    df.to_csv(args.out, index=False)
    _print_table(df, _RUN_COLS + [c for c in df.columns if c.startswith("K_")])
    print(len(df), "rows written to", args.out)


def _suffixed(path: str, suffix: str) -> str:
    root, ext = os.path.splitext(path)
    return f"{root}_{suffix}{ext or '.pdf'}"


def cmd_plot(args):
    """Capacity and/or energy-share figures: stacked bars by regime, one panel per gamma."""
    from .plots import capacity_panels, energy_panels, contract_panels
    df = _collect_rows(args.dirs, args.regimes)
    gammas = [float(g) for g in args.gammas.split(",")] if args.gammas else None
    regimes = [r.strip() for r in args.regimes.split(",") if r.strip()] if args.regimes else None
    kinds = {"both": ["capacity", "energy"],
             "all": ["capacity", "energy", "contract"]}.get(args.kind, [args.kind])
    common = dict(gammas=gammas, regimes=regimes, title=args.title, panel_width=args.panel_width,
                  height=args.height, dpi=args.dpi)
    for kind in kinds:
        # With several figures the name is disambiguated; with one, --out is taken literally.
        out = _suffixed(args.out, kind) if len(kinds) > 1 else args.out
        d = os.path.dirname(os.path.abspath(out))
        os.makedirs(d, exist_ok=True)
        if kind == "capacity":
            out = capacity_panels(df, out=out, capacity_unit=args.unit,
                                  show_curtailment=not args.no_curtailment, **common)
        elif kind == "energy":
            out = energy_panels(df, out=out, show_curtailment=not args.no_curtailment, **common)
        else:
            out = contract_panels(df, out=out, capacity_unit=args.unit,
                                  show_price=not args.no_price, show_empty=args.keep_empty, **common)
        print("figure written to", out)


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
        s.add_argument("--jobs", type=int, default=1,
                       help="parallel processes over regimes (run) / gammas (sweep); each holds ~3 GB at full size")
        s.add_argument("--quiet", action="store_true")
        if name == "run":
            s.add_argument("--regimes", default="P1,P2,R2,R3,R4,R5,R6"); s.add_argument("--gamma", type=float, default=0.0)
        else:
            s.add_argument("--regime", default="R2"); s.add_argument("--gammas", default="0,0.1,0.25,0.5,1.0")
        s.set_defaults(fn=fn)

    s = sub.add_parser("combine", help="merge per-regime row_*.json (from parallel jobs) into one summary.csv")
    s.add_argument("dirs", nargs="+", help="directories to scan recursively for row_*.json")
    s.add_argument("--out", required=True, help="output CSV")
    s.add_argument("--regimes", help="column order, e.g. P1,P2,R2,R3,R4,R5,R6 (default: the REGIMES order)")
    s.add_argument("--by-gamma", action="store_true", help="sort by gamma first (for sweeps)")
    s.set_defaults(fn=cmd_combine)

    s = sub.add_parser("plot", help="stacked capacity and/or energy-share bars by regime, "
                                    "one panel per gamma, with curtailment on a secondary axis")
    s.add_argument("dirs", nargs="+", help="directories to scan recursively for row_*.json")
    s.add_argument("--out", default="capacity.png", help="output figure; the extension picks the "
                                                       "format (.png/.pdf/.svg). With more than one "
                                                       "--kind the kind is appended to the stem")
    s.add_argument("--kind", default="capacity",
                   choices=["capacity", "energy", "contract", "both", "all"],
                   help="capacity: installed MW stacked by technology, curtailment hours overlaid; "
                        "energy: share of load served, curtailed VRE share overlaid; "
                        "contract: contracted capacity chi_z*K_z, forward price overlaid; "
                        "both: capacity+energy; all: all three")
    s.add_argument("--gammas", help="which gammas to draw, in panel order (default: all present)")
    s.add_argument("--regimes", help="which regimes to draw, in bar order (default: the REGIMES order)")
    s.add_argument("--title")
    s.add_argument("--unit", default="GW", choices=["GW", "MW"])
    s.add_argument("--panel-width", type=float, default=3.4, help="inches per panel")
    s.add_argument("--height", type=float, default=4.0, help="figure height, inches")
    s.add_argument("--dpi", type=int, default=300)
    s.add_argument("--no-curtailment", action="store_true", help="drop the curtailment overlay")
    s.add_argument("--no-price", action="store_true", help="drop the forward-price overlay (contract)")
    s.add_argument("--keep-empty", action="store_true",
                   help="contract figure: keep slots for regimes with no forward market "
                        "(P1/P2/R2/R6), so the regime axis matches the other figures")
    s.set_defaults(fn=cmd_plot)

    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    logging.getLogger("eq_model").setLevel(logging.DEBUG if args.verbose else logging.INFO)
    args.fn(args)


if __name__ == "__main__":
    main()
