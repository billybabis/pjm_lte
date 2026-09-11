"""
EIA-860M generator inventory -> monthly PJM solar / wind nameplate capacity and the
">10 MW plant" coverage share used to put PJM's gen_by_fuel numerator and the EIA
denominator on the same footing (parameters.tex, "Data").

Capacity basis: **AC nameplate** ("Nameplate Capacity (MW)").  The DC column
("DC Net Capacity (MW)") is read and reported for solar but not used in the model.

All numeric columns are parsed with :func:`eq_model.io_utils.parse_number`, so a value
written as ``"1,281.0"`` is 1281.0 and an unparsable non-blank value raises.
"""
from __future__ import annotations

import logging
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from .io_utils import DataError, find_col, parse_number, read_csv_stack

log = logging.getLogger("eq_model")

SOLAR_TECH = {"solar photovoltaic"}
WIND_TECH = {"onshore wind turbine", "offshore wind turbine"}
SOLAR_ESC = {"SUN"}
WIND_ESC = {"WND"}


def read_eia_generators(path: str, kind: str) -> pd.DataFrame:
    """Read an EIA-860M sheet exported to CSV (kind = 'active' | 'retired').

    Returns a normalised frame with columns:
    plant_id, plant_name, generator_id, technology, esc, ba, state, cap_ac_mw, cap_dc_mw,
    op_date (month start), ret_date (month start or NaT), status, vre ("solar"/"wind"/"").
    """
    df = read_csv_stack(path)
    c = lambda names, req=True: find_col(df, names, required=req)
    out = pd.DataFrame({
        "plant_id": df[c(["Plant ID", "plant_id", "plant_code"])].astype(str).str.strip(),
        "plant_name": df[c(["Plant Name", "plant_name"], req=False) or df.columns[0]].astype(str),
        "generator_id": df[c(["Generator ID", "generator_id", "gen_id"], req=False) or df.columns[0]].astype(str),
        "technology": df[c(["Technology"], req=False) or df.columns[0]].astype(str).str.strip(),
        "esc": df[c(["Energy Source Code", "energy_source_code"], req=False) or df.columns[0]].astype(str).str.strip().str.upper(),
        "ba": df[c(["Balancing Authority Code", "balancing_authority_code", "ba_code", "BA"], req=True)].astype(str).str.strip().str.upper(),
        "state": df[c(["Plant State", "state"], req=False) or df.columns[0]].astype(str).str.strip(),
        "status": df[c(["Status"], req=False) or df.columns[0]].astype(str).str.strip(),
    })
    cap_col = c(["Nameplate Capacity (MW)", "nameplate_capacity_mw", "nameplate_capacity"])
    out["cap_ac_mw"] = parse_number(df[cap_col], f"eia.{cap_col}")
    dc_col = c(["DC Net Capacity (MW)", "dc_net_capacity_mw", "dc_capacity_mw"], req=False)
    out["cap_dc_mw"] = parse_number(df[dc_col], f"eia.{dc_col}") if dc_col else np.nan

    def month_start(ycol_names, mcol_names):
        yc = c(ycol_names, req=False)
        mc = c(mcol_names, req=False)
        if yc is None:
            return pd.Series(pd.NaT, index=df.index)
        y = parse_number(df[yc], f"eia.{yc}", strict=False)
        m = parse_number(df[mc], f"eia.{mc}", strict=False) if mc else pd.Series(1.0, index=df.index)
        # EIA-860M writes 0 / 88 / 99 in the month column when the month is unknown
        # (common for pre-1990 retired units).  Anything outside 1..12 -> January.
        bad = m.notna() & ~m.between(1, 12)
        if bad.any():
            log.warning("eia[%s]: %d rows with an out-of-range %s (%s) - month set to 1",
                        kind, int(bad.sum()), mc, sorted(m[bad].unique().tolist())[:5])
            m = m.where(~bad, 1.0)
        m = m.fillna(1.0)
        ok = y.notna()
        res = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
        res[ok] = pd.to_datetime({"year": y[ok].astype(int), "month": m[ok].astype(int), "day": 1})
        return res

    out["op_date"] = month_start(["Operating Year", "operating_year"], ["Operating Month", "operating_month"])
    if kind == "retired":
        out["ret_date"] = month_start(["Retirement Year", "retirement_year"], ["Retirement Month", "retirement_month"])
    else:
        out["ret_date"] = pd.NaT

    tech_l = out["technology"].str.lower()
    out["vre"] = np.where(tech_l.isin(SOLAR_TECH) | ((tech_l == "") & out["esc"].isin(SOLAR_ESC)), "solar",
                 np.where(tech_l.isin(WIND_TECH) | ((tech_l == "") & out["esc"].isin(WIND_ESC)), "wind", ""))
    out["kind"] = kind
    n_missing_cap = int(out["cap_ac_mw"].isna().sum())
    if n_missing_cap:
        log.warning("eia[%s]: %d rows with blank AC nameplate capacity (kept as NaN, excluded from sums)",
                    kind, n_missing_cap)
    log.info("eia[%s]: %d rows; VRE rows solar=%d wind=%d; BAs=%d", kind, len(out),
             int((out.vre == "solar").sum()), int((out.vre == "wind").sum()), out.ba.nunique())
    return out


def monthly_vre_capacity(active: pd.DataFrame, retired: Optional[pd.DataFrame] = None,
                         ba: str = "PJM", start: str = "2017-01", end: str = "2026-08",
                         threshold_mw: float = 10.0, statuses: Optional[Iterable[str]] = None,
                         ) -> pd.DataFrame:
    """Monthly AC nameplate capacity of PJM solar and wind plus the coverage share.

    coverage_{r,month} = capacity in plants whose *plant-level* capacity of technology r
    (summed over its operating generators in that month) exceeds ``threshold_mw``,
    divided by total capacity of r.  This is the denominator adjustment
    theta = gen / (capacity * coverage) in parameters.tex.

    Returns columns: month, solar_mw, wind_mw, solar_coverage, wind_coverage, solar_mw_dc,
    n_solar_plants, n_wind_plants.
    """
    frames = [active] + ([retired] if retired is not None else [])
    g = pd.concat(frames, ignore_index=True)
    g = g[(g["ba"] == ba.upper()) & (g["vre"] != "")].copy()
    if statuses is not None:
        g = g[g["status"].isin(list(statuses))]
    if g["op_date"].isna().any():
        log.warning("eia: %d PJM VRE generators without operating date dropped", int(g["op_date"].isna().sum()))
        g = g[g["op_date"].notna()]
    months = pd.period_range(start, end, freq="M").to_timestamp()
    rows = []
    for m in months:
        live = g[(g["op_date"] <= m) & (g["ret_date"].isna() | (g["ret_date"] > m))]
        rec = {"month": m}
        for r in ("solar", "wind"):
            sub = live[live["vre"] == r]
            plant_cap = sub.groupby("plant_id")["cap_ac_mw"].sum(min_count=1)
            total = float(plant_cap.sum())
            big = float(plant_cap[plant_cap > threshold_mw].sum())
            rec[f"{r}_mw"] = total
            rec[f"{r}_coverage"] = big / total if total > 0 else np.nan
            rec[f"n_{r}_plants"] = int(plant_cap.notna().sum())
            if r == "solar":
                rec["solar_mw_dc"] = float(sub["cap_dc_mw"].sum(min_count=1)) if sub["cap_dc_mw"].notna().any() else np.nan
        rows.append(rec)
    out = pd.DataFrame(rows)
    log.info("eia: monthly PJM VRE capacity built for %d months (threshold %.1f MW); "
             "last month solar=%.0f MW AC (%.0f DC), wind=%.0f MW; coverage solar=%.3f wind=%.3f",
             len(out), threshold_mw, out.solar_mw.iloc[-1], out.solar_mw_dc.iloc[-1] if out.solar_mw_dc.notna().any() else float('nan'),
             out.wind_mw.iloc[-1], out.solar_coverage.iloc[-1], out.wind_coverage.iloc[-1])
    return out


def load_vre_capacity_csv(path: str) -> pd.DataFrame:
    """Load a *derived* monthly VRE capacity file (e.g. the user's ``vre_capacity.csv``).

    Accepts wide (month, solar..., wind..., optional coverage columns) or long
    (month, technology, mw) layouts.  Returns the same schema as
    :func:`monthly_vre_capacity` with NaN coverage when not present.
    """
    df = read_csv_stack(path)
    df = df.drop(columns=["__source_file"])
    cols_l = {c: c.lower() for c in df.columns}
    date_col = find_col(df, ["month", "date", "period", "year_month", "ym", "time", "datetime"], required=False, contains=True)
    if date_col is None:
        raise DataError(f"vre_capacity: no month/date column found in {list(df.columns)}")
    month = pd.to_datetime(df[date_col].astype(str).str.strip(), format="mixed").dt.to_period("M").dt.to_timestamp()
    out = pd.DataFrame({"month": month})

    tech_col = find_col(df, ["technology", "tech", "fuel", "resource"], required=False)
    if tech_col is not None and not any("solar" in cols_l[c] for c in df.columns):
        # long layout
        mw_col = find_col(df, ["mw", "capacity_mw", "nameplate_mw", "capacity"], contains=True)
        df["_mw"] = parse_number(df[mw_col], f"vre_capacity.{mw_col}")
        df["_tech"] = df[tech_col].astype(str).str.lower().str.strip()
        wide = df.assign(month=month).pivot_table(index="month", columns="_tech", values="_mw", aggfunc="sum")
        out = pd.DataFrame({"month": wide.index})
        out["solar_mw"] = wide.filter(like="solar").sum(axis=1).values
        out["wind_mw"] = wide.filter(like="wind").sum(axis=1).values
        out["solar_coverage"] = np.nan
        out["wind_coverage"] = np.nan
        return out.drop_duplicates("month").sort_values("month").reset_index(drop=True)

    def pick(sub: str, extra_excl: tuple = ("coverage", "share", "dc", "count", "plants", "n_")):
        cands = [c for c in df.columns if sub in cols_l[c] and not any(x in cols_l[c] for x in extra_excl)]
        if not cands:
            raise DataError(f"vre_capacity: no {sub} capacity column in {list(df.columns)}")
        return cands[0]

    out["solar_mw"] = parse_number(df[pick("solar")], "vre_capacity.solar")
    out["wind_mw"] = parse_number(df[pick("wind")], "vre_capacity.wind")
    for r in ("solar", "wind"):
        cov = [c for c in df.columns if r in cols_l[c] and ("coverage" in cols_l[c] or "share" in cols_l[c])]
        out[f"{r}_coverage"] = parse_number(df[cov[0]], f"vre_capacity.{cov[0]}") if cov else np.nan
    dc = [c for c in df.columns if "solar" in cols_l[c] and "dc" in cols_l[c]]
    out["solar_mw_dc"] = parse_number(df[dc[0]], f"vre_capacity.{dc[0]}") if dc else np.nan
    out = out.drop_duplicates("month").sort_values("month").reset_index(drop=True)
    log.info("vre_capacity.csv: %d months %s..%s; coverage columns present: solar=%s wind=%s",
             len(out), out.month.iloc[0].date(), out.month.iloc[-1].date(),
             out.solar_coverage.notna().any(), out.wind_coverage.notna().any())
    return out
