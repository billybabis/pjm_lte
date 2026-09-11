"""
Loaders for PJM Data Miner 2 exports.

* ``hrl_load_metered``  -> hourly RTO load with explicit area resolution and a
  zonal-sum vs RTO-row cross-check (the export mixes zonal rows with aggregate rows;
  summing everything double/triple counts).
* ``gen_by_fuel``       -> hourly generation by fuel type (wide table).

Both return DataFrames indexed by ``datetime_beginning_utc`` (tz-aware UTC).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Set

import numpy as np
import pandas as pd

from .io_utils import (DataError, find_col, normalize_label, parse_datetime, parse_number,
                       read_csv_stack)

log = logging.getLogger("eq_model")

# Labels (after normalize_label) that denote the RTO-wide total row.
RTO_LABELS: Set[str] = {"RTO", "PJMRTO", "PJM", "RTOTOTAL", "PJMTOTAL", "TOTAL"}
# Labels that denote market-region aggregates (subsets of the RTO, supersets of zones).
REGION_LABELS: Set[str] = {"MIDATL", "MIDATLANTIC", "PJMMIDATL", "PJMMIDATLANTIC",
                           "WEST", "PJMWEST", "SOUTH", "PJMSOUTH", "WESTERN", "SOUTHERN",
                           "MIDATLANTICREGION", "WESTREGION", "SOUTHREGION"}


@dataclass
class AreaResolution:
    """Diagnostics of the load area classification and cross-check."""
    area_column: str
    rto_labels: List[str]
    region_labels: List[str]
    zonal_labels: List[str]
    n_hours: int
    used: str                                   # "rto_row" | "zonal_sum"
    rel_diff_mean: float = float("nan")         # |zonal_sum - rto| / rto, mean over hours
    rel_diff_max: float = float("nan")
    n_hours_diff_gt_1pct: int = 0
    n_hours_missing_rto: int = 0
    notes: List[str] = field(default_factory=list)

    def report(self) -> str:
        lines = [f"Load area resolution (column={self.area_column!r}):",
                 f"  RTO-total labels   : {self.rto_labels}",
                 f"  region aggregates  : {self.region_labels}",
                 f"  zonal labels ({len(self.zonal_labels)}): {self.zonal_labels}",
                 f"  hours              : {self.n_hours}",
                 f"  D_hy taken from    : {self.used}"]
        if self.rto_labels:
            lines.append(f"  zonal-sum vs RTO   : mean |diff|/RTO = {self.rel_diff_mean:.4%}, "
                         f"max = {self.rel_diff_max:.4%}, hours with |diff|>1% = {self.n_hours_diff_gt_1pct}, "
                         f"hours without RTO row = {self.n_hours_missing_rto}")
        for n in self.notes:
            lines.append(f"  NOTE: {n}")
        return "\n".join(lines)


def classify_area(label: str, extra_rto: Iterable[str] = (), extra_region: Iterable[str] = ()) -> str:
    n = normalize_label(label)
    if n in RTO_LABELS or n in {normalize_label(x) for x in extra_rto}:
        return "rto"
    if n in REGION_LABELS or n in {normalize_label(x) for x in extra_region}:
        return "region"
    if n == "":
        return "blank"
    return "zone"


def load_pjm_load(path: str, area_mode: str = "auto", area_column: Optional[str] = None,
                  extra_rto_labels: Iterable[str] = (), extra_region_labels: Iterable[str] = (),
                  tolerance_mean: float = 0.02, tolerance_max: float = 0.10,
                  ) -> "tuple[pd.Series, AreaResolution]":
    """Return hourly PJM RTO load (MW), indexed by UTC hour, plus diagnostics.

    Parameters
    ----------
    path : directory of hrl_load_metered CSVs (or a single file)
    area_mode : "auto" (RTO row if present and consistent with the zonal sum, else zonal sum),
                "rto" (require the RTO row), "zonal_sum" (ignore aggregate rows and sum zones)
    area_column : override for the column holding the area label (default: load_area, then zone)
    """
    df = read_csv_stack(path)
    tcol = find_col(df, ["datetime_beginning_utc", "datetime_utc", "utc"])
    mwcol = find_col(df, ["mw", "load_mw", "mw_load", "value"])
    if area_column is None:
        area_column = find_col(df, ["load_area", "zone", "area", "region"], required=True)
    zone_col = find_col(df, ["zone"], required=False)

    df["_utc"] = parse_datetime(df[tcol], tcol, utc=True)
    df["_mw"] = parse_number(df[mwcol], f"load.{mwcol}")

    # Combined label: prefer the area column; if blank, fall back to the zone column so that
    # aggregate rows encoded as (zone='RTO', load_area='') are still recognised.
    lab = df[area_column].astype(str)
    if zone_col is not None and zone_col != area_column:
        blank = lab.str.strip() == ""
        lab = lab.where(~blank, df[zone_col].astype(str))
    df["_label"] = lab.str.strip()
    df["_class"] = df["_label"].map(lambda x: classify_area(x, extra_rto_labels, extra_region_labels))

    labels_by_class: Dict[str, List[str]] = {}
    for cls, grp in df.groupby("_class"):
        labels_by_class[cls] = sorted(grp["_label"].unique().tolist())
    if labels_by_class.get("blank"):
        log.warning("load: %d rows with blank area label dropped", int((df["_class"] == "blank").sum()))
        df = df[df["_class"] != "blank"]

    # Duplicate (utc, label) rows: keep the first, warn.
    dup = df.duplicated(subset=["_utc", "_label"], keep="first")
    if dup.any():
        log.warning("load: %d duplicate (utc, area) rows dropped (first kept)", int(dup.sum()))
        df = df[~dup]

    zonal = df[df["_class"] == "zone"].groupby("_utc")["_mw"].sum(min_count=1)
    rto = df[df["_class"] == "rto"].groupby("_utc")["_mw"].sum(min_count=1)
    region = df[df["_class"] == "region"].groupby("_utc")["_mw"].sum(min_count=1)

    res = AreaResolution(area_column=area_column,
                         rto_labels=labels_by_class.get("rto", []),
                         region_labels=labels_by_class.get("region", []),
                         zonal_labels=labels_by_class.get("zone", []),
                         n_hours=int(zonal.index.nunique() if len(zonal) else rto.index.nunique()),
                         used="")
    if len(res.rto_labels) > 1:
        res.notes.append(f"more than one RTO-like label {res.rto_labels}; they are summed - check!")

    if len(rto):
        both = pd.concat([zonal.rename("zonal"), rto.rename("rto")], axis=1)
        have = both.dropna()
        rel = (have["zonal"] - have["rto"]).abs() / have["rto"].abs().clip(lower=1.0)
        res.rel_diff_mean = float(rel.mean()) if len(rel) else float("nan")
        res.rel_diff_max = float(rel.max()) if len(rel) else float("nan")
        res.n_hours_diff_gt_1pct = int((rel > 0.01).sum())
        res.n_hours_missing_rto = int(both["rto"].isna().sum())
        if len(region):
            r2 = pd.concat([region.rename("region"), rto.rename("rto")], axis=1).dropna()
            rr = ((r2["region"] - r2["rto"]).abs() / r2["rto"].abs().clip(lower=1.0)).mean()
            res.notes.append(f"sum of region aggregates vs RTO: mean rel diff {rr:.4%}")
        if len(zonal) == 0:
            res.notes.append("no zonal rows found - only the RTO row is available")
    else:
        if area_mode == "rto":
            raise DataError("area_mode='rto' but no RTO-total row was found; labels: "
                            f"{labels_by_class}")
        res.notes.append("no RTO-total row found; D_hy = sum of zonal rows")

    consistent = (len(rto) > 0 and np.isfinite(res.rel_diff_mean)
                  and res.rel_diff_mean <= tolerance_mean and res.rel_diff_max <= tolerance_max)
    # 'auto': the RTO row is the direct metered total and is preferred whenever it exists; an
    # inconsistent zonal sum most likely means an unrecognised aggregate label among the "zones"
    # (which would inflate the sum), so we warn loudly rather than fall back to the sum.
    if area_mode == "rto" or (area_mode == "auto" and len(rto)):
        series = rto
        res.used = "rto_row"
        if len(zonal) and not consistent:
            msg = ("RTO row used, but the zonal sum disagrees with it beyond tolerance "
                   f"(mean {res.rel_diff_mean:.3%}, max {res.rel_diff_max:.3%}). Check the label classification "
                   "above (an aggregate classified as a zone inflates the sum; a zone classified as an aggregate "
                   "deflates it) and pass extra_rto_labels/extra_region_labels if needed.")
            res.notes.append(msg)
            log.warning("load: %s", msg)
    else:
        series = zonal
        res.used = "zonal_sum"
    series = series.sort_index()
    series.name = "load_mw"
    res.n_hours = int(len(series))
    log.info("\n%s", res.report())
    return series, res


def load_gen_by_fuel(path: str, fuels: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """Return an hourly wide table (UTC index) with one column per fuel type (MW).

    All fuel types present in the files are returned; ``fuels`` optionally restricts
    to a subset (matched case-insensitively, e.g. ["solar", "wind"]).
    """
    df = read_csv_stack(path)
    tcol = find_col(df, ["datetime_beginning_utc", "datetime_utc", "utc"])
    fcol = find_col(df, ["fuel_type", "fuel", "fuel_category"])
    mwcol = find_col(df, ["mw", "value", "generation_mw"])
    df["_utc"] = parse_datetime(df[tcol], tcol, utc=True)
    df["_mw"] = parse_number(df[mwcol], f"gen_by_fuel.{mwcol}")
    df["_fuel"] = df[fcol].astype(str).str.strip().str.lower()
    dup = df.duplicated(subset=["_utc", "_fuel"], keep="first")
    if dup.any():
        log.warning("gen_by_fuel: %d duplicate (utc, fuel) rows dropped", int(dup.sum()))
        df = df[~dup]
    wide = df.pivot(index="_utc", columns="_fuel", values="_mw").sort_index()
    wide.columns.name = None
    if fuels is not None:
        want = [f.lower() for f in fuels]
        missing = [f for f in want if f not in wide.columns]
        if missing:
            raise DataError(f"fuel types {missing} not in gen_by_fuel; available: {list(wide.columns)}")
        wide = wide[want]
    log.info("gen_by_fuel: %d hours, fuels=%s", len(wide), list(wide.columns))
    return wide
