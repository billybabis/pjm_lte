"""
Assemble the hourly model panel: D_hy, theta_rhy, lambda_h, daylight_hy.

Conventions (parameters.tex "Conventions"):
* years are *local* (America/New_York) calendar years, each exactly 8760 hours
  (Feb 29 of leap years is dropped by default);
* hours are chronological in UTC, so DST transitions do not create gaps/duplicates;
* lambda_h is the mean intraday load profile by local hour-of-day, normalised to max 1;
* theta_rhy = gen_rhy / (cap_ry * coverage_ry), clipped to [0, 1] with a report.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .io_utils import DataError
from .params import ModelParams

log = logging.getLogger("eq_model")
TZ = "America/New_York"


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return str(o)


@dataclass
class HourlyPanel:
    years: np.ndarray                 # (Y,) int
    utc: np.ndarray                   # (Y, H) datetime64[ns] (UTC)
    hour_of_day: np.ndarray           # (Y, H) local hour 0..23
    D: np.ndarray                     # (Y, H) load, MW
    theta: Dict[str, np.ndarray]      # vre_key -> (Y, H) availability in [0,1]
    lam: np.ndarray                   # (24,) contract shape, max 1
    daylight: np.ndarray              # (Y, H) bool
    meta: dict = field(default_factory=dict)

    @property
    def Y(self) -> int:
        return len(self.years)

    @property
    def H(self) -> int:
        return self.D.shape[1]

    @property
    def lam_hy(self) -> np.ndarray:
        """lambda mapped onto every (y,h): (Y,H)."""
        return self.lam[self.hour_of_day]

    @property
    def D_max(self) -> float:
        return float(self.D.max())

    def q_bar(self, params: ModelParams) -> float:
        """Total contracted capacity Q_bar (MW) - see params.q_bar_rule."""
        rule = params.q_bar_rule
        if rule == "fixed":
            if params.q_bar_mw is None:
                raise ValueError("q_bar_rule='fixed' requires q_bar_mw")
            return float(params.q_bar_mw)
        if rule == "mean_profile_peak":
            ref = float(np.max(self.meta["mean_profile_mw"]))     # max_h Dbar_h
        elif rule == "peak_load":
            ref = self.D_max
        elif rule == "mean_load":
            ref = float(self.D.mean())
        else:
            raise ValueError(f"unknown q_bar_rule {rule!r}")
        return params.q_bar_coverage * ref

    def subset(self, years: Sequence[int]) -> "HourlyPanel":
        idx = [int(np.where(self.years == y)[0][0]) for y in years]
        return HourlyPanel(years=self.years[idx], utc=self.utc[idx], hour_of_day=self.hour_of_day[idx],
                           D=self.D[idx], theta={k: v[idx] for k, v in self.theta.items()},
                           lam=self.lam, daylight=self.daylight[idx], meta=dict(self.meta))

    def thin(self, step: int) -> "HourlyPanel":
        """Keep every ``step``-th hour (for quick smoke tests only - breaks storage chronology)."""
        sl = slice(None, None, step)
        return HourlyPanel(years=self.years, utc=self.utc[:, sl], hour_of_day=self.hour_of_day[:, sl],
                           D=self.D[:, sl], theta={k: v[:, sl] for k, v in self.theta.items()},
                           lam=self.lam, daylight=self.daylight[:, sl], meta=dict(self.meta, thinned=step))

    def summary(self) -> str:
        lines = [f"Panel: Y={self.Y} years {[int(y) for y in self.years]}, H={self.H} hours/yr, "
                 f"D_max={self.D_max:,.0f} MW, mean load={self.D.mean():,.0f} MW"]
        for k, th in self.theta.items():
            cf = th.mean(axis=1)
            lines.append(f"  theta[{k}]: annual mean CF by year = {np.round(cf, 3).tolist()}")
        lines.append(f"  daylight share = {self.daylight.mean():.3f}; lambda_h = {np.round(self.lam, 3).tolist()}")
        for k, v in self.meta.items():
            if k.startswith("theta_clip") or k.startswith("gap"):
                lines.append(f"  {k}: {v}")
        return "\n".join(lines)

    # persistence -----------------------------------------------------------
    def save(self, path: str) -> None:
        import json
        np.savez_compressed(path, years=self.years, utc=self.utc.astype("datetime64[ns]"),
                            hour_of_day=self.hour_of_day, D=self.D, lam=self.lam, daylight=self.daylight,
                            theta_keys=np.array(list(self.theta.keys())),
                            **{f"theta_{k}": v for k, v in self.theta.items()},
                            meta=np.array([json.dumps(self.meta, default=_json_default)]))

    @classmethod
    def load(cls, path: str) -> "HourlyPanel":
        import json
        z = np.load(path, allow_pickle=False)
        keys = list(z["theta_keys"])
        meta = {}
        try:
            meta = json.loads(str(z["meta"][0]))
        except Exception:
            pass
        return cls(years=z["years"], utc=z["utc"], hour_of_day=z["hour_of_day"], D=z["D"],
                   theta={k: z[f"theta_{k}"] for k in keys}, lam=z["lam"], daylight=z["daylight"], meta=meta)


def _local_frame(index_utc: pd.DatetimeIndex) -> pd.DataFrame:
    loc = index_utc.tz_convert(TZ)
    return pd.DataFrame({"year": loc.year, "month": loc.month, "day": loc.day, "hour": loc.hour}, index=index_utc)


def _complete_hourly_index(years: Sequence[int]) -> pd.DatetimeIndex:
    """Every hour (UTC) of each requested *local* calendar year, concatenated in order.
    Works for non-contiguous year lists (e.g. leave-one-year-out sensitivities)."""
    parts = []
    for y in sorted(set(int(v) for v in years)):
        start = pd.Timestamp(f"{y}-01-01 00:00", tz=TZ).tz_convert("UTC")
        end = pd.Timestamp(f"{y}-12-31 23:00", tz=TZ).tz_convert("UTC")
        parts.append(pd.date_range(start, end, freq="h"))
    return parts[0].append(parts[1:]) if len(parts) > 1 else parts[0]


def _fill_gaps(s: pd.Series, full_index: pd.DatetimeIndex, name: str, max_gap: int = 6) -> "tuple[pd.Series, dict]":
    s = s.reindex(full_index)
    n_missing = int(s.isna().sum())
    info = {"n_missing": n_missing}
    if n_missing:
        # longest run of NaNs
        isna = s.isna().values.astype(int)
        runs, cur = [], 0
        for v in isna:
            cur = cur + 1 if v else 0
            runs.append(cur)
        info["longest_gap_h"] = int(max(runs))
        if info["longest_gap_h"] > max_gap:
            log.warning("%s: %d missing hours, longest gap %d h (> %d) - linearly interpolated; check the data!",
                        name, n_missing, info["longest_gap_h"], max_gap)
        else:
            log.info("%s: %d missing hours interpolated (longest gap %d h)", name, n_missing, info["longest_gap_h"])
        s = s.interpolate(limit_direction="both")
    return s, info


def build_panel(load: pd.Series, gen: pd.DataFrame, capacity: pd.DataFrame, params: ModelParams,
                years: Optional[Sequence[int]] = None, drop_feb29: bool = True,
                gen_columns: Optional[Dict[str, str]] = None, coverage_default: float = 1.0,
                theta_clip_max: float = 1.0) -> HourlyPanel:
    """Combine hourly load (UTC index), hourly generation by fuel (UTC index, columns incl.
    'solar','wind'), and monthly capacity (columns month, solar_mw, wind_mw[, *_coverage]).
    """
    gen_columns = gen_columns or {"solar": "solar", "wind": "wind"}
    loc = _local_frame(load.index)
    all_years = sorted(loc["year"].unique().tolist())
    if years is None:
        # keep only complete local years (>= 8700 hours present)
        counts = loc.groupby("year").size()
        years = [y for y in all_years if counts.get(y, 0) >= 8700]
        dropped = sorted(set(all_years) - set(years))
        if dropped:
            log.info("build_panel: incomplete years dropped: %s", dropped)
    years = [int(y) for y in years]
    full = _complete_hourly_index(years)

    D, gap_load = _fill_gaps(load, full, "load")
    th_series: Dict[str, pd.Series] = {}
    gaps = {"load": gap_load}
    cap = capacity.copy()
    cap["month"] = pd.to_datetime(cap["month"]).dt.to_period("M").dt.to_timestamp()
    cap = cap.set_index("month").sort_index()
    locf = _local_frame(full)
    month_key = pd.to_datetime({"year": locf["year"], "month": locf["month"], "day": 1}).values
    clip_info = {}
    for key, col in gen_columns.items():
        if col not in gen.columns:
            raise DataError(f"gen_by_fuel lacks column {col!r} for {key}; have {list(gen.columns)}")
        g, gap_g = _fill_gaps(gen[col], full, f"gen[{col}]")
        gaps[f"gen_{key}"] = gap_g
        cap_col, cov_col = f"{key}_mw", f"{key}_coverage"
        if cap_col not in cap.columns:
            raise DataError(f"capacity table lacks {cap_col!r}")
        cap_m = cap[cap_col].reindex(pd.DatetimeIndex(month_key))
        if cap_m.isna().any():
            missing = sorted(set(pd.DatetimeIndex(month_key)[cap_m.isna().values].strftime("%Y-%m")))
            raise DataError(f"capacity table has no {cap_col} for months {missing[:6]}{'...' if len(missing) > 6 else ''}")
        if cov_col in cap.columns and cap[cov_col].notna().any():
            cov_m = cap[cov_col].reindex(pd.DatetimeIndex(month_key)).ffill().bfill()
        else:
            log.warning("no %s in capacity table - using coverage=%.3f (theta will be biased low if PJM's "
                        "gen_by_fuel omits small plants)", cov_col, coverage_default)
            cov_m = pd.Series(coverage_default, index=pd.DatetimeIndex(month_key))
        denom = cap_m.values * cov_m.values
        raw = g.values / np.where(denom > 0, denom, np.nan)
        n_neg = int((raw < 0).sum())
        n_over = int((raw > theta_clip_max).sum())
        n_nan = int(np.isnan(raw).sum())
        clip_info[f"theta_clip_{key}"] = {"n_negative_set_0": n_neg, f"n_over_{theta_clip_max}_clipped": n_over,
                                          "n_zero_capacity_set_0": n_nan, "max_raw": float(np.nanmax(raw))}
        th = np.clip(np.nan_to_num(raw, nan=0.0), 0.0, theta_clip_max)
        th_series[key] = pd.Series(th, index=full)

    # drop Feb 29 (local) -> exactly 8760 hours per local year
    keep = np.ones(len(full), dtype=bool)
    if drop_feb29:
        keep &= ~((locf["month"].values == 2) & (locf["day"].values == 29))
    idx = full[keep]
    locf = locf.loc[idx]
    D = D.loc[idx]
    th_series = {k: v.loc[idx] for k, v in th_series.items()}

    years = sorted(years)
    Yn = len(years)
    counts = locf.groupby("year").size()
    if sorted(counts.index.tolist()) != years:
        raise DataError(f"panel years {sorted(counts.index.tolist())} != requested {years}")
    if drop_feb29 and not all(counts.get(y, 0) == 8760 for y in years):
        raise DataError(f"expected 8760 hours per local year after dropping Feb 29; got {counts.to_dict()}")
    if counts.nunique() != 1:
        raise DataError(f"years have different hour counts {counts.to_dict()}; use drop_feb29=True")
    H = int(counts.iloc[0])
    D_arr = D.values.reshape(Yn, H)
    hod = locf["hour"].values.reshape(Yn, H)
    utc_arr = idx.tz_convert("UTC").tz_localize(None).values.reshape(Yn, H)
    theta = {k: v.values.reshape(Yn, H) for k, v in th_series.items()}

    # contract shape lambda_h (24 values): mean intraday profile normalised to max 1
    prof = pd.Series(D_arr.ravel()).groupby(hod.ravel()).mean()
    prof = prof.reindex(range(24)).values
    lam = prof / prof.max()

    # daylight indicator
    if params.daylight_rule == "solar_threshold":
        if "solar" not in theta:
            raise DataError("daylight_rule='solar_threshold' needs theta['solar']")
        daylight = theta["solar"] > params.daylight_theta_threshold
    elif params.daylight_rule == "fixed_hours":
        a, b = params.daylight_fixed_hours
        daylight = (hod >= a) & (hod < b)
    else:
        raise ValueError(params.daylight_rule)

    meta = {"years": years, "mean_profile_mw": prof.tolist(), "gaps": gaps, **clip_info,
            "drop_feb29": drop_feb29, "coverage_used": {k: bool(f"{k}_coverage" in cap.columns and cap[f"{k}_coverage"].notna().any()) for k in gen_columns}}
    panel = HourlyPanel(years=np.array(years), utc=utc_arr, hour_of_day=hod, D=D_arr, theta=theta,
                        lam=lam, daylight=daylight, meta=meta)
    log.info("\n%s", panel.summary())
    return panel
