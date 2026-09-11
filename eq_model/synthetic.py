"""
Synthetic PJM-like data, in the *raw file formats* the loaders expect, plus a direct
in-memory panel generator for fast model tests.  Used by the test-suite and for smoke
runs when the real Data Miner / EIA files are not at hand.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .panel import TZ, HourlyPanel

ZONES = ["AE", "AEP", "APS", "ATSI", "BC", "CE", "DAY", "DEOK", "DOM", "DPL", "DUQ", "EKPC",
         "JC", "ME", "PE", "PEP", "PL", "PN", "PS", "RECO"]
REGION_OF = {z: ("MIDATL" if z in {"AE", "BC", "JC", "ME", "PE", "PEP", "PL", "PN", "PS", "RECO", "DPL"}
                 else "SOUTH" if z in {"DOM", "EKPC"} else "WEST") for z in ZONES}


def _fmt_dm(ts: pd.DatetimeIndex) -> np.ndarray:
    """Data Miner timestamp text, e.g. '1/1/2017 5:00:00 AM'."""
    return np.array([f"{t.month}/{t.day}/{t.year} {t.strftime('%I:%M:%S %p').lstrip('0')}" for t in ts])


def hourly_index(years: Sequence[int]) -> pd.DatetimeIndex:
    start = pd.Timestamp(f"{min(years)}-01-01 00:00", tz=TZ).tz_convert("UTC")
    end = pd.Timestamp(f"{max(years)}-12-31 23:00", tz=TZ).tz_convert("UTC")
    return pd.date_range(start, end, freq="h")


def synth_profiles(idx: pd.DatetimeIndex, seed: int = 0) -> Dict[str, np.ndarray]:
    """Load (MW), solar CF, wind CF for every hour of ``idx`` (UTC)."""
    rng = np.random.default_rng(seed)
    loc = idx.tz_convert(TZ)
    hod = loc.hour.values
    doy = loc.dayofyear.values
    n = len(idx)
    year_eff = np.array([(y - 2017) * 0.15 + rng.normal(0, 0.5) for y in loc.year.values])  # per-hour noise in temp
    temp = 12 - 14 * np.cos(2 * np.pi * (doy - 15) / 365) + 6 * np.sin(2 * np.pi * (hod - 9) / 24) + year_eff \
        + np.convolve(rng.normal(0, 1.2, n), np.ones(48) / 48, mode="same") * 6
    diurnal = 1.0 + 0.16 * np.sin(2 * np.pi * (hod - 10) / 24) + 0.05 * np.sin(4 * np.pi * (hod - 3) / 24)
    load = 88_000 * diurnal * (1 + 0.012 * np.clip(temp - 18, 0, None) ** 1.3 / 10 + 0.010 * np.clip(2 - temp, 0, None) / 4)
    load *= 1 + rng.normal(0, 0.015, n)
    # solar: clear-sky bell * seasonal * cloudiness
    daylen = 12 + 3 * np.sin(2 * np.pi * (doy - 80) / 365)
    x = (hod + 0.5 - 12.5) / (daylen / 2)
    clear = np.clip(np.cos(np.pi / 2 * x), 0, None) ** 1.3 * (0.7 + 0.3 * np.sin(2 * np.pi * (doy - 80) / 365))
    cloud = np.clip(1 - 0.6 * np.abs(np.convolve(rng.normal(0, 1, n), np.ones(12) / 12, mode="same")), 0.15, 1)
    solar = np.clip(clear * cloud * 0.95, 0, 1)
    # wind: AR(1) in logit space, higher in winter/night
    w = np.zeros(n)
    e = rng.normal(0, 0.35, n)
    for t in range(1, n):
        w[t] = 0.97 * w[t - 1] + e[t]
    season = 0.35 + 0.25 * np.cos(2 * np.pi * (doy - 20) / 365) - 0.06 * np.sin(2 * np.pi * (hod - 3) / 24)
    wind = 1 / (1 + np.exp(-(w * 0.9 + np.log(season / (1 - season)))))
    wind = np.clip(wind, 0.0, 1.0)
    return {"load": load, "solar": solar, "wind": wind}


def write_synthetic_raw(outdir: str, years: Sequence[int] = (2017, 2018, 2019), seed: int = 0,
                        solar_mw_start: float = 800.0, wind_mw_start: float = 8000.0) -> Dict[str, str]:
    """Write synthetic raw files mirroring the layout described by the user:
    data/raw/load/*.csv, data/raw/gen_by_fuel/*.csv, data/raw/capacity/{generators_active,
    generators_retired,vre_capacity}.csv.  Returns the directory paths."""
    rng = np.random.default_rng(seed + 1)
    idx = hourly_index(years)
    prof = synth_profiles(idx, seed)
    loc = idx.tz_convert(TZ)
    ept_txt = _fmt_dm(loc)
    utc_txt = _fmt_dm(idx)

    # monthly capacity (AC), grows over time; DC = 1.3 x AC for solar.
    # The fleet is built from explicit "big" (>10 MW) and "small" (5 MW) plants, and the
    # truth coverage series is accumulated directly from those sums, independently of eia.py.
    months = pd.period_range(f"{min(years)}-01", f"{max(years) + 1}-08", freq="M").to_timestamp()
    k = np.arange(len(months))
    nm = len(months)
    solar_big = np.zeros(nm); solar_small = np.zeros(nm)
    wind_big = np.zeros(nm); wind_small = np.zeros(nm)
    solar_big[0], solar_small[0] = solar_mw_start * 0.60, solar_mw_start * 0.40
    wind_big[0], wind_small[0] = wind_mw_start * 0.985, wind_mw_start * 0.015
    inc_s = solar_mw_start * 0.06 * (1.04 ** k)          # monthly additions
    inc_w = wind_mw_start * 0.006 * np.ones(nm)
    small_share = 0.35 - 0.25 * k / nm                    # small plants' share of additions falls over time
    for i in range(1, nm):
        s_small = 5.0 * np.round(inc_s[i] * small_share[i] / 5.0)      # multiples of 5 MW
        solar_small[i] = solar_small[i - 1] + s_small
        solar_big[i] = solar_big[i - 1] + (inc_s[i] - s_small)
        wind_small[i] = wind_small[i - 1]
        wind_big[i] = wind_big[i - 1] + inc_w[i]
    solar_mw = solar_big + solar_small
    wind_mw = wind_big + wind_small
    solar_cov = solar_big / solar_mw
    wind_cov = wind_big / wind_mw

    # --- load files ------------------------------------------------------
    load_dir = os.path.join(outdir, "load"); os.makedirs(load_dir, exist_ok=True)
    zshare = rng.dirichlet(np.ones(len(ZONES)) * 3)
    for y in years:
        m = loc.year.values == y
        rows = []
        for zi, z in enumerate(ZONES):
            mw = prof["load"][m] * zshare[zi] * (1 + rng.normal(0, 0.02, m.sum()))
            rows.append(pd.DataFrame({"datetime_beginning_utc": utc_txt[m], "datetime_beginning_ept": ept_txt[m],
                                      "nerc_region": "RFC", "mkt_region": REGION_OF[z], "zone": z, "load_area": z,
                                      "mw": np.round(mw, 3), "is_verified": "TRUE"}))
        zonal = pd.concat(rows)
        tot = zonal.groupby("datetime_beginning_utc", sort=False)["mw"].sum()
        # aggregate rows: three market regions + RTO total (this is the double-counting trap)
        for reg in ["MIDATL", "WEST", "SOUTH"]:
            sub = zonal[zonal["mkt_region"] == reg].groupby("datetime_beginning_utc", sort=False)["mw"].sum()
            rows.append(pd.DataFrame({"datetime_beginning_utc": sub.index, "datetime_beginning_ept": ept_txt[m],
                                      "nerc_region": "RFC", "mkt_region": reg, "zone": reg, "load_area": reg,
                                      "mw": np.round(sub.values, 3), "is_verified": "TRUE"}))
        rows.append(pd.DataFrame({"datetime_beginning_utc": tot.index, "datetime_beginning_ept": ept_txt[m],
                                  "nerc_region": "RFC", "mkt_region": "RTO", "zone": "RTO", "load_area": "RTO",
                                  "mw": np.round(tot.values * (1 + rng.normal(0, 0.001, len(tot))), 3),
                                  "is_verified": "TRUE"}))
        pd.concat(rows).to_csv(os.path.join(load_dir, f"hrl_load_metered_{y}.csv"), index=False)

    # --- gen_by_fuel files ----------------------------------------------
    gen_dir = os.path.join(outdir, "gen_by_fuel"); os.makedirs(gen_dir, exist_ok=True)
    mkey = pd.DatetimeIndex(pd.to_datetime({"year": loc.year, "month": loc.month, "day": 1}))
    cap_s = pd.Series(solar_mw, index=months).reindex(mkey).values
    cap_w = pd.Series(wind_mw, index=months).reindex(mkey).values
    cov_s = pd.Series(solar_cov, index=months).reindex(mkey).values
    cov_w = pd.Series(wind_cov, index=months).reindex(mkey).values
    gen_solar = prof["solar"] * cap_s * cov_s
    gen_wind = prof["wind"] * cap_w * cov_w
    for y in years:
        m = loc.year.values == y
        frames = []
        others = {"Gas": 0.40, "Nuclear": 0.33, "Coal": 0.20, "Hydro": 0.02, "Oil": 0.002, "Other": 0.003,
                  "Multiple Fuels": 0.005, "Other Renewables": 0.005, "Storage": 0.001}
        base = prof["load"][m] - gen_solar[m] - gen_wind[m]
        fuels = {"Solar": gen_solar[m] + rng.normal(0, 0.5, m.sum()) * (prof["solar"][m] == 0),   # tiny night noise
                 "Wind": gen_wind[m]}
        for f, sh in others.items():
            fuels[f] = base * sh
        total = sum(fuels.values())
        for f, v in fuels.items():
            frames.append(pd.DataFrame({"datetime_beginning_utc": utc_txt[m], "datetime_beginning_ept": ept_txt[m],
                                        "fuel_type": f, "mw": np.round(v, 3),
                                        "fuel_percentage_of_total": np.round(v / total, 4),
                                        "is_renewable": "TRUE" if f in {"Solar", "Wind", "Hydro", "Other Renewables"} else "FALSE"}))
        pd.concat(frames).to_csv(os.path.join(gen_dir, f"gen_by_fuel_{y}.csv"), index=False)

    # --- EIA-860M style inventories ----------------------------------------
    cap_dir = os.path.join(outdir, "capacity"); os.makedirs(cap_dir, exist_ok=True)
    gens = []
    pid = 1000

    def add_plant(tech, esc, pm, ba, cap_mw, op, ret=None, n_units=1, dc_ratio=None):
        nonlocal pid
        pid += 1
        for u in range(n_units):
            c = cap_mw / n_units
            gens.append({"Entity ID": 1, "Entity Name": "Synthetic LLC", "Plant ID": pid, "Plant Name": f"Plant {pid}",
                         "Sector": "IPP Non-CHP", "Plant State": "PA", "Generator ID": f"G{u+1}", "Unit Code": "",
                         "Nameplate Capacity (MW)": f"{c:,.1f}",                      # <-- comma formatted
                         "Net Summer Capacity (MW)": f"{c*0.95:,.1f}", "Net Winter Capacity (MW)": f"{c:,.1f}",
                         "Technology": tech, "Energy Source Code": esc, "Prime Mover Code": pm,
                         "Operating Month": op.month, "Operating Year": op.year,
                         "Retirement Month": ret.month if ret is not None else "", "Retirement Year": ret.year if ret is not None else "",
                         "Status": "(OP) Operating" if ret is None else "(RE) Retired",
                         "Nameplate Energy Capacity (MWh)": "", "DC Net Capacity (MW)": f"{c*dc_ratio:,.1f}" if dc_ratio else "",
                         "Latitude": 40.0, "Longitude": -78.0, "Balancing Authority Code": ba, "County": "X"})

    m0 = months[0]
    # initial fleet (existing before the first month): big plants (>10 MW) and 5 MW small plants
    add_plant("Solar Photovoltaic", "SUN", "PV", "PJM", solar_big[0], m0 - pd.DateOffset(months=6), n_units=3, dc_ratio=1.3)
    for _ in range(int(round(solar_small[0] / 5.0))):
        add_plant("Solar Photovoltaic", "SUN", "PV", "PJM", 5.0, m0 - pd.DateOffset(months=3), dc_ratio=1.3)
    add_plant("Onshore Wind Turbine", "WND", "WT", "PJM", wind_big[0], m0 - pd.DateOffset(years=2), n_units=4)  # "1,970.0"
    for _ in range(int(round(wind_small[0] / 5.0))):
        add_plant("Onshore Wind Turbine", "WND", "WT", "PJM", 5.0, m0 - pd.DateOffset(years=1))
    # growth: big additions as 1-2 unit plants, small additions as 5 MW plants
    for i in range(1, nm):
        big_s = solar_big[i] - solar_big[i - 1]
        n_small = int(round((solar_small[i] - solar_small[i - 1]) / 5.0))
        if big_s > 0:
            add_plant("Solar Photovoltaic", "SUN", "PV", "PJM", big_s, months[i], n_units=2, dc_ratio=1.3)
        for _ in range(n_small):
            add_plant("Solar Photovoltaic", "SUN", "PV", "PJM", 5.0, months[i], dc_ratio=1.3)
        add_plant("Onshore Wind Turbine", "WND", "WT", "PJM", wind_big[i] - wind_big[i - 1], months[i], n_units=2)
    # non-PJM and non-VRE noise rows
    add_plant("Solar Photovoltaic", "SUN", "PV", "MISO", 250.0, m0, dc_ratio=1.3)
    add_plant("Natural Gas Fired Combined Cycle", "NG", "CC", "PJM", 1281.0, m0 - pd.DateOffset(years=5), n_units=3)
    add_plant("Nuclear", "NUC", "ST", "PJM", 2300.0, m0 - pd.DateOffset(years=30), n_units=2)
    active = pd.DataFrame([g for g in gens if g["Status"].startswith("(OP)")])
    # a retired wind plant that was live during the first year only
    gens_ret = []
    gens.clear()
    add_plant("Onshore Wind Turbine", "WND", "WT", "PJM", 60.0, m0 - pd.DateOffset(years=10), ret=months[11])
    retired = pd.DataFrame(gens)
    active.drop(columns=["Retirement Month", "Retirement Year"]).to_csv(os.path.join(cap_dir, "generators_active_07_2026.csv"), index=False)
    retired.to_csv(os.path.join(cap_dir, "generators_retired_07_2026.csv"), index=False)
    # derived vre_capacity.csv (as if from the user's build_vre_capacity.py) - WITHOUT coverage
    pd.DataFrame({"month": months.strftime("%Y-%m"), "solar_mw": np.round(solar_mw, 1), "wind_mw": np.round(wind_mw, 1)}) \
        .to_csv(os.path.join(cap_dir, "vre_capacity.csv"), index=False)
    truth = pd.DataFrame({"month": months, "solar_mw": solar_mw, "wind_mw": wind_mw,
                          "solar_coverage": solar_cov, "wind_coverage": wind_cov})
    truth.to_csv(os.path.join(cap_dir, "_synthetic_truth.csv"), index=False)
    return {"load": load_dir, "gen_by_fuel": gen_dir, "capacity": cap_dir}


def make_synthetic_panel(years: Sequence[int] = (2017, 2018, 2019), H: Optional[int] = None,
                         seed: int = 0, params=None) -> HourlyPanel:
    """In-memory panel (no files).  ``H`` < 8760 keeps the first H hours of each year
    (chronological, storage-consistent) for fast tests."""
    from .params import ModelParams
    params = params or ModelParams()
    years = [int(y) for y in years]
    idx = hourly_index(years)
    loc = idx.tz_convert(TZ)
    keep = ~((loc.month == 2) & (loc.day == 29))
    idx, loc = idx[keep], loc[keep]
    prof = synth_profiles(idx, seed)
    Yn = len(years)
    Hfull = 8760
    D = prof["load"].reshape(Yn, Hfull)
    hod = loc.hour.values.reshape(Yn, Hfull)
    theta = {"solar": prof["solar"].reshape(Yn, Hfull), "wind": prof["wind"].reshape(Yn, Hfull)}
    utc = idx.tz_localize(None).values.reshape(Yn, Hfull)
    if H is not None and H < Hfull:
        D, hod, utc = D[:, :H], hod[:, :H], utc[:, :H]
        theta = {k: v[:, :H] for k, v in theta.items()}
    prof24 = pd.Series(D.ravel()).groupby(hod.ravel()).mean().reindex(range(24)).values
    lam = prof24 / np.nanmax(prof24)
    daylight = theta["solar"] > params.daylight_theta_threshold
    return HourlyPanel(years=np.array(years), utc=utc, hour_of_day=hod, D=D, theta=theta, lam=lam,
                       daylight=daylight, meta={"years": years, "mean_profile_mw": prof24.tolist(), "synthetic": True})
