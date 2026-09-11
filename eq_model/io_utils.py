"""
Small, defensive I/O helpers shared by the PJM and EIA loaders.

The key one is :func:`parse_number`, which converts strings such as ``"1,281.0"``
to floats and *refuses* to let a non-blank value silently become NaN/0.
"""
from __future__ import annotations

import glob
import logging
import os
import re
from typing import Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

log = logging.getLogger("eq_model")


class DataError(RuntimeError):
    pass


def list_csvs(path: str, pattern: str = "*.csv") -> List[str]:
    if os.path.isfile(path):
        return [path]
    files = sorted(glob.glob(os.path.join(path, pattern)))
    if not files:
        raise DataError(f"no files matching {pattern!r} under {path!r}")
    return files


def read_csv_stack(path: str, pattern: str = "*.csv", **kw) -> pd.DataFrame:
    """Read every CSV under ``path`` as *strings* and concatenate.

    Reading as strings is deliberate: pandas' default parser would turn a column
    containing ``"1,281.0"`` into object dtype in some files and float in others,
    and downstream ``astype(float)`` / ``fillna(0)`` calls then hide the problem.
    All numeric conversion goes through :func:`parse_number`.
    """
    files = list_csvs(path, pattern)
    frames = []
    for f in files:
        df = pd.read_csv(f, dtype=str, keep_default_na=False, na_values=[], **kw)
        df["__source_file"] = os.path.basename(f)
        frames.append(df)
        log.info("read %s: %d rows, %d cols", os.path.basename(f), len(df), df.shape[1])
    out = pd.concat(frames, ignore_index=True)
    out.columns = [c.strip() for c in out.columns]
    return out


_BLANK = {"", "na", "n/a", "nan", "null", "none", "-", "--", "."}


def parse_number(s: pd.Series, name: str = "", allow_blank: bool = True,
                 strict: bool = True) -> pd.Series:
    """Convert a string Series to float, handling thousands separators.

    * ``"1,281.0"`` -> 1281.0 ; ``"(12.5)"`` -> -12.5 ; ``" 3 "`` -> 3.0
    * blank / NA tokens -> NaN (never 0)
    * any *other* unparsable token raises :class:`DataError` when ``strict``
      (otherwise logs a warning and yields NaN).

    Returns a float Series aligned with ``s``.
    """
    if s.dtype.kind in "fiu":
        return s.astype(float)
    raw = s.astype(str).str.strip()
    is_blank = raw.str.lower().isin(_BLANK)
    cleaned = (raw.str.replace(",", "", regex=False)
                  .str.replace("$", "", regex=False)
                  .str.replace(r"^\((.*)\)$", r"-\1", regex=True))
    out = pd.to_numeric(cleaned, errors="coerce")
    bad = out.isna() & ~is_blank
    if bad.any():
        examples = raw[bad].unique()[:5].tolist()
        msg = (f"parse_number[{name}]: {int(bad.sum())} non-blank values could not be parsed, "
               f"e.g. {examples}")
        if strict:
            raise DataError(msg)
        log.warning(msg)
    if not allow_blank and is_blank.any():
        raise DataError(f"parse_number[{name}]: {int(is_blank.sum())} blank values where none allowed")
    n_comma = int(raw.str.contains(",", regex=False).sum())
    if n_comma:
        log.info("parse_number[%s]: %d values contained thousands separators (handled)", name, n_comma)
    return out.astype(float)


def find_col(df: pd.DataFrame, candidates: Sequence[str], required: bool = True,
             contains: bool = False) -> Optional[str]:
    """Case/space-insensitive column lookup. ``contains`` allows substring matches."""
    norm = {re.sub(r"[^a-z0-9]", "", c.lower()): c for c in df.columns}
    for cand in candidates:
        key = re.sub(r"[^a-z0-9]", "", cand.lower())
        if key in norm:
            return norm[key]
    if contains:
        for cand in candidates:
            key = re.sub(r"[^a-z0-9]", "", cand.lower())
            for k, c in norm.items():
                if key in k:
                    return c
    if required:
        raise DataError(f"none of the columns {list(candidates)} found; have {list(df.columns)}")
    return None


def parse_datetime(s: pd.Series, name: str = "", utc: bool = False) -> pd.Series:
    """Parse Data Miner style timestamps ('1/1/2017 5:00:00 AM' or ISO)."""
    raw = s.astype(str).str.strip()
    fmts = ["%m/%d/%Y %I:%M:%S %p", "%m/%d/%Y %H:%M:%S", "%m/%d/%Y %H:%M", "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M:%SZ"]
    for fmt in fmts:
        try:
            out = pd.to_datetime(raw, format=fmt)
            break
        except (ValueError, TypeError):
            continue
    else:
        try:
            out = pd.to_datetime(raw, format="mixed")
        except Exception as e:  # pragma: no cover
            raise DataError(f"could not parse datetime column {name!r}: {e}")
    if out.isna().any():
        raise DataError(f"{int(out.isna().sum())} unparsable timestamps in {name!r}")
    if utc:
        out = out.dt.tz_localize("UTC") if out.dt.tz is None else out.dt.tz_convert("UTC")
    return out


def normalize_label(x) -> str:
    """'PJM RTO' -> 'PJMRTO', 'Mid-Atl' -> 'MIDATL' (used for area classification)."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    return re.sub(r"[^A-Z0-9]", "", str(x).upper())
