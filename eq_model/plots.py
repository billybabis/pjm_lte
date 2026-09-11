"""
Publication figures from a directory (or several) of solved results.

Two figure types, both laid out as one panel per gamma, side by side:

``capacity_panels``  stacked installed capacity K_z by regime, curtailment *hours* overlaid.
``energy_panels``    stacked share of load served by technology, curtailment *share* overlaid.
``contract_panels``  stacked contracted capacity chi_z*K_z, forward price p_hat overlaid.

They answer different questions and should be read together.  Nameplate MW are not comparable
across technologies with different capacity factors, so the total height of a capacity bar is
not a meaningful quantity -- a mix that swaps wind for gas gets shorter without serving less
load.  The energy panels are the ones whose columns genuinely sum to 100 % of load.

Reads the ``row_<regime>.json`` files written by ``run``/``sweep`` (see cli.py), so it works
equally on a local run and on a directory of artifacts downloaded from the CI matrix.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

log = logging.getLogger("eq_model")

# Colour-blind-safe qualitative palette (Okabe-Ito), assigned by technology so that a
# technology keeps its colour across every figure in the paper.
TECH_COLORS: Dict[str, str] = {
    "nuclear":  "#CC79A7",
    "coal":     "#000000",
    "ccgt":     "#0072B2",
    "ct":       "#56B4E9",
    "solar":    "#E69F00",
    "wind":     "#009E73",
    "storage4": "#D55E00",
    "storage8": "#8C4A00",
}
# Stacking order, bottom to top: firm thermal, then variable, then storage.
TECH_ORDER: List[str] = ["nuclear", "coal", "ccgt", "ct", "solar", "wind", "storage4", "storage8"]

TECH_LABELS: Dict[str, str] = {
    "nuclear": "Nuclear", "coal": "Coal", "ccgt": "CCGT", "ct": "CT",
    "solar": "Solar", "wind": "Wind", "storage4": "Storage (4 h)", "storage8": "Storage (8 h)",
}

CURT_COLOR = "#444444"
LOST_LOAD_COLOR = "#B22222"       # unserved energy: a share of load that no technology served


def _require_mpl():
    try:
        import matplotlib
    except ImportError as e:                                   # pragma: no cover
        raise SystemExit("plotting needs matplotlib: pip install matplotlib") from e
    matplotlib.use("Agg")                                      # no display on CI runners
    import matplotlib.pyplot as plt
    return plt


def _tech_columns(df: pd.DataFrame) -> List[str]:
    """Technologies present in the table, in TECH_ORDER, dropping any that are zero everywhere."""
    present = [z for z in TECH_ORDER if f"K_{z}_MW" in df.columns]
    present += [c[2:-3] for c in df.columns
                if c.startswith("K_") and c.endswith("_MW") and c[2:-3] not in present]
    keep = []
    for z in present:
        k = df[f"K_{z}_MW"].to_numpy(dtype=float)
        if np.nanmax(np.abs(k)) > 1.0:                         # ignore numerically-zero technologies
            keep.append(z)
    return keep


def _gamma_col(df: pd.DataFrame) -> str:
    """Group panels by the gamma the run was launched with.  The planner regimes P1/P2 are
    risk-neutral by construction and report gamma=0 in every run, so grouping on their own
    ``gamma`` would drop them from every panel but the first."""
    return "gamma_requested" if "gamma_requested" in df.columns else "gamma"


def _curtailment_hours(df: pd.DataFrame) -> Optional[np.ndarray]:
    if "curt_hours_any_mean" in df.columns:
        return df["curt_hours_any_mean"].to_numpy(dtype=float)
    per_tech = [c for c in df.columns if c.startswith("curt_hours_") and c.endswith("_mean")
                and c != "curt_hours_any_mean"]
    if per_tech:
        # No union available (older results): the max over technologies is the tightest
        # lower bound on the number of hours with any curtailment.
        log.warning("no curt_hours_any_mean column; using max over %s as a lower bound", per_tech)
        return df[per_tech].max(axis=1).to_numpy(dtype=float)
    return None


def _panel_figure(n: int, panel_width: float, height: float):
    plt = _require_mpl()
    fig, axes = plt.subplots(1, n, figsize=(panel_width * n, height), sharey=True, squeeze=False)
    return fig, axes[0]


def _style_panel(ax, x, labels, g: float):
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_title(rf"$\gamma = {g:g}$", fontsize=11)
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def _panel_legend(fig, handles, labels, title: Optional[str]):
    ncol = min(len(labels), 5)
    fig.legend(handles, labels, loc="lower center", ncol=ncol, frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, -0.02))
    if title:
        fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0.08, 1, 0.97 if title else 1.0))


def _panel_setup(df: pd.DataFrame, gammas, regimes):
    """Shared filtering/ordering for both figure types."""
    if regimes is not None:
        df = df[df["regime"].isin(list(regimes))]
    if not len(df):
        raise SystemExit("nothing to plot after filtering")
    gcol = _gamma_col(df)
    gs = list(gammas) if gammas is not None else sorted(df[gcol].unique())
    missing = [g for g in gs if not (df[gcol] == g).any()]
    if missing:
        raise SystemExit(f"no rows for gamma={missing}; present: {sorted(df[gcol].unique())}")
    bar_order = list(regimes) if regimes is not None else list(pd.unique(df["regime"].astype(str)))
    return df, gcol, gs, bar_order


def _panel_rows(df: pd.DataFrame, gcol: str, g: float, bar_order: List[str]) -> pd.DataFrame:
    sub = df[df[gcol] == g].copy()
    sub["regime"] = sub["regime"].astype(str)
    return sub.set_index("regime").reindex(bar_order)      # missing regimes keep their slot


def energy_panels(df: pd.DataFrame, gammas: Optional[Sequence[float]] = None,
                  regimes: Optional[Sequence[str]] = None, out: str = "energy.pdf",
                  title: Optional[str] = None, panel_width: float = 3.4, height: float = 4.0,
                  dpi: int = 300, show_curtailment: bool = True):
    """Share of load served, stacked by technology, one panel per gamma.

    Columns sum to 100 % of load: every technology's generation plus any unserved energy.
    Storage is *net* of round-trip losses, so its segment is negative and is drawn below the
    axis rather than stacked on top -- charging consumes more than discharging returns.
    Curtailed VRE energy (as a share of what was available) goes on the secondary axis.
    """
    plt = _require_mpl()
    df, gcol, gs, bar_order = _panel_setup(df, gammas, regimes)

    techs = [z for z in TECH_ORDER if f"share_{z}" in df.columns]
    techs += [c[6:] for c in df.columns if c.startswith("share_") and c[6:] not in techs
              and c[6:] != "lost_load"]
    techs = [z for z in techs if np.nanmax(np.abs(df[f"share_{z}"].to_numpy(dtype=float))) > 1e-4]
    has_ll = ("share_lost_load" in df.columns
              and np.nanmax(df["share_lost_load"].to_numpy(dtype=float)) > 1e-6)

    fig, axes = _panel_figure(len(gs), panel_width, height)
    curt_col = "curt_share_vre" if "curt_share_vre" in df.columns else None
    curt_max = float(np.nanmax(df[curt_col].to_numpy(dtype=float))) * 100 if (
        curt_col and show_curtailment and df[curt_col].notna().any()) else 0.0
    twins = []

    for ax, g in zip(axes, gs):
        sub = _panel_rows(df, gcol, g, bar_order)
        x = np.arange(len(sub))
        pos = np.zeros(len(sub)); neg = np.zeros(len(sub))
        for z in techs + (["lost_load"] if has_ll else []):
            v = np.nan_to_num(sub[f"share_{z}"].to_numpy(dtype=float), nan=0.0) * 100.0
            color = LOST_LOAD_COLOR if z == "lost_load" else TECH_COLORS.get(z, "#999999")
            label = "Unserved energy" if z == "lost_load" else TECH_LABELS.get(z, z)
            up, dn = np.clip(v, 0, None), np.clip(v, None, 0)
            ax.bar(x, up, bottom=pos, width=0.68, color=color, edgecolor="white",
                   linewidth=0.4, label=label, zorder=2)
            pos += up
            if (dn < 0).any():                             # storage net consumption
                ax.bar(x, dn, bottom=neg, width=0.68, color=color, edgecolor="white",
                       linewidth=0.4, zorder=2)
                neg += dn
        ax.axhline(0, color="#333333", linewidth=0.8, zorder=3)
        _style_panel(ax, x, sub.index, g)

        if show_curtailment and curt_col is not None:
            curt = np.nan_to_num(sub[curt_col].to_numpy(dtype=float), nan=0.0) * 100.0
            tw = ax.twinx()
            tw.plot(x, curt, linestyle="none", marker="D", markersize=5.5, color=CURT_COLOR,
                    markeredgecolor="white", markeredgewidth=0.7, zorder=4, clip_on=False)
            top = max(curt_max * 1.25, 1.0)
            tw.set_ylim(-0.045 * top, top)                 # keep a zero marker off the tick labels
            tw.spines["top"].set_visible(False)
            twins.append(tw)

    axes[0].set_ylabel("Share of load served (%)")
    for tw in twins[:-1]:
        tw.set_yticklabels([])
    if twins:
        twins[-1].set_ylabel("VRE curtailed (% of available)", color=CURT_COLOR)
        twins[-1].tick_params(axis="y", colors=CURT_COLOR)

    handles, labels = axes[0].get_legend_handles_labels()
    if twins:
        from matplotlib.lines import Line2D
        handles.append(Line2D([], [], linestyle="none", marker="D", markersize=5.5, color=CURT_COLOR))
        labels.append("Curtailed VRE (right axis)")
    _panel_legend(fig, handles, labels, title)
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    log.info("wrote %s (%d panel(s), %d technologies)", out, len(gs), len(techs))
    return out


def contract_panels(df: pd.DataFrame, gammas: Optional[Sequence[float]] = None,
                    regimes: Optional[Sequence[str]] = None, out: str = "contracts.pdf",
                    title: Optional[str] = None, capacity_unit: str = "GW",
                    panel_width: float = 3.4, height: float = 4.0, dpi: int = 300,
                    show_price: bool = True, show_empty: bool = False):
    """Contracted capacity Q_z = chi_z * K_z, stacked by technology, one panel per gamma.

    The forward market clears at sum_z Q_z = Q_bar, so every contracting regime's bar has the
    same height and the figure is about the *mix*: which technologies take the fixed pool of
    contracts, and how that reallocates as agents become more risk averse.  The forward
    clearing price p_hat goes on the secondary axis.

    Planner (P1, P2) and merchant (R2, R6) regimes have no forward market -- chi_z = 0 by
    construction -- so they carry no information here and are omitted.  ``show_empty=True``
    keeps their slots, labelled, when the regime axis must line up with the capacity and
    energy figures.
    """
    plt = _require_mpl()
    df, gcol, gs, bar_order = _panel_setup(df, gammas, regimes)

    chi_techs = [z for z in TECH_ORDER if f"chi_{z}" in df.columns and f"K_{z}_MW" in df.columns]
    if not chi_techs:
        raise SystemExit("no chi_* columns found; these results predate the contract reporting")
    scale = 1e-3 if capacity_unit.upper() == "GW" else 1.0
    q = {z: (df[f"chi_{z}"].to_numpy(dtype=float) * df[f"K_{z}_MW"].to_numpy(dtype=float))
         for z in chi_techs}
    chi_techs = [z for z in chi_techs if np.nanmax(np.abs(q[z])) > 1.0]
    q_bar = float(np.nanmax(df["Q_bar_MW"].to_numpy(dtype=float))) * scale if "Q_bar_MW" in df else 0.0

    # Planner and merchant regimes have no forward market at all, so a bar for them would be
    # empty by construction rather than by outcome.  Drop them unless asked to keep the slot.
    if not show_empty:
        contracted = df.assign(_q=sum(q[z] for z in chi_techs)).groupby(
            df["regime"].astype(str))["_q"].max()
        keep = [r for r in bar_order if contracted.get(r, 0.0) > 1.0]
        dropped = [r for r in bar_order if r not in keep]
        if dropped:
            log.info("contract figure: no forward market in %s; omitted (pass show_empty to keep "
                     "the slots)", ", ".join(dropped))
        if not keep:
            raise SystemExit("no regime in this selection has a forward market (chi = 0 everywhere); "
                             "the contract figure needs R3, R4 or R5")
        bar_order = keep

    fig, axes = _panel_figure(len(gs), panel_width, height)
    p_all = df["p_hat"].to_numpy(dtype=float) if ("p_hat" in df.columns and show_price) else None
    p_max = float(np.nanmax(p_all)) if p_all is not None and np.isfinite(p_all).any() else 0.0
    twins = []

    for ax, g in zip(axes, gs):
        sub = _panel_rows(df, gcol, g, bar_order)
        x = np.arange(len(sub))
        bottom = np.zeros(len(sub))
        for z in chi_techs:
            v = np.nan_to_num(sub[f"chi_{z}"].to_numpy(dtype=float), nan=0.0) *                 np.nan_to_num(sub[f"K_{z}_MW"].to_numpy(dtype=float), nan=0.0) * scale
            v = np.clip(v, 0.0, None)
            ax.bar(x, v, bottom=bottom, width=0.68, color=TECH_COLORS.get(z, "#999999"),
                   edgecolor="white", linewidth=0.4, label=TECH_LABELS.get(z, z), zorder=2)
            bottom += v
        if q_bar > 0:
            ax.axhline(q_bar, color="#333333", linewidth=0.9, linestyle=(0, (4, 3)), zorder=3)
        # Say why a bar is empty: no forward market is a regime property, not missing data.
        for i, r in enumerate(sub.index):
            if bottom[i] <= 1e-9 * max(q_bar, 1.0):
                ax.text(x[i], q_bar * 0.5 if q_bar else 0.5, "no forward market", rotation=90,
                        ha="center", va="center", fontsize=7.5, color="#888888", zorder=4)
        _style_panel(ax, x, sub.index, g)

        if p_all is not None:
            tw = ax.twinx()
            ph = sub["p_hat"].to_numpy(dtype=float)
            tw.plot(x, np.where(np.isfinite(ph), ph, np.nan), linestyle="none", marker="o",
                    markersize=5.0, color=CURT_COLOR, markeredgecolor="white",
                    markeredgewidth=0.7, zorder=4, clip_on=False)
            top = max(p_max * 1.25, 1.0)
            tw.set_ylim(-0.045 * top, top)
            tw.spines["top"].set_visible(False)
            twins.append(tw)

    axes[0].set_ylabel(f"Contracted capacity $Q_z$ ({capacity_unit})")
    for tw in twins[:-1]:
        tw.set_yticklabels([])
    if twins:
        twins[-1].set_ylabel(r"Forward price $\hat{p}$ (\$/MWh)", color=CURT_COLOR)
        twins[-1].tick_params(axis="y", colors=CURT_COLOR)

    handles, labels = axes[0].get_legend_handles_labels()
    from matplotlib.lines import Line2D
    if q_bar > 0:
        handles.append(Line2D([], [], color="#333333", linewidth=0.9, linestyle=(0, (4, 3))))
        labels.append(r"$\bar{Q}$" + f" = {q_bar:,.0f} {capacity_unit} (contract pool)")
    if twins:
        handles.append(Line2D([], [], linestyle="none", marker="o", markersize=5.0, color=CURT_COLOR))
        labels.append(r"$\hat{p}$ (right axis)")
    _panel_legend(fig, handles, labels, title)
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    log.info("wrote %s (%d panel(s), %d technologies)", out, len(gs), len(chi_techs))
    return out


def capacity_panels(df: pd.DataFrame, gammas: Optional[Sequence[float]] = None,
                    regimes: Optional[Sequence[str]] = None, out: str = "capacity.pdf",
                    title: Optional[str] = None, capacity_unit: str = "GW",
                    panel_width: float = 3.4, height: float = 4.0, dpi: int = 300,
                    show_curtailment: bool = True):
    """Stacked capacity bars by regime, one panel per gamma, curtailment hours on the right axis.

    ``df`` is the combined summary table (one row per regime x gamma).  Panels share the
    capacity axis so bar heights are comparable across gammas; the curtailment axis is shared
    too, and only the right-most panel labels it.
    """
    plt = _require_mpl()

    if regimes is not None:
        df = df[df["regime"].isin(list(regimes))]
    if not len(df):
        raise SystemExit("nothing to plot after filtering")
    gcol = _gamma_col(df)
    gs = list(gammas) if gammas is not None else sorted(df[gcol].unique())
    missing = [g for g in gs if not (df[gcol] == g).any()]
    if missing:
        raise SystemExit(f"no rows for gamma={missing}; present: {sorted(df[gcol].unique())}")
    # Every panel shows the same regimes in the same slots, so bars line up across gammas.
    bar_order = list(regimes) if regimes is not None else [
        r for r in pd.unique(df["regime"].astype(str))]

    techs = _tech_columns(df)
    scale = 1e-3 if capacity_unit.upper() == "GW" else 1.0

    fig, axes = plt.subplots(1, len(gs), figsize=(panel_width * len(gs), height),
                             sharey=True, squeeze=False)
    axes = axes[0]
    # Common right-hand axis limit so curtailment markers are comparable across panels.
    curt_all = _curtailment_hours(df) if show_curtailment else None
    curt_max = float(np.nanmax(curt_all)) if curt_all is not None and len(curt_all) else 0.0
    twins = []

    for ax, g in zip(axes, gs):
        sub = df[df[gcol] == g].copy()
        sub["regime"] = sub["regime"].astype(str)
        sub = sub.set_index("regime").reindex(bar_order)          # missing regimes keep their slot
        absent = [r for r in bar_order if sub[f"K_{techs[0]}_MW"].isna().get(r, True)]
        if absent:
            log.warning("gamma=%g has no rows for %s; leaving those slots empty", g, absent)
        x = np.arange(len(sub))
        bottom = np.zeros(len(sub))
        for z in techs:
            v = sub[f"K_{z}_MW"].to_numpy(dtype=float) * scale
            v = np.clip(np.nan_to_num(v, nan=0.0), 0.0, None)  # tiny negative LP values; absent = 0
            ax.bar(x, v, bottom=bottom, width=0.68, color=TECH_COLORS.get(z, "#999999"),
                   edgecolor="white", linewidth=0.4, label=TECH_LABELS.get(z, z), zorder=2)
            bottom += v
        ax.set_xticks(x)
        ax.set_xticklabels(sub.index, fontsize=9)
        ax.set_title(rf"$\gamma = {g:g}$", fontsize=11)
        ax.grid(axis="y", color="#DDDDDD", linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

        if show_curtailment:
            curt = _curtailment_hours(sub.reset_index())
            if curt is not None:
                tw = ax.twinx()
                tw.plot(x, curt, linestyle="none", marker="D", markersize=5.5,
                        color=CURT_COLOR, markeredgecolor="white", markeredgewidth=0.7,
                        zorder=4, clip_on=False)
                top = max(curt_max * 1.25, 1.0)
                tw.set_ylim(-0.045 * top, top)             # keep a zero marker off the tick labels
                tw.spines["top"].set_visible(False)
                twins.append(tw)

    axes[0].set_ylabel(f"Installed capacity ({capacity_unit})")
    for tw in twins[:-1]:
        tw.set_yticklabels([])
    if twins:
        twins[-1].set_ylabel("Curtailment (mean hours/yr)", color=CURT_COLOR)
        twins[-1].tick_params(axis="y", colors=CURT_COLOR)

    handles, labels = axes[0].get_legend_handles_labels()
    if twins:
        from matplotlib.lines import Line2D
        handles.append(Line2D([], [], linestyle="none", marker="D", markersize=5.5, color=CURT_COLOR))
        labels.append("Curtailment hours (right axis)")
    ncol = min(len(labels), 5)
    fig.legend(handles, labels, loc="lower center", ncol=ncol, frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, -0.02))
    if title:
        fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0.08, 1, 0.97 if title else 1.0))
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    log.info("wrote %s (%d panel(s), %d technologies)", out, len(gs), len(techs))
    return out
