"""
Sparse LP construction and a thin HiGHS wrapper.

:class:`SystemLP` builds the planner LP (eq. planner in empirical_model_setup.tex)

    min  sum_y f_y sum_h [ sum_m (c_m + e_m tau) q_mhy + VOLL * l_hy ]
         + sum_z I_z K_z  (+ Psi(K) if reliability costs are internalised)
    s.t. balance(h,y):  sum_m q + sum_r q + sum_s (q+ - q-) + l = D_hy          [f_y p_hy]
         q_mhy <= K_m ;  q_rhy <= theta_rhy K_r
         q+_shy <= K_s ; q-_shy <= K_s ; E_shy <= d_s K_s
         E_shy = E_{s,h-1,y} + eps q-_shy - q+_shy / eps ,  E_{s,0,y} = E_{s,H,y}

over Y*H chronological hours with capacity variables K_z.  The same object is reused
for the market regimes by changing the K cost coefficients (warm-started re-solves)
and for dispatch-only evaluations by fixing K through its bounds.

Sign conventions (HiGHS, minimisation): the dual of the balance row is d(obj)/d(D_hy)
= f_y * p_hy >= 0; the dual of a "<= 0" capacity row is <= 0 and equals
-(marginal value of one MW of capacity in that hour, weighted by f_y).
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import scipy.sparse as sp

from .panel import HourlyPanel
from .params import ModelParams, Regime, Tech

log = logging.getLogger("eq_model")

try:
    import highspy
    _HAVE_HIGHSPY = True
except ImportError:  # pragma: no cover
    highspy = None
    _HAVE_HIGHSPY = False


# ---------------------------------------------------------------------------
# HiGHS wrapper
# ---------------------------------------------------------------------------

class HighsModel:
    """Minimal wrapper: pass a CSC LP once, then re-solve after cost/bound edits."""

    def __init__(self, c: np.ndarray, A: sp.csc_matrix, lb: np.ndarray, ub: np.ndarray,
                 rl: np.ndarray, ru: np.ndarray, name: str = "lp", options: Optional[dict] = None):
        if not _HAVE_HIGHSPY:
            raise RuntimeError("highspy is required: pip install highspy")
        self.n, self.m = A.shape[1], A.shape[0]
        self.name = name
        self.h = highspy.Highs()
        self.h.setOptionValue("output_flag", False)
        self.h.setOptionValue("log_to_console", False)
        for k, v in (options or {}).items():
            self.h.setOptionValue(k, v)
        lp = highspy.HighsLp()
        lp.num_col_ = int(self.n)
        lp.num_row_ = int(self.m)
        lp.col_cost_ = np.asarray(c, dtype=np.float64)
        lp.col_lower_ = np.asarray(lb, dtype=np.float64)
        lp.col_upper_ = np.asarray(ub, dtype=np.float64)
        lp.row_lower_ = np.asarray(rl, dtype=np.float64)
        lp.row_upper_ = np.asarray(ru, dtype=np.float64)
        lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
        lp.a_matrix_.start_ = np.asarray(A.indptr, dtype=np.int32)
        lp.a_matrix_.index_ = np.asarray(A.indices, dtype=np.int32)
        lp.a_matrix_.value_ = np.asarray(A.data, dtype=np.float64)
        lp.sense_ = highspy.ObjSense.kMinimize
        self.h.passModel(lp)
        self._solved = False

    def set_costs(self, idx: Sequence[int], vals: Sequence[float]) -> None:
        idx = np.asarray(idx, dtype=np.int32)
        vals = np.asarray(vals, dtype=np.float64)
        self.h.changeColsCost(len(idx), idx, vals)

    def set_bounds(self, idx: Sequence[int], lb: Sequence[float], ub: Sequence[float]) -> None:
        idx = np.asarray(idx, dtype=np.int32)
        self.h.changeColsBounds(len(idx), idx, np.asarray(lb, dtype=np.float64), np.asarray(ub, dtype=np.float64))

    def solve(self, solver: str = "auto", time_limit: Optional[float] = None) -> dict:
        """solver: 'auto' (simplex warm start if a basis exists, else ipm+crossover for big
        models), 'simplex', 'ipm', 'pdlp'."""
        t0 = time.time()
        if solver == "auto":
            if self._solved:
                self.h.setOptionValue("solver", "simplex")
            elif self.n > 200_000:
                self.h.setOptionValue("solver", "ipm")
                self.h.setOptionValue("run_crossover", "on")
            else:
                self.h.setOptionValue("solver", "simplex")
        else:
            self.h.setOptionValue("solver", solver)
            if solver == "ipm":
                self.h.setOptionValue("run_crossover", "on")
        if time_limit:
            self.h.setOptionValue("time_limit", float(time_limit))
        self.h.run()
        status = self.h.getModelStatus()
        st = self.h.modelStatusToString(status)
        if st not in ("Optimal",):
            raise RuntimeError(f"HiGHS[{self.name}] status {st}")
        sol = self.h.getSolution()
        info = self.h.getInfo()
        self._solved = True
        out = {"x": np.asarray(sol.col_value, dtype=float),
               "col_dual": np.asarray(sol.col_dual, dtype=float),
               "row_dual": np.asarray(sol.row_dual, dtype=float),
               "row_value": np.asarray(sol.row_value, dtype=float),
               "obj": float(info.objective_function_value),
               "status": st, "time": time.time() - t0,
               "simplex_iters": int(info.simplex_iteration_count), "ipm_iters": int(info.ipm_iteration_count)}
        log.debug("HiGHS[%s]: %s obj=%.6e in %.1fs (simplex %d, ipm %d iters)", self.name, st, out["obj"],
                  out["time"], out["simplex_iters"], out["ipm_iters"])
        return out


# ---------------------------------------------------------------------------
# System LP
# ---------------------------------------------------------------------------

@dataclass
class LPSolution:
    K: Dict[str, float]                       # MW
    price: np.ndarray                         # (Y, H)  $/MWh   (= balance dual / f_y)
    q: Dict[str, np.ndarray]                  # tech -> (Y,H) net output (storage: q+ - q-)
    q_charge: Dict[str, np.ndarray]           # storage -> (Y,H) q-
    q_discharge: Dict[str, np.ndarray]        # storage -> (Y,H) q+
    soc: Dict[str, np.ndarray]                # storage -> (Y,H) E
    lost_load: np.ndarray                     # (Y,H)
    cap_margin: Dict[str, np.ndarray]         # tech -> (Y,) sum_h of capacity-constraint dual value, $/MW-yr (undiscounted)
    obj: float
    solve_time: float
    cost_K_used: Dict[str, float]             # objective coefficient on K_z actually used ($/MW-yr)
    meta: dict = field(default_factory=dict)


class SystemLP:
    """The monolithic LP over ``panel`` (all years).  Build once; re-solve many times."""

    def __init__(self, panel: HourlyPanel, params: ModelParams, tau: float, include_psi: bool,
                 k_upper: Optional[Dict[str, float]] = None, k_fixed: Optional[Dict[str, float]] = None,
                 name: str = "system", options: Optional[dict] = None):
        self.panel, self.params, self.tau, self.include_psi = panel, params, tau, include_psi
        self.techs: List[Tech] = list(params.techs)
        self.Y, self.H = panel.Y, panel.H
        self.T = self.Y * self.H
        self.f = np.full(self.Y, 1.0 / self.Y)
        self.name = name
        self._layout()
        c, A, lb, ub, rl, ru = self._build(k_upper or {}, k_fixed or {})
        self.nnz = A.nnz
        log.info("SystemLP[%s]: %d cols, %d rows, %d nnz (Y=%d, H=%d)", name, A.shape[1], A.shape[0], A.nnz, self.Y, self.H)
        self.model = HighsModel(c, A, lb, ub, rl, ru, name=name, options=options)
        self.base_cost_K = {t.name: float(c[self.col_K[t.name]]) for t in self.techs}

    # ----------------------------------------------------------------- layout
    def _layout(self) -> None:
        T = self.T
        self.col: Dict[str, int] = {}         # block name -> starting column
        n = 0
        for t in self.techs:
            if t.kind in ("thermal", "vre"):
                self.col[f"q:{t.name}"] = n; n += T
            else:
                self.col[f"qp:{t.name}"] = n; n += T
                self.col[f"qm:{t.name}"] = n; n += T
                self.col[f"E:{t.name}"] = n; n += T
        self.col["l"] = n; n += T
        self.col_K: Dict[str, int] = {}
        for t in self.techs:
            self.col_K[t.name] = n; n += 1
        self.ncol = n
        self.row: Dict[str, int] = {}
        m = 0
        self.row["balance"] = m; m += T
        for t in self.techs:
            if t.kind in ("thermal", "vre"):
                self.row[f"cap:{t.name}"] = m; m += T
            else:
                self.row[f"capp:{t.name}"] = m; m += T
                self.row[f"capm:{t.name}"] = m; m += T
                self.row[f"capE:{t.name}"] = m; m += T
                self.row[f"dyn:{t.name}"] = m; m += T
        self.nrow = m

    # ------------------------------------------------------------------ build
    def _build(self, k_upper: Dict[str, float], k_fixed: Dict[str, float]):
        P, T, Y, H = self.params, self.T, self.Y, self.H
        fy = np.repeat(self.f, H)                       # (T,) f_y per hour
        D = self.panel.D.ravel()
        mc = P.marginal_cost(self.tau)
        I = P.investment_cost(H)
        if H != 8760:
            log.warning("SystemLP: H=%d != 8760 - annual fixed costs prorated by H/8760 (test mode)", H)
        ar = np.arange(T, dtype=np.int64)
        rows, cols, vals = [], [], []

        def add(r, c, v):
            rows.append(np.asarray(r, dtype=np.int64)); cols.append(np.asarray(c, dtype=np.int64)); vals.append(np.asarray(v, dtype=np.float64))

        c = np.zeros(self.ncol)
        lb = np.zeros(self.ncol)
        ub = np.full(self.ncol, np.inf)
        rl = np.zeros(self.nrow)
        ru = np.zeros(self.nrow)
        # balance: equality D
        rb = self.row["balance"]
        rl[rb:rb + T] = D; ru[rb:rb + T] = D
        for t in self.techs:
            kcol = self.col_K[t.name]
            if t.kind == "thermal":
                q0 = self.col[f"q:{t.name}"]
                c[q0:q0 + T] = fy * mc[t.name]
                add(rb + ar, q0 + ar, np.ones(T))                        # balance
                r0 = self.row[f"cap:{t.name}"]
                add(r0 + ar, q0 + ar, np.ones(T)); add(r0 + ar, np.full(T, kcol), -np.ones(T))
                rl[r0:r0 + T] = -np.inf; ru[r0:r0 + T] = 0.0
            elif t.kind == "vre":
                q0 = self.col[f"q:{t.name}"]
                th = self.panel.theta[t.vre_key].ravel()
                add(rb + ar, q0 + ar, np.ones(T))
                r0 = self.row[f"cap:{t.name}"]
                add(r0 + ar, q0 + ar, np.ones(T)); add(r0 + ar, np.full(T, kcol), -th)
                rl[r0:r0 + T] = -np.inf; ru[r0:r0 + T] = 0.0
            else:
                qp, qm, E = self.col[f"qp:{t.name}"], self.col[f"qm:{t.name}"], self.col[f"E:{t.name}"]
                eps = P.storage_eff_oneway
                add(rb + ar, qp + ar, np.ones(T)); add(rb + ar, qm + ar, -np.ones(T))
                for key, c0, coef in ((f"capp:{t.name}", qp, 1.0), (f"capm:{t.name}", qm, 1.0), (f"capE:{t.name}", E, 1.0)):
                    r0 = self.row[key]
                    add(r0 + ar, c0 + ar, np.ones(T))
                    add(r0 + ar, np.full(T, kcol), -np.full(T, t.duration_h if key.startswith("capE") else 1.0))
                    rl[r0:r0 + T] = -np.inf; ru[r0:r0 + T] = 0.0
                # dynamics: E_t - E_{t-1} - eps*qm_t + qp_t/eps = 0, cyclic within each year
                r0 = self.row[f"dyn:{t.name}"]
                prev = ar - 1
                first = (ar % H == 0)
                prev[first] = ar[first] + H - 1                         # E_{s,0,y} = E_{s,H,y}
                add(r0 + ar, E + ar, np.ones(T)); add(r0 + ar, E + prev, -np.ones(T))
                add(r0 + ar, qm + ar, -np.full(T, eps)); add(r0 + ar, qp + ar, np.full(T, 1.0 / eps))
                rl[r0:r0 + T] = 0.0; ru[r0:r0 + T] = 0.0
            # capacity cost
            c[kcol] = I[t.name] + (self.psi_coef(t) if self.include_psi else 0.0)
            if t.name in k_fixed:
                lb[kcol] = ub[kcol] = float(k_fixed[t.name])
            elif t.name in k_upper:
                ub[kcol] = float(k_upper[t.name])
        # lost load
        l0 = self.col["l"]
        c[l0:l0 + T] = fy * P.voll
        ub[l0:l0 + T] = D
        add(rb + ar, l0 + ar, np.ones(T))
        A = sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                          shape=(self.nrow, self.ncol)).tocsc()
        A.sum_duplicates()
        return c, A, lb, ub, rl, ru

    # ---------------------------------------------------------- reliability
    def psi_coef(self, t: Tech) -> float:
        """Reliability cost per MW of VRE capacity per year (f-weighted), $/MW-yr.
        Psi_r(K_r) = psi_coef(r) * K_r.  Zero for non-VRE."""
        return psi_coefficient(self.panel, self.params, t)

    # ------------------------------------------------------------------ solve
    def solve(self, cost_K: Optional[Dict[str, float]] = None, solver: str = "auto",
              k_fixed: Optional[Dict[str, float]] = None, k_upper: Optional[Dict[str, float]] = None) -> LPSolution:
        """Solve with (optionally) overridden K cost coefficients ($/MW-yr, *including* any
        Psi term you want the objective to carry).  Warm-starts after the first solve."""
        used = dict(self.base_cost_K)
        if cost_K:
            used.update({k: float(v) for k, v in cost_K.items()})
            self.model.set_costs([self.col_K[k] for k in used], [used[k] for k in used])
        if k_fixed is not None or k_upper is not None:
            idx, lbs, ubs = [], [], []
            for t in self.techs:
                idx.append(self.col_K[t.name])
                if k_fixed is not None and t.name in k_fixed:
                    lbs.append(float(k_fixed[t.name])); ubs.append(float(k_fixed[t.name]))
                else:
                    lbs.append(0.0); ubs.append(float((k_upper or {}).get(t.name, np.inf)))
            self.model.set_bounds(idx, lbs, ubs)
        res = self.model.solve(solver=solver)
        return self._unpack(res, used)

    def _unpack(self, res: dict, used: Dict[str, float]) -> LPSolution:
        x, rd = res["x"], res["row_dual"]
        Y, H, T = self.Y, self.H, self.T
        fy = np.repeat(self.f, H)
        rb = self.row["balance"]
        price = (rd[rb:rb + T] / fy).reshape(Y, H)
        K = {t.name: float(x[self.col_K[t.name]]) for t in self.techs}
        q, qc, qd, soc, margin = {}, {}, {}, {}, {}
        for t in self.techs:
            if t.kind in ("thermal", "vre"):
                q0 = self.col[f"q:{t.name}"]
                q[t.name] = x[q0:q0 + T].reshape(Y, H)
                r0 = self.row[f"cap:{t.name}"]
                w = self.panel.theta[t.vre_key].ravel() if t.kind == "vre" else 1.0   # d(row)/dK = -theta
                margin[t.name] = (-rd[r0:r0 + T] * w / fy).reshape(Y, H).sum(axis=1)
            else:
                qp, qm, E = self.col[f"qp:{t.name}"], self.col[f"qm:{t.name}"], self.col[f"E:{t.name}"]
                qd[t.name] = x[qp:qp + T].reshape(Y, H)
                qc[t.name] = x[qm:qm + T].reshape(Y, H)
                soc[t.name] = x[E:E + T].reshape(Y, H)
                q[t.name] = qd[t.name] - qc[t.name]
                tot = np.zeros(T)
                for key, w in ((f"capp:{t.name}", 1.0), (f"capm:{t.name}", 1.0), (f"capE:{t.name}", t.duration_h)):
                    r0 = self.row[key]
                    tot += -rd[r0:r0 + T] * w
                margin[t.name] = (tot / fy).reshape(Y, H).sum(axis=1)
        l0 = self.col["l"]
        ll = x[l0:l0 + T].reshape(Y, H)
        return LPSolution(K=K, price=price, q=q, q_charge=qc, q_discharge=qd, soc=soc, lost_load=ll,
                          cap_margin=margin, obj=res["obj"], solve_time=res["time"], cost_K_used=used,
                          meta={"simplex_iters": res["simplex_iters"], "ipm_iters": res["ipm_iters"]})


def psi_coefficient(panel: HourlyPanel, params: ModelParams, t: Tech) -> float:
    """f-weighted annual reliability cost per MW of VRE capacity ($/MW-yr); 0 for non-VRE.

    wind : sum_y f_y sum_h theta_wind,hy * (flex_price*0.189 + reg_price*0.005)
    solar: sum_y f_y sum_h 1[daylight_hy] * (flex_price*0.076 + reg_price*0.003) / divisor
    """
    if t.kind != "vre":
        return 0.0
    f = 1.0 / panel.Y
    if t.vre_key == "wind":
        return float(f * panel.theta["wind"].sum() * params.psi_wind_per_mwh_available())
    if t.vre_key == "solar":
        return float(f * panel.daylight.sum() * params.psi_solar_per_mw_daylight_hour())
    raise ValueError(t.vre_key)


def psi_by_year(panel: HourlyPanel, params: ModelParams, K: Dict[str, float]) -> Dict[str, np.ndarray]:
    """Realised reliability cost by year for each VRE technology, $/yr: (Y,) arrays."""
    out = {}
    for t in params.vre:
        if t.vre_key == "wind":
            out[t.name] = panel.theta["wind"].sum(axis=1) * params.psi_wind_per_mwh_available() * K[t.name]
        else:
            out[t.name] = panel.daylight.sum(axis=1) * params.psi_solar_per_mw_daylight_hour() * K[t.name]
    return out


# ---------------------------------------------------------------------------
# Storage arbitrage value per MW at given prices (agent's own problem, K_s = 1)
# ---------------------------------------------------------------------------

class StorageArbitrageLP:
    """max sum_h p_h (q+_h - q-_h)  s.t. q+,q- <= 1, 0 <= E <= d, dynamics, cyclic.
    One year at a time; costs (prices) are swapped between solves (warm start)."""

    def __init__(self, H: int, duration: float, eps: float):
        self.H, self.d, self.eps = H, duration, eps
        T = H
        ar = np.arange(T)
        prev = ar - 1; prev[0] = T - 1
        n = 3 * T
        rows = np.concatenate([ar, ar, ar, ar])
        cols = np.concatenate([2 * T + ar, 2 * T + prev, T + ar, ar])
        vals = np.concatenate([np.ones(T), -np.ones(T), -np.full(T, eps), np.full(T, 1.0 / eps)])
        A = sp.coo_matrix((vals, (rows, cols)), shape=(T, n)).tocsc()
        lb = np.zeros(n); ub = np.concatenate([np.ones(T), np.ones(T), np.full(T, duration)])
        self.model = HighsModel(np.zeros(n), A, lb, ub, np.zeros(T), np.zeros(T), name=f"storage{duration:g}h")

    def value(self, price: np.ndarray) -> float:
        """Arbitrage profit per MW over one year ($/MW-yr)."""
        T = self.H
        c = np.concatenate([-price, price, np.zeros(T)])       # minimise -profit
        self.model.set_costs(np.arange(3 * T), c)
        res = self.model.solve(solver="simplex")
        return -res["obj"]
