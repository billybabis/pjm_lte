"""
Single-year economic dispatch with *fixed* capacities, used as the Benders subproblem
and as the "spot market given K" evaluation.

    V_y(K) = min sum_h [ sum_m (c_m + e_m tau) q_mh + VOLL l_h ]
             s.t. balance_h (dual p_h), 0 <= q_mh <= K_m, 0 <= q_rh <= theta_rh K_r,
                  0 <= q+/- <= K_s, 0 <= E_h <= d_s K_s, storage dynamics (cyclic).

V_y is convex and piecewise linear in K; its subgradient is -m_y(K) where
m_zy(K) >= 0 is the value of one more MW of z in year y (sum over hours of the
reduced cost on the binding upper bound).  Because capacities enter only through
variable bounds, re-solving after a change in K is a warm-started dual simplex.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import scipy.sparse as sp

from .lp import HighsModel
from .panel import HourlyPanel
from .params import ModelParams, Tech

log = logging.getLogger("eq_model")


@dataclass
class DispatchResult:
    V: float                       # objective, $ (one year, undiscounted)
    price: np.ndarray              # (H,) $/MWh
    margin: Dict[str, float]       # tech -> $/MW-yr  (= -dV/dK_z)
    q: Dict[str, np.ndarray]       # tech -> (H,) net output
    q_charge: Dict[str, np.ndarray]
    q_discharge: Dict[str, np.ndarray]
    soc: Dict[str, np.ndarray]
    lost_load: np.ndarray          # (H,)
    time: float
    iters: int


class DispatchLP:
    """One year, fixed K.  Build once per (year, tau); call ``solve(K)`` repeatedly."""

    def __init__(self, panel: HourlyPanel, y: int, params: ModelParams, tau: float,
                 options: Optional[dict] = None):
        self.panel, self.y, self.params, self.tau = panel, y, params, tau
        self.techs: List[Tech] = list(params.techs)
        H = self.H = panel.H
        self.D = panel.D[y]
        self.mc = params.marginal_cost(tau)
        ar = np.arange(H)
        self.col: Dict[str, int] = {}
        n = 0
        for t in self.techs:
            if t.kind in ("thermal", "vre"):
                self.col[f"q:{t.name}"] = n; n += H
            else:
                self.col[f"qp:{t.name}"] = n; n += H
                self.col[f"qm:{t.name}"] = n; n += H
                self.col[f"E:{t.name}"] = n; n += H
        self.col["l"] = n; n += H
        self.ncol = n
        self.row = {"balance": 0}
        m = H
        for t in params.storage:
            self.row[f"dyn:{t.name}"] = m; m += H
        self.nrow = m
        rows, cols, vals = [], [], []
        c = np.zeros(n)
        for t in self.techs:
            if t.kind in ("thermal", "vre"):
                q0 = self.col[f"q:{t.name}"]
                rows.append(ar); cols.append(q0 + ar); vals.append(np.ones(H))
                c[q0:q0 + H] = self.mc[t.name]
            else:
                qp, qm, E = self.col[f"qp:{t.name}"], self.col[f"qm:{t.name}"], self.col[f"E:{t.name}"]
                eps = params.storage_eff_oneway
                rows += [ar, ar]; cols += [qp + ar, qm + ar]; vals += [np.ones(H), -np.ones(H)]
                r0 = self.row[f"dyn:{t.name}"]
                prev = ar - 1; prev[0] = H - 1
                rows += [r0 + ar, r0 + ar, r0 + ar, r0 + ar]
                cols += [E + ar, E + prev, qm + ar, qp + ar]
                vals += [np.ones(H), -np.ones(H), -np.full(H, eps), np.full(H, 1.0 / eps)]
        l0 = self.col["l"]
        rows.append(ar); cols.append(l0 + ar); vals.append(np.ones(H))
        c[l0:l0 + H] = params.voll
        A = sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                          shape=(self.nrow, self.ncol)).tocsc()
        rl = np.zeros(self.nrow); ru = np.zeros(self.nrow)
        rl[:H] = self.D; ru[:H] = self.D
        lb = np.zeros(n)
        ub = np.zeros(n)
        ub[l0:l0 + H] = self.D
        self.model = HighsModel(c, A, lb, ub, rl, ru, name=f"dispatch{panel.years[y]}", options=options)
        self._ub_idx = np.arange(n - H)      # all but l
        self._lb_all = np.zeros(n - H)

    def _bounds_for(self, K: Dict[str, float]) -> np.ndarray:
        H = self.H
        ub = np.zeros(self.ncol - H)
        for t in self.techs:
            if t.kind == "thermal":
                q0 = self.col[f"q:{t.name}"]; ub[q0:q0 + H] = K[t.name]
            elif t.kind == "vre":
                q0 = self.col[f"q:{t.name}"]; ub[q0:q0 + H] = self.panel.theta[t.vre_key][self.y] * K[t.name]
            else:
                qp, qm, E = self.col[f"qp:{t.name}"], self.col[f"qm:{t.name}"], self.col[f"E:{t.name}"]
                ub[qp:qp + H] = K[t.name]; ub[qm:qm + H] = K[t.name]; ub[E:E + H] = t.duration_h * K[t.name]
        return ub

    def solve(self, K: Dict[str, float], solver: str = "simplex") -> DispatchResult:
        H = self.H
        ub = self._bounds_for(K)
        self.model.set_bounds(self._ub_idx, self._lb_all, ub)
        res = self.model.solve(solver=solver)
        x, cd, rd = res["x"], res["col_dual"], res["row_dual"]
        price = rd[:H].copy()
        q, qc, qd, soc, margin = {}, {}, {}, {}, {}
        for t in self.techs:
            if t.kind in ("thermal", "vre"):
                q0 = self.col[f"q:{t.name}"]
                q[t.name] = x[q0:q0 + H]
                # reduced cost on a binding upper bound is <= 0 (min); value of relaxing = -cd
                w = self.panel.theta[t.vre_key][self.y] if t.kind == "vre" else 1.0
                margin[t.name] = float(np.sum(np.minimum(cd[q0:q0 + H], 0.0) * -w))
            else:
                qp, qm, E = self.col[f"qp:{t.name}"], self.col[f"qm:{t.name}"], self.col[f"E:{t.name}"]
                qd[t.name] = x[qp:qp + H]; qc[t.name] = x[qm:qm + H]; soc[t.name] = x[E:E + H]
                q[t.name] = qd[t.name] - qc[t.name]
                margin[t.name] = float(-np.sum(np.minimum(cd[qp:qp + H], 0.0)) - np.sum(np.minimum(cd[qm:qm + H], 0.0))
                                       - t.duration_h * np.sum(np.minimum(cd[E:E + H], 0.0)))
        l0 = self.col["l"]
        return DispatchResult(V=res["obj"], price=price, margin=margin, q=q, q_charge=qc, q_discharge=qd, soc=soc,
                              lost_load=x[l0:l0 + H].copy(), time=res["time"], iters=res["simplex_iters"] + res["ipm_iters"])
