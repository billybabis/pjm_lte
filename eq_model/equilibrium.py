"""
Capacity problems solved by trust-region cutting planes over the yearly dispatch LPs,
and the risk-averse market equilibrium on top of them.

Planner (P1, P2):
    min_K  sum_z c_z K_z + sum_y f_y V_y(K),   c_z = I_z + psi_z   (psi: reliability cost per MW)

V_y(K) is the convex piecewise-linear optimal dispatch cost of year y (dispatch.py).  Each
evaluation of a candidate K yields V_y(K) and a subgradient (-m_y(K)), i.e. one Benders cut
per year.  Cuts depend only on the panel and tau, so a pool is shared across regimes,
gamma values and the outer iterations of the market solver.

Market (R2-R6): the risk-averse competitive equilibrium is NOT the solution of a single
optimisation problem (agents hold heterogeneous risk-adjusted probabilities), but given
prices every agent's problem is an LP whose zero-profit condition can be written as

    sum_y f_y m_zy(K) = I_eff_z,      I_eff_z := sum_y f_y m_zy(K) - rho_z^*(K, p_hat),

i.e. the equilibrium K solves the *planner-form* problem with effective (risk-premium
loaded) capacity costs I_eff.  I_eff is found by a damped fixed-point iteration; each
step re-solves the planner-form problem with the shared cut pool (cheap after the first
solve).  At a fixed point rho_z^* = 0 for active technologies and <= 0 for inactive ones,
spot markets clear hour by hour (the dispatch LP is the competitive spot equilibrium given K)
and the forward market clears (agents.forward_market).
"""
from __future__ import annotations

import logging
import multiprocessing as mp
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linprog

from .agents import ForwardResult, MarginEvaluator, forward_market, rho
from .dispatch import DispatchLP, DispatchResult
from .lp import SystemLP, psi_coefficient
from .panel import HourlyPanel
from .params import ModelParams, Regime

log = logging.getLogger("eq_model")

_K_SCALE = 1e3     # master works in GW
_C_SCALE = 1e6     # ... and $M
# Smallest step on an effective cost, as a fraction of I_z.  Too large (0.02) and a secant
# overshoot can repeat forever; too small (1e-5, ~$1/MW-yr for a CT) and steps cannot move
# capacity off an LP plateau, so the solver freezes while prices drift (R6, gamma=0.9, tau=100
# stalled at a 2.4% residual on ~$1 steps).  1e-3 converged that cell and left tau=280 identical.
_MIN_STEP_FRAC = 1e-3


@dataclass
class Cut:
    K: np.ndarray            # (nz,) MW
    V: np.ndarray            # (Y,) $
    m: np.ndarray            # (Y, nz) $/MW-yr   (subgradient of V_y is -m_y)


@dataclass
class Evaluation:
    K: Dict[str, float]
    V: np.ndarray                          # (Y,)
    margins: Dict[str, np.ndarray]         # z -> (Y,)
    price: np.ndarray                      # (Y,H)
    dispatch: List[DispatchResult]
    time: float


@dataclass
class CapacityResult:
    K: Dict[str, float]
    cost_K: Dict[str, float]
    evaluation: Evaluation                 # at K, with KKT-consistent prices if completed
    objective: float                       # sum c K + sum f V
    lower_bound: float
    foc_residual: float                    # max relative FOC violation at K (with the evaluation's prices)
    foc: Dict[str, float]                  # z -> (f.m_z - c_z)/c_z
    iterations: int
    n_evaluations: int
    converged: bool
    completed: bool                        # prices from the monolithic re-solve (exact KKT) or from dispatch
    history: List[dict] = field(default_factory=list)
    completion_time: float = 0.0


def _dispatch_worker(panel, params, tau, years, task_q, result_q):
    """Process owning the DispatchLPs of a subset of years (warm starts stay local)."""
    lps = {y: DispatchLP(panel, y, params, tau) for y in years}
    while True:
        K = task_q.get()
        if K is None:
            break
        try:
            result_q.put({y: lps[y].solve(K) for y in years})
        except Exception as e:                      # pragma: no cover
            result_q.put(e)


class CapacityProblem:
    """Trust-region cutting-plane solver for  min_K c.K + sum_y f_y V_y(K)  with a reusable cut pool,
    plus an exact price-completion step.  ``workers`` > 1 solves the yearly dispatches in parallel
    processes (each process keeps its own warm-started HiGHS models).

    Why price completion?  At the optimal K the yearly dispatch LPs are dual-degenerate in the
    few hours where load exactly equals installed capacity: any price between the marginal
    unit's cost and VOLL is a valid dual there, and the fixed-K dispatch returns an arbitrary
    one.  The KKT conditions of the *capacity* problem (zero profit for active technologies)
    select specific values.  ``complete`` recovers them by solving the monolithic LP warm-
    started from the dispatch solution with K boxed tightly around the cutting-plane optimum
    (a few thousand pivots instead of a cold solve)."""

    def __init__(self, panel: HourlyPanel, params: ModelParams, tau: float, k_max_mult: float = 4.0,
                 options: Optional[dict] = None, workers: int = 1):
        self.panel, self.params, self.tau = panel, params, tau
        self.names = params.tech_names
        self.nz, self.Y, self.H = len(self.names), panel.Y, panel.H
        self.f = np.full(self.Y, 1.0 / self.Y)
        self.workers = max(1, min(int(workers), self.Y))
        self._procs: List[mp.Process] = []
        if self.workers > 1:
            ctx = mp.get_context("fork") if "fork" in mp.get_all_start_methods() else mp.get_context()
            self._task_qs, self._res_q = [], ctx.Queue()
            for i in range(self.workers):
                years = list(range(i, self.Y, self.workers))
                q = ctx.Queue()
                p = ctx.Process(target=_dispatch_worker, args=(panel, params, tau, years, q, self._res_q), daemon=True)
                p.start()
                self._procs.append(p); self._task_qs.append(q)
            self.dispatch = None
        else:
            self.dispatch = [DispatchLP(panel, y, params, tau, options=options) for y in range(self.Y)]
        self.meval = MarginEvaluator(panel, params, tau)
        self.pool: List[Cut] = []
        self.k_max = k_max_mult * panel.D_max
        self.n_eval = 0
        self._cache: Dict[Tuple[float, ...], Evaluation] = {}
        self._system: Optional["SystemLP"] = None      # monolithic LP, built lazily for completion

    # ------------------------------------------------------------- completion
    def complete(self, K: Dict[str, float], cost_K: Dict[str, float], box_rel: float = 0.03,
                 box_abs: float = 300.0, max_expand: int = 4) -> Tuple[Evaluation, Dict[str, float], float]:
        """Exact KKT prices at the capacity optimum near K for objective coefficients cost_K.
        Returns (evaluation with exact prices, K (possibly moved within the box), elapsed)."""
        t0 = time.time()
        if self._system is None:
            self._system = SystemLP(self.panel, self.params, self.tau, include_psi=False, name=f"system_tau{self.tau:g}")
        sysm = self._system
        idx = [sysm.col_K[z] for z in self.names]
        # 1) fixed-K solve (a dispatch; warm from whatever basis the model holds).  The solution is
        # discarded - this exists to move the basis to a good starting point for step 2.  Skipping
        # it after the first completion was tried and is 3-18% SLOWER: fixed-K is a much easier
        # subproblem than the box, and reaching the box optimum through it beats going there
        # directly from the previous box basis.  (Results are identical either way, ~1e-16.)
        sysm.model.set_costs(idx, [cost_K[z] for z in self.names])
        Kc = {z: float(K[z]) for z in self.names}
        sysm.model.set_bounds(idx, [Kc[z] for z in self.names], [Kc[z] for z in self.names])
        sysm.model.solve(solver="simplex")
        # 2) relax K within a box and re-solve; expand if K ends on the box boundary
        for attempt in range(max_expand + 1):
            lo = [max(0.0, Kc[z] - max(box_rel * Kc[z], box_abs)) for z in self.names]
            hi = [Kc[z] + max(box_rel * Kc[z], box_abs) for z in self.names]
            sysm.model.set_bounds(idx, lo, hi)
            res = sysm.model.solve(solver="simplex")
            sol = sysm._unpack(res, dict(cost_K))
            Knew = {z: float(sol.K[z]) for z in self.names}
            on_bd = [z for i, z in enumerate(self.names)
                     if (Knew[z] >= hi[i] - 1e-6) or (lo[i] > 0 and Knew[z] <= lo[i] + 1e-6)]
            if not on_bd:
                break
            log.info("price completion: K at box boundary for %s - expanding box (attempt %d)", on_bd, attempt + 1)
            Kc = Knew
            box_rel *= 3.0; box_abs *= 3.0
        margins = self.meval.margins(sol.price)
        # yearly dispatch cost at Knew (for the objective and a valid cut): the monolithic duals
        # restricted to year y are optimal duals of year y's dispatch at Knew, so (V_y, margins)
        # is a valid (V, subgradient) pair.
        res_y = self._solve_years(Knew)
        V = np.array([r.V for r in res_y])
        ev = Evaluation(K=Knew, V=V, margins=margins, price=sol.price, dispatch=res_y, time=time.time() - t0)
        m = np.stack([margins[z] for z in self.names], axis=1)
        self.pool.append(Cut(K=np.array([Knew[z] for z in self.names], float), V=V, m=m))
        self._cache[tuple(round(Knew[z], 6) for z in self.names)] = ev
        while len(self._cache) > 4:
            self._cache.pop(next(iter(self._cache)))
        return ev, Knew, time.time() - t0

    def _solve_years(self, K: Dict[str, float]) -> List[DispatchResult]:
        if self.workers == 1:
            return [d.solve(K) for d in self.dispatch]
        for q in self._task_qs:
            q.put(dict(K))
        out: Dict[int, DispatchResult] = {}
        for _ in self._task_qs:
            r = self._res_q.get()
            if isinstance(r, Exception):
                raise r
            out.update(r)
        return [out[y] for y in range(self.Y)]

    def close(self) -> None:
        for q in getattr(self, "_task_qs", []):
            q.put(None)
        for p in self._procs:
            p.join(timeout=5)
        self._procs = []

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    # --------------------------------------------------------------- evaluate
    def evaluate(self, K: Dict[str, float]) -> Evaluation:
        key = tuple(round(float(K[z]), 6) for z in self.names)
        if key in self._cache:
            return self._cache[key]
        t0 = time.time()
        res = self._solve_years(K)
        price = np.stack([r.price for r in res])
        margins = self.meval.margins(price)          # price-based, exact also for K_z = 0
        V = np.array([r.V for r in res])
        ev = Evaluation(K=dict(K), V=V, margins=margins, price=price, dispatch=res, time=time.time() - t0)
        m = np.stack([margins[z] for z in self.names], axis=1)          # (Y, nz)
        self.pool.append(Cut(K=np.array([K[z] for z in self.names], float), V=V, m=m))
        self.n_eval += 1
        self._cache[key] = ev
        while len(self._cache) > 4:                       # keep memory bounded (dispatch arrays are large)
            self._cache.pop(next(iter(self._cache)))
        log.debug("evaluate #%d: K=%s  f.V=%.4e  (%.1fs)", self.n_eval, {z: round(K[z]) for z in self.names},
                  float(self.f @ V), ev.time)
        return ev

    # ----------------------------------------------------------------- master
    def _master(self, c: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> Tuple[np.ndarray, float]:
        """min c.K + f.theta  s.t. theta_y >= V_jy - m_jy.(K - K_j)  (all cuts), lo<=K<=hi, theta>=0.
        Scaled: K in GW, $ in $M."""
        nz, Y = self.nz, self.Y
        ncut = len(self.pool)
        A = np.zeros((ncut * Y, nz + Y))
        b = np.zeros(ncut * Y)
        for j, cut in enumerate(self.pool):
            mj = cut.m * _K_SCALE / _C_SCALE                      # $M/GW
            Kj = cut.K / _K_SCALE
            Vj = cut.V / _C_SCALE
            r = slice(j * Y, (j + 1) * Y)
            A[r, :nz] = -mj                                        # -theta_y - m.K <= -(V + m.K_j)
            A[r, nz:] = -np.eye(Y)
            b[r] = -(Vj + mj @ Kj)
        obj = np.concatenate([c * _K_SCALE / _C_SCALE, self.f])
        bounds = [(lo[i] / _K_SCALE, hi[i] / _K_SCALE) for i in range(nz)] + [(0, None)] * Y
        r = linprog(obj, A_ub=A, b_ub=b, bounds=bounds, method="highs")
        if r.status != 0:
            raise RuntimeError(f"master LP failed: {r.message}")
        return r.x[:nz] * _K_SCALE, float(r.fun * _C_SCALE)

    # ------------------------------------------------------------------ solve
    def _foc(self, ev: Evaluation, c: np.ndarray, k_active_min: float) -> Tuple[Dict[str, float], float]:
        fm = np.array([self.f @ ev.margins[z] for z in self.names])
        Karr = np.array([ev.K[z] for z in self.names])
        resid = (fm - c) / c
        active = Karr > k_active_min
        foc_res = float(max(np.max(np.abs(resid[active])) if active.any() else 0.0,
                            np.max(resid[~active]) if (~active).any() else 0.0))
        return {z: float(resid[i]) for i, z in enumerate(self.names)}, foc_res

    def solve(self, cost_K: Dict[str, float], K0: Dict[str, float], tol_gap: float = 1e-6,
              max_iter: int = 300, delta0: Optional[float] = None, delta_min: float = 0.5, eta: float = 0.1,
              k_active_min: float = 1.0, complete: bool = True, verbose: bool = True) -> CapacityResult:
        """Trust-region cutting planes until the relative optimality gap (UB - LB)/UB <= tol_gap,
        then (optionally) the exact price completion.  ``tol_gap`` refers to the *objective*;
        the reported FOC residual after completion is the meaningful equilibrium accuracy."""
        c = np.array([cost_K[z] for z in self.names], float)
        Kc = np.array([max(0.0, float(K0[z])) for z in self.names])
        delta = delta0 if delta0 is not None else 0.15 * self.panel.D_max
        ev = self.evaluate(dict(zip(self.names, Kc)))
        UB = float(c @ Kc + self.f @ ev.V)
        hist = []
        converged = False
        LB = -np.inf
        for it in range(1, max_iter + 1):
            _, LB = self._master(c, np.zeros(self.nz), np.full(self.nz, self.k_max))   # global lower bound
            lo = np.maximum(0.0, Kc - delta); hi = np.minimum(self.k_max, Kc + delta)
            Kcand, pred = self._master(c, lo, hi)                                        # trust-region candidate
            gap = (UB - LB) / max(abs(UB), 1.0)
            pred_dec = UB - pred
            _, foc_res = self._foc(ev, c, k_active_min)
            hist.append({"iter": it, "UB": UB, "LB": LB, "gap": gap, "foc_dispatch": foc_res, "delta": delta,
                         "K": dict(zip(self.names, Kc.round(1)))})
            if verbose:
                log.info("  cap it %3d  UB=%.6e  gap=%.2e  pred_dec=%.2e  delta=%.0f  K=%s", it, UB, gap, pred_dec,
                         delta, {z: round(v) for z, v in zip(self.names, Kc)})
            if gap <= tol_gap or (pred_dec <= tol_gap * abs(UB) and delta <= delta_min):
                converged = True
                break
            evc = self.evaluate(dict(zip(self.names, Kcand)))
            UBc = float(c @ Kcand + self.f @ evc.V)
            act_dec = UB - UBc
            on_boundary = bool(np.any(np.isclose(Kcand, lo) & (lo > 0)) or np.any(np.isclose(Kcand, hi) & (hi < self.k_max)))
            if pred_dec > 0 and act_dec >= eta * pred_dec:
                Kc, ev, UB = Kcand, evc, UBc
                if on_boundary and act_dec >= 0.5 * pred_dec:
                    delta = min(2.0 * delta, self.k_max)
            else:
                if UBc < UB:            # still accept an improving step, but shrink
                    Kc, ev, UB = Kcand, evc, UBc
                delta = max(0.5 * delta, delta_min)
        if not converged:
            log.warning("capacity problem: gap %.2e > tol after %d iterations", gap, len(hist))
        completed = False
        ct = 0.0
        if complete:
            ev, Knew, ct = self.complete(dict(zip(self.names, Kc)), cost_K)
            Kc = np.array([Knew[z] for z in self.names])
            UB = float(c @ Kc + self.f @ ev.V)
            completed = True
        foc, foc_res = self._foc(ev, c, k_active_min)
        if np.any(Kc >= 0.99 * self.k_max):
            log.warning("capacity at the artificial upper bound k_max=%.0f MW for %s - check cost_K/I_eff",
                        self.k_max, [z for z, v in zip(self.names, Kc) if v >= 0.99 * self.k_max])
        if verbose:
            log.info("  cap done: %d iters, %d evals, gap=%.2e, FOC resid=%.2e (%s), completion %.1fs",
                     len(hist), self.n_eval, gap, foc_res, "exact prices" if completed else "dispatch prices", ct)
        return CapacityResult(K=dict(zip(self.names, map(float, Kc))), cost_K=dict(cost_K), evaluation=ev,
                              objective=UB, lower_bound=LB, foc_residual=foc_res, foc=foc, iterations=len(hist),
                              n_evaluations=self.n_eval, converged=converged, completed=completed, history=hist,
                              completion_time=ct)


# ---------------------------------------------------------------------------
# regimes
# ---------------------------------------------------------------------------

@dataclass
class RegimeResult:
    regime: Regime
    gamma: float
    tau: float
    K: Dict[str, float]
    I: Dict[str, float]                     # I_z used ($/MW-yr, prorated if H<8760)
    psi_coef: Dict[str, float]              # reliability cost per MW-yr (VRE)
    cost_K: Dict[str, float]                # objective coefficient in the final planner-form problem
    risk_premium: Dict[str, float]          # cost_K - I - (psi if internalised)  ($/MW-yr)
    margins: Dict[str, np.ndarray]          # z -> (Y,) spot margin per MW
    rho_star: Dict[str, float]              # per-MW risk-adjusted profit at equilibrium
    forward: Optional[ForwardResult]
    evaluation: Evaluation
    capacity_result: CapacityResult
    outer_history: List[dict]
    converged: bool
    q_bar: float
    Lambda: float
    Pbar: np.ndarray                        # (Y,)
    runtime: float
    markdown: Dict[str, float] = field(default_factory=dict)   # a_z ($/MWh) applied to contracted energy
    gamma_by_tech: Dict[str, float] = field(default_factory=dict)  # gamma_z each technology's agents used


def default_start(panel: HourlyPanel, params: ModelParams) -> Dict[str, float]:
    D = panel.D_max
    start = {z: 0.0 for z in params.tech_names}
    for t in params.thermal:
        start[t.name] = {"ccgt": 0.75, "ct": 0.35, "nuclear": 0.1, "coal": 0.0}.get(t.name, 0.05) * D
    for t in params.vre:
        start[t.name] = 0.15 * D
    for t in params.storage:
        start[t.name] = 0.03 * D
    return start


def solve_regime(panel: HourlyPanel, params: ModelParams, regime: Regime, gamma: Optional[float] = None,
                 cap: Optional[CapacityProblem] = None, K0: Optional[Dict[str, float]] = None,
                 tol: float = 1e-2, max_outer: int = 60, verbose: bool = True) -> RegimeResult:
    """Solve one regime.  ``cap`` (a CapacityProblem for the same panel and tau) can be passed
    to share its cut pool across regimes/gammas.

    Note: the price completion runs on every outer iteration and is single-process, which makes it
    the bulk of the runtime.  Deferring it to late iterations was tried and abandoned -- the
    dispatch LPs' degenerate duals understate scarcity rents, which both traps a residual-based
    trigger (the residual never falls) and degrades the secant update enough to need 2-3x the
    outer iterations.  Every variant measured slower than completing every time.  The per-iteration
    cost split is recorded in ``outer_history`` (``secs``, ``completion_s``)."""
    t0 = time.time()
    gamma = params.gamma if gamma is None else gamma
    gamma_z = params.gamma_by_tech(gamma)          # per-technology; uniform unless params.gamma_scaling
    tau = params.tau_scc if regime.carbon_priced else 0.0
    if cap is None or cap.tau != tau:
        cap = CapacityProblem(panel, params, tau)
    names = params.tech_names
    I = params.investment_cost(panel.H)
    psi = {t.name: psi_coefficient(panel, params, t) for t in params.techs}
    lam_hy = panel.lam_hy
    Lambda = float(lam_hy[0].sum())
    K0 = K0 or default_start(panel, params)
    outer_hist: List[dict] = []

    if regime.planner:
        cost = {z: I[z] + psi[z] for z in names}
        cr = cap.solve(cost, K0, verbose=verbose)
        ev = cr.evaluation
        Pbar = (lam_hy * ev.price).sum(axis=1)
        rs = {z: rho(ev.margins[z] - cost[z], 0.0) for z in names}   # planner: risk neutral, cost incl. psi
        return RegimeResult(regime, 0.0, tau, cr.K, I, psi, cost, {z: 0.0 for z in names}, ev.margins, rs, None,
                            ev, cr, [], cr.converged, 0.0, Lambda, Pbar, time.time() - t0, {},
                            {z: 0.0 for z in names})

    # ----------------------------------------------------------- market regimes
    contracts = regime.contracts
    q_bar = panel.q_bar(params) if contracts else 0.0
    markdown = params.markdowns() if regime.reliability == "priced" else {}
    markdown = {t.name: markdown.get(t.vre_key, 0.0) for t in params.vre}
    I_eff = np.array([I[z] for z in names], float)
    I_arr = I_eff.copy()
    Kc = dict(K0)
    # Outer iteration on the effective capacity costs.  Natural map: I_eff <- f.m - rho*.
    # Its slope d(rho*_z)/d(I_eff_z) is >= 1 for active technologies and can reach 1 + gamma*Y
    # when the re-priced scarcity hour falls in the agent's worst year, so a plain damped
    # iteration is slow/oscillatory.  We use a per-technology safeguarded secant (quasi-Newton)
    # step for active technologies; inactive ones (K=0) take the natural step, which is exact
    # for them (their rho* does not depend on their own I_eff).
    converged = False
    resid = np.inf
    cr = fwd = ev = None
    I_solved = I_eff.copy()
    x_prev = None; F_prev = None
    # d(rho*)/d(I_eff) for a technology: rho(pi) = mean_y pi_y + gamma * min_y pi_y, and a unit
    # rise in I_eff lowers every year's pi by one, so the derivative is -(1 + gamma).  It does NOT
    # scale with the number of years: the mean already averages over them.  Using 1 + gamma*Y here
    # made every step (1+gamma*Y)/(1+gamma) times too short, so the residual fell by a fixed factor
    # 1 - 1/(1+gamma*Y) per iteration - 0.90 at gamma=1, needing ~66 iterations and blowing past
    # max_outer.  Measured slopes are ~1.0-1.4 against the 3.7-10 this used to assume.
    g_arr = np.array([gamma_z[z] for z in names], float)   # the slope is per technology, so is gamma
    slope_max = 2.0 * (1.0 + g_arr * panel.Y) + 1.0         # generous ceiling for secant estimates
    slope = 1.0 + g_arr
    max_step_frac = 0.25
    best = None                                             # (x, resid) of the best iterate so far
    best_state = None
    for outer in range(1, max_outer + 1):
        I_solved = I_eff.copy()
        inner_gap = 1e-5 if (not np.isfinite(resid) or resid > 1e-2) else 1e-6   # loose early; completion re-optimises K
        t_outer = time.time()
        cr = cap.solve(dict(zip(names, I_eff)), Kc, tol_gap=inner_gap, verbose=False)
        t_outer = time.time() - t_outer
        ev = cr.evaluation
        Kc = cr.K
        Pbar = (lam_hy * ev.price).sum(axis=1)
        fwd = forward_market(ev.margins, I, cr.K, Pbar, Lambda, gamma_z, q_bar, markdown, contracts)
        rs = np.array([fwd.rho_star[z] for z in names])
        Karr = np.array([cr.K[z] for z in names])
        active = Karr > 1.0
        rel = rs / I_arr
        resid = float(max(np.max(np.abs(rel[active])) if active.any() else 0.0,
                          np.max(rel[~active]) if (~active).any() else 0.0))
        fm = np.array([cap.f @ ev.margins[z] for z in names])
        # Timing split: ``completion_s`` is the monolithic price-completion LP, which is
        # single-process by construction -- extra cores do not touch it.
        rec = {"outer": outer, "resid": resid, "secs": round(t_outer, 1),
               "completion_s": round(cr.completion_time, 1), "evals": cr.n_evaluations,
               "inner_iters": cr.iterations, "inner_foc": cr.foc_residual,
               "p_hat": fwd.p_hat, "K": {z: round(cr.K[z]) for z in names},
               "I_eff": dict(zip(names, I_eff.round(0))), "rho_star": dict(zip(names, rs.round(0)))}
        outer_hist.append(rec)
        if verbose:
            log.info("[%s g=%.3g] outer %2d: resid=%.2e p_hat=%.2f K=%s premium%%=%s", regime.name, gamma, outer, resid,
                     fwd.p_hat, {z: round(cr.K[z]) for z in names},
                     {z: round(100 * (I_eff[i] / I_arr[i] - 1), 2) for i, z in enumerate(names)})
        if resid <= tol and cr.foc_residual <= 3 * tol:
            converged = True
            break
        # A residual that keeps revisiting values it has already hit, with the step limit
        # already tiny, is a stalled search: further iterations cannot improve on ``best``.
        if len(outer_hist) >= 8 and max_step_frac <= _MIN_STEP_FRAC:
            recent = [round(h["resid"], 10) for h in outer_hist[-8:]]
            if len(set(recent)) <= 3 and best is not None and resid >= best[1]:
                log.warning("[%s gamma=%.3g] search stalled at resid=%.2e after %d iterations "
                            "(step limit exhausted); reporting the best iterate",
                            regime.name, gamma, best[1], len(outer_hist))
                break
        # ---- update of the effective costs ----
        F = rs.copy()                                   # driven to 0 (active) / <= 0 (inactive)
        x = I_eff.copy()
        if x_prev is not None:
            dx = x - x_prev
            dF = F - F_prev
            # only trust secant slopes from steps that are large relative to the residual scale
            # (the map is a fine staircase: tiny steps give meaningless slopes)
            meaningful = np.abs(dx) > np.maximum(2e-4 * I_arr, 0.25 * np.abs(F_prev))
            est = np.where(meaningful, dF / np.where(meaningful, dx, 1.0), slope)
            slope = np.where(meaningful & active, np.clip(est, 0.7, slope_max), slope)
        if best is not None and resid > 1.3 * best[1]:
            # Got worse: bisect back towards the best iterate and shrink the step limit.
            # The floor must be small enough that this can always damp further.  With a floor
            # of 0.02 the limit stopped shrinking after four corrections, so the same overshoot
            # repeated forever and the residual ran round a fixed cycle (seen in R6 at high
            # gamma: a period-3/5 loop bottoming out ~1e-2, never reaching tol).
            max_step_frac = max(_MIN_STEP_FRAC, 0.5 * max_step_frac)
            x_new = best[0] + 0.5 * (x - best[0])
            x_prev, F_prev = x, F
            I_eff = np.clip(x_new, 0.2 * I_arr, 6.0 * I_arr)
            continue
        if best is None or resid < best[1]:
            best = (x.copy(), resid)
            best_state = (cr, ev, fwd, I_solved.copy(), dict(Kc), Pbar.copy(), resid)
        x_new = x.copy()
        for i, z in enumerate(names):
            if active[i]:
                step = -F[i] / slope[i]
                lim = max_step_frac * I_arr[i]
                x_new[i] = x[i] + float(np.clip(step, -lim, lim))
            else:
                x_new[i] = fm[i] - F[i]                 # natural (exact) step for inactive technologies
        x_prev, F_prev = x, F
        I_eff = np.clip(x_new, 0.2 * I_arr, 6.0 * I_arr)
    if not converged:
        if best_state is not None and best_state[6] < resid:
            cr, ev, fwd, I_solved, Kc, Pbar, resid = best_state          # report the best iterate, not the last
        log.warning("[%s gamma=%.3g] market equilibrium not converged: best resid=%.2e after %d outer iterations",
                    regime.name, gamma, resid, len(outer_hist))
    cost = dict(zip(names, map(float, I_solved)))
    premium = {z: cost[z] - I[z] for z in names}
    return RegimeResult(regime, gamma, tau, cr.K, I, psi, cost, premium, ev.margins, fwd.rho_star, fwd, ev, cr,
                        outer_hist, converged, q_bar, Lambda, Pbar, time.time() - t0, dict(markdown),
                        dict(gamma_z))
