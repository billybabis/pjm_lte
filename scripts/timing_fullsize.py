"""Full-size timing run on synthetic data: 9 years x 8760 h (same size as PJM 2017-2025)."""
import logging, time, sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S", stream=sys.stdout)
from eq_model.synthetic import make_synthetic_panel
from eq_model.params import ModelParams, REGIMES
from eq_model.equilibrium import CapacityProblem, solve_regime
from eq_model.welfare import evaluate_welfare, summary_row
import pandas as pd

years = tuple(range(2017, 2026))
t = time.time()
panel = make_synthetic_panel(years=years)
print("panel built", round(time.time() - t, 1), "s", panel.summary(), flush=True)
P = ModelParams(gamma=0.3)
cap = CapacityProblem(panel, P, tau=0.0)
rows = []
t = time.time(); p1 = solve_regime(panel, P, REGIMES["P1"], cap=cap, verbose=True)
print("P1 TOTAL", round(time.time() - t, 1), "s; iters", p1.capacity_result.iterations, "evals", cap.n_eval,
      "completion", round(p1.capacity_result.completion_time, 1), "s; K", {k: round(v) for k, v in p1.K.items()},
      "FOC", p1.capacity_result.foc_residual, flush=True)
rows.append(summary_row(evaluate_welfare(panel, P, p1), p1, P))
for reg in ["R2", "R3"]:
    t = time.time(); r = solve_regime(panel, P, REGIMES[reg], cap=cap, K0=p1.K, verbose=True)
    print(reg, "TOTAL", round(time.time() - t, 1), "s; outer", len(r.outer_history), "evals", cap.n_eval, "conv", r.converged,
          "K", {k: round(v) for k, v in r.K.items()}, "premium%", {z: round(100 * r.risk_premium[z] / r.I[z], 2) for z in r.K if r.K[z] > 1},
          "p_hat", r.forward.p_hat if r.forward else None, flush=True)
    rows.append(summary_row(evaluate_welfare(panel, P, r), r, P))
pd.DataFrame(rows).to_csv("/tmp/claude-0/-home-claude/440e866a-d0dc-519b-9242-a0ad706b16e1/scratchpad/timing_summary.csv", index=False)
print("DONE", flush=True)
