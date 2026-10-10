"""
Fit our data-constrained scaling law with ensembling,

    L = E + A / N^alpha + B / (UR)^beta + P (R-1)^delta (N / U^gamma)^kappa
          - F / (N^nu U^eps R^rho) * (1 - 1/M)^mu

to a CSV of (U, d, R, M, val_loss) rows, one per rung, e.g. scores.csv.
Huber loss on log-loss, each rung weighted by its compute C = 6 N U R M.

Usage:
    python fit_scaling_law.py scores.csv
"""

import argparse

import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize

parser = argparse.ArgumentParser()
parser.add_argument("csv", nargs="?", default="scores.csv")
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

HUBER_DELTA = 1e-3
SCALE = 1e8    # N, U and UR are fit in units of 1e8
R_REF = 16     # and R in units of 16 in the ensembling term

df = pd.read_csv(args.csv)
dupes = df[df.duplicated(["U", "d", "R", "M"], keep=False)]
if len(dupes):
    raise SystemExit(f"multiple rows for the same (U, d, R, M):\n{dupes}")

U = torch.tensor(df.U.values * 1e6, dtype=torch.float64)
N = torch.tensor(49152 * df.d.values**3 + 3219456 * df.d.values, dtype=torch.float64)
R = torch.tensor(df.R.values, dtype=torch.float64)
M = torch.tensor(df.M.values, dtype=torch.float64)
y = torch.tensor(np.log(df.val_loss.values), dtype=torch.float64)
C = 6 * N * U * R * M
w = C / C.mean()

n, u, ur = torch.log(N / SCALE), torch.log(U / SCALE), torch.log(U * R / SCALE)
log_r1 = torch.log(torch.clamp(R - 1, min=1))
ens = torch.where(M > 1, torch.log(1 - 1 / M), torch.zeros_like(M))

# theta: log E, log A, alpha, log B, beta, log P, delta, kappa, gamma, F, mu, nu, eps, rho
BOUNDS = [(-8, np.log(3)), (-8, 4), (0.01, 2), (-8, 4), (0.01, 2),
          (-14, 3), (0, 3), (0.01, 3), (0, 30),
          (0, 1.5), (0.05, 3), (-0.5, 1.5), (-0.5, 1.5), (-0.5, 1.5)]


def predict_log_loss(th):
    # th: (..., 14), vectorized over leading dims
    th = th[..., None]
    e, a, al, b, be, p, de, ka, ga, F, mu, nu, eps, rho = th.unbind(-2)
    penalty = torch.where(R > 1, p + de * log_r1 + ka * (n - ga * u), -torch.inf)
    L1 = torch.logsumexp(torch.stack([e + 0 * n, a - al * n, b - be * ur, penalty]), 0).exp()
    gain = F * torch.exp(-nu * n - eps * u - rho * torch.log(R / R_REF)) * torch.where(M > 1, torch.exp(mu * ens), 0)
    return torch.log(L1 - gain)


def objective(th):
    r = (predict_log_loss(th) - y).abs()
    huber = torch.where(r <= HUBER_DELTA, 0.5 * r**2, HUBER_DELTA * (r - 0.5 * HUBER_DELTA))
    return (w * huber).sum(-1) / w.sum()


def f(x):
    th = torch.tensor(x, dtype=torch.float64, requires_grad=True)
    loss = 1e4 * objective(th)  # scaled so scipy's tolerances don't stop early
    if not torch.isfinite(loss):
        return np.inf, np.zeros_like(x)
    loss.backward()
    return loss.item(), th.grad.numpy()


def polish(x0, maxiter):
    return minimize(f, x0, jac=True, method="L-BFGS-B", bounds=BOUNDS,
                    options=dict(maxiter=maxiter, ftol=1e-12, gtol=1e-10))


# random screen -> short polish of the best starts -> full polish -> a few random hops
rng = np.random.default_rng(args.seed)
k = 8192
starts = np.stack([
    np.log(rng.uniform(0.1, 3.0, k)), np.log(rng.uniform(0.02, 5.0, k)), rng.uniform(0.05, 1.2, k),
    np.log(rng.uniform(0.02, 5.0, k)), rng.uniform(0.05, 1.2, k),
    np.log(10 ** rng.uniform(-4, 0, k)), rng.uniform(0.05, 1.5, k), rng.uniform(0.05, 1.5, k), rng.uniform(0.2, 8.0, k),
    rng.uniform(0.03, 0.4, k), rng.uniform(0.3, 1.3, k), rng.uniform(-0.1, 0.6, k), rng.uniform(-0.1, 0.6, k),
    rng.uniform(-0.1, 0.5, k)], 1)
with torch.no_grad():
    screen = objective(torch.tensor(starts)).nan_to_num(torch.inf).numpy()
short = sorted((polish(x0, 200) for x0 in starts[np.argsort(screen)[:24]]), key=lambda r: r.fun)
best = min((polish(r.x, 1000) for r in short[:6]), key=lambda r: r.fun)
lo, hi = np.array(BOUNDS).T
for sigma in (0.5, 0.25, 0.25, 0.1, 0.1, 0.05):
    r = polish(np.clip(best.x + rng.normal(0, sigma, len(lo)), lo, hi), 1000)
    if r.fun < best.fun:
        best = r

e, a, al, b, be, p, de, ka, ga, F, mu, nu, eps, rho = best.x
params = dict(E=np.exp(e), A=np.exp(a) * SCALE**al, alpha=al, B=np.exp(b) * SCALE**be, beta=be,
              P=np.exp(p) * SCALE**(ka * (ga - 1)), delta=de, kappa=ka, gamma=ga,
              F=F * SCALE**(nu + eps) * R_REF**rho, mu=mu, nu=nu, epsilon=eps, rho=rho)
print(f"fit on {len(df)} rungs, weighted Huber loss {best.fun / 1e4:.3e}")
for name, v in params.items():
    print(f"  {name:<8} {v:.4g}")
