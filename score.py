"""
Score an ensemble of M seeds of one (U, d, R, eLR, WD) rung. All snapshots of
all seeds are candidates; the ensembling strategy (which snapshots, how they
are weighted) is chosen on the calibration split and reported on val.

Usage:
    python score.py -U 100 -d 11 -R 8 -M 4
"""

import os
import csv
import json
import argparse

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("-U", type=float, required=True, help="unique training tokens, in millions")
parser.add_argument("-d", type=int, required=True)
parser.add_argument("-R", type=int, required=True)
parser.add_argument("-M", type=int, default=1, help="ensemble size: uses seeds 0..M-1")
parser.add_argument("-elr", type=float, default=None, help="default: our law")
parser.add_argument("-wd", type=float, default=None, help="default: our law")
parser.add_argument("--runs-csv", type=str, default="runs.csv")
parser.add_argument("--scores-csv", type=str, default="scores.csv")
args = parser.parse_args()

AGE_DECAY = (0.5, 1.0, 2.0, 4.0, 8.0)          # in snapshots
SOFTMAX_TEMPS = (0.005, 0.01, 0.02, 0.05, 0.1)  # in nats
EM_SHORTLIST = 4

U = args.U * 1e6
N = 49152 * args.d**3 + 3219456 * args.d
elr = args.elr if args.elr is not None else float(f"{3.953e4 * U**-1.0141 * N**0.2735 * args.R**0.1199:.4g}")
wd = args.wd if args.wd is not None else float(f"{107.5 * U**-0.8702 * N**0.3978 * args.R**0.6600:.4g}")

# find seeds 0..M-1 in runs.csv
close = lambda a, b: abs(float(a) / b - 1) < 1e-6
runs = {}
with open(args.runs_csv) as f:
    for row in csv.DictReader(f):
        if (close(row["U"], args.U) and int(row["d"]) == args.d and int(row["R"]) == args.R
                and close(row["eLR"], elr) and close(row["WD"], wd)):
            runs[int(row["seed"])] = row["path"]
missing = [s for s in range(args.M) if s not in runs]
if missing:
    raise SystemExit(f"U={args.U:g}M d={args.d} R={args.R} eLR={elr:g} WD={wd:g}: "
                     f"missing seeds {missing} in {args.runs_csv}")


def load(path, split):
    info = json.load(open(os.path.join(path, "args.json")))
    return np.memmap(os.path.join(path, f"{split}_nll.f16"), dtype=np.float16, mode="r").reshape(-1, info[f"{split}_tokens"])


# every snapshot of every seed, with its age (0 = that run's final snapshot)
calib, val, age = [], [], []
for seed in range(args.M):
    calib.append(load(runs[seed], "calib"))
    val += list(load(runs[seed], "val"))
    age += list(range(len(calib[-1]) - 1, -1, -1))
calib_nll = np.concatenate(calib).astype(np.float32)
calib_p = np.exp(-calib_nll)
age = np.array(age)
n = len(age)


def scan(groups, w):
    """Calib loss of groups[:k] for every k, adding one group at a time."""
    mix, total, losses = np.zeros(calib_p.shape[1]), 0.0, []
    for g in groups:
        mix += w[g] @ calib_p[g]
        total += w[g].sum()
        losses.append(-np.log(np.maximum(mix / total, 1e-30)).mean())
    return losses


def val_loss(rows, w):
    mix = np.zeros(val[0].shape[0])
    for i, wi in zip(rows, w / w.sum()):
        mix += wi * np.exp(-val[i].astype(np.float32))
    return -np.log(np.maximum(mix, 1e-30)).mean()


def em_weights(rows, iters=30, tol=1e-6):
    p = calib_p[rows].astype(np.float64)
    w = np.full(len(rows), 1 / len(rows))
    for _ in range(iters):
        new = (w[:, None] * p / np.maximum(w @ p, 1e-30)).mean(1)
        new /= new.sum()
        if np.abs(new - w).max() < tol:
            return new
        w = new
    return w


member_calib = calib_nll.mean(1)
weightings = {"equal": np.ones(n), "linear-age": age.max() + 1.0 - age}
for tau in AGE_DECAY:
    weightings[f"age-decay{tau:g}"] = np.exp(-age / tau)
for temp in SOFTMAX_TEMPS:
    weightings[f"softmax{temp:g}"] = np.exp(-(member_calib - member_calib.min()) / temp)

# recency: the last k snapshots of every seed; quality: the k best snapshots overall
scored = []
recency = [np.flatnonzero(age == j) for j in range(age.max() + 1)]
for name, w in weightings.items():
    for k, loss in enumerate(scan(recency, w), 1):
        rows = np.flatnonzero(age < k)
        scored.append((loss, f"last{k}-{name}", rows, w[rows]))
quality = [[i] for i in np.argsort(member_calib)]
for k, loss in enumerate(scan(quality, np.ones(n)), 1):
    scored.append((loss, f"top{k}-equal", np.argsort(member_calib)[:k], np.ones(k)))

# fitted weights, only at the most promising subset sizes
best_sizes = {len(rows) for _, _, rows, _ in sorted(scored, key=lambda s: s[0])[:EM_SHORTLIST]}
for k in range(1, age.max() + 2):
    rows = np.flatnonzero(age < k)
    if len(rows) >= 2 and len(rows) in best_sizes:
        w = np.zeros(n)
        w[rows] = em_weights(rows)
        scored.append((scan([rows], w)[0], f"last{k}-fitted", rows, w[rows]))

calib_loss, strategy, rows, w = min(scored, key=lambda s: s[0])
loss = val_loss(rows, w)
print(f"U={args.U:g}M d={args.d} R={args.R} M={args.M} eLR={elr:g} WD={wd:g}: "
      f"{strategy} ({len(rows)} of {n} snapshots) | calib {calib_loss:.4f} | val {loss:.4f}")

# one row per (U, d, R, M, eLR, WD)
header = ["U", "d", "R", "M", "eLR", "WD", "val_loss", "calib_loss", "strategy"]
new = dict(U=f"{args.U:g}", d=args.d, R=args.R, M=args.M, eLR=f"{elr:g}", WD=f"{wd:g}",
           val_loss=f"{loss:.6f}", calib_loss=f"{calib_loss:.6f}", strategy=strategy)
rows_out = []
if os.path.exists(args.scores_csv):
    with open(args.scores_csv) as f:
        key = lambda r: (float(r["U"]), int(r["d"]), int(r["R"]), int(r["M"]), float(r["eLR"]), float(r["WD"]))
        rows_out = [r for r in csv.DictReader(f) if key(r) != key(new)]
with open(args.scores_csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=header)
    writer.writeheader()
    writer.writerows(rows_out + [new])
