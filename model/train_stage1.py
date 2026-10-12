"""Stage 1 training: Head A (surface Cp) on the XFOIL data.

Trains on the `train` split only. Early stopping and LR scheduling use `val`.
`test` and `test_unseen_family` are touched only once, at the end, with the
best-on-val checkpoint. Fully reproducible from --seed.

Outputs (under overnight/): stage1_training_log.csv, stage1_results.json,
stage1_results.md, stage1_per_airfoil_error.csv, stage1_curves.png,
stage1_examples.png; checkpoint in overnight/checkpoints/ (gitignored).

The loss is plain MSE on Cp, the same quantity as the headline RMSE.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fno  # noqa: E402
from data_stage1 import SPLIT_NAMES, Stage1Data, cl_from_cp_torch  # noqa: E402

ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "overnight")


@torch.no_grad()
def predict(net, data, idx, bs=256):
    net.eval()
    preds, cls = [], []
    for i in range(0, len(idx), bs):
        b = idx[i:i + bs]
        g_in, sxy, cp, alpha, cl = data.batch(b)
        p = net(g_in, sxy)
        preds.append(p)
        cls.append(cl_from_cp_torch(sxy, p, alpha))
    return torch.cat(preds), torch.cat(cls)


@torch.no_grad()
def metrics(net, data, idx):
    pred, cl_pred = predict(net, data, idx)
    tgt = data.case_cp[idx]
    cl_true = data.case_cl[idx]
    err = pred - tgt
    rmse = float(err.pow(2).mean().sqrt())
    rel = float((err.norm(dim=1) / tgt.norm(dim=1)).mean())
    cl_err = cl_pred - cl_true
    cl_rmse = float(cl_err.pow(2).mean().sqrt())
    ss_res = float(cl_err.pow(2).sum())
    ss_tot = float((cl_true - cl_true.mean()).pow(2).sum())
    return {"n": len(idx), "cp_rmse": rmse, "cp_rel_l2": rel,
            "cl_rmse": cl_rmse, "cl_r2": 1.0 - ss_res / ss_tot,
            "cl_mae": float(cl_err.abs().mean())}


def free_ram_gb() -> float:
    """Free physical RAM in GB (Windows). Returns inf where it cannot be read."""
    try:
        import ctypes

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        m = MEMORYSTATUSEX()
        m.dwLength = ctypes.sizeof(m)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        return m.ullAvailPhys / 2 ** 30
    except Exception:
        return float("inf")


def wait_for_ram(min_gb: float) -> float:
    """Block (checking every 30 s) while free RAM is below min_gb, so a heavy job
    on the same laptop (Fluent) never gets starved. Returns the seconds paused."""
    t = time.time()
    announced = False
    while free_ram_gb() < min_gb:
        if not announced:
            print(f"PAUSED: free RAM {free_ram_gb():.2f} GB < {min_gb:g} GB, waiting", flush=True)
            announced = True
        time.sleep(30)
    if announced:
        print(f"RESUMED after {(time.time() - t) / 60:.1f} min", flush=True)
    return time.time() - t if announced else 0.0


def main() -> int:
    global OUT
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--modes", type=int, default=12)
    p.add_argument("--width", type=int, default=32)
    p.add_argument("--patience", type=int, default=30, help="epochs without val improvement")
    p.add_argument("--time_cap_min", type=float, default=45.0)
    p.add_argument("--tag", default="", help="suffix for every output file (e.g. s43), "
                   "so extra-seed runs never overwrite the main run")
    p.add_argument("--splits", default="", help="splits CSV (default data_generation/splits.csv)")
    p.add_argument("--out_dir", default="", help="output folder (default overnight/)")
    p.add_argument("--threads", type=int, default=2, help="CPU threads (keep low on a shared laptop)")
    p.add_argument("--min_free_gb", type=float, default=3.0,
                   help="pause while free RAM is below this many GB (0 disables)")
    args = p.parse_args()
    sfx = f"_{args.tag}" if args.tag else ""
    if args.out_dir:
        OUT = args.out_dir
    torch.set_num_threads(args.threads)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.min_free_gb > 0:
        wait_for_ram(args.min_free_gb)
    data = Stage1Data(device, args.splits or None)
    eval_splits = [s for s in SPLIT_NAMES[1:] if len(data.indices(s)) > 0]
    net = fno.Stage1Net(width=args.width, modes=args.modes).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=8, min_lr=1e-5)

    tr, va = data.indices("train"), data.indices("val")
    print(f"device {device}; params {fno.count_parameters(net):,}; "
          f"train {len(tr)} val {len(va)} cases; seed {args.seed}", flush=True)

    os.makedirs(os.path.join(OUT, "checkpoints"), exist_ok=True)
    ckpt = os.path.join(OUT, "checkpoints", f"stage1_best{sfx}.pt")
    gen = torch.Generator(device="cpu").manual_seed(args.seed)

    log, best, best_ep, stale = [], float("inf"), -1, 0
    t0 = time.time()
    paused_s = 0.0  # time spent waiting for RAM; not counted against the time cap
    reason = "max epochs"
    for ep in range(1, args.epochs + 1):
        if args.min_free_gb > 0:
            paused_s += wait_for_ram(args.min_free_gb)
        net.train()
        perm = tr[torch.randperm(len(tr), generator=gen).to(device)]
        tot, nb = 0.0, 0
        for i in range(0, len(perm), args.batch):
            g_in, sxy, cp, _, _ = data.batch(perm[i:i + args.batch])
            loss = torch.mean((net(g_in, sxy) - cp) ** 2)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.detach())
            nb += 1
        v = metrics(net, data, va)
        sched.step(v["cp_rmse"])
        row = {"epoch": ep, "train_rmse": (tot / nb) ** 0.5, "val_cp_rmse": v["cp_rmse"],
               "val_cl_rmse": v["cl_rmse"], "lr": opt.param_groups[0]["lr"],
               "minutes": (time.time() - t0 - paused_s) / 60}
        log.append(row)
        if v["cp_rmse"] < best:
            best, best_ep, stale = v["cp_rmse"], ep, 0
            torch.save(net.state_dict(), ckpt)
        else:
            stale += 1
        if ep == 1 or ep % 10 == 0:
            print(f"ep {ep:3d}  train {row['train_rmse']:.4f}  val Cp RMSE {v['cp_rmse']:.4f}  "
                  f"val CL RMSE {v['cl_rmse']:.4f}  lr {row['lr']:.1e}  {row['minutes']:.1f} min "
                  f"(best {best:.4f} @ {best_ep})", flush=True)
        if stale >= args.patience:
            reason = f"early stop: no val improvement for {args.patience} epochs"
            break
        if (time.time() - t0 - paused_s) / 60 >= args.time_cap_min:
            reason = f"time cap {args.time_cap_min:g} min"
            break
    minutes = (time.time() - t0 - paused_s) / 60
    print(f"stopped after {len(log)} epochs, {minutes:.1f} min: {reason}; best val Cp RMSE {best:.4f} @ epoch {best_ep}",
          flush=True)

    with open(os.path.join(OUT, f"stage1_training_log{sfx}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(log[0].keys()))
        w.writeheader()
        w.writerows(log)

    # ---- final evaluation with the best-on-val checkpoint ----
    net.load_state_dict(torch.load(ckpt, map_location=device))
    results = {s: metrics(net, data, data.indices(s)) for s in ["train"] + eval_splits}
    # trivial reference: predict the mean Cp curve of the training set
    tr_mean = data.case_cp[tr].mean(dim=0, keepdim=True)
    for s in eval_splits:
        idx = data.indices(s)
        results[s]["cp_rmse_mean_curve_baseline"] = float(
            (data.case_cp[idx] - tr_mean).pow(2).mean().sqrt())
    summary = {"seed": args.seed, "epochs_run": len(log), "best_epoch": best_ep,
               "stop_reason": reason, "minutes": minutes, "paused_minutes": paused_s / 60,
               "splits_file": args.splits or "data_generation/splits.csv",
               "params": fno.count_parameters(net), "results": results}
    with open(os.path.join(OUT, f"stage1_results{sfx}.json"), "w") as f:
        json.dump(summary, f, indent=2)

    md = ["# Stage 1 results (Head A, Cp on XFOIL data)", "",
          f"Seed {args.seed}. {len(log)} epochs in {minutes:.1f} min ({reason}); "
          f"best validation epoch {best_ep}. {summary['params']:,} parameters.", "",
          "| split | cases | Cp RMSE | Cp relative L2 | CL RMSE (from Cp) | CL R2 | Cp RMSE of a mean-curve guess |",
          "|---|---|---|---|---|---|---|"]
    for s in ["train"] + eval_splits:
        r = results[s]
        md.append(f"| {s} | {r['n']} | {r['cp_rmse']:.4f} | {r['cp_rel_l2']:.4f} | "
                  f"{r['cl_rmse']:.4f} | {r['cl_r2']:.4f} | "
                  f"{r.get('cp_rmse_mean_curve_baseline', float('nan')):.4f} |")
    open(os.path.join(OUT, f"stage1_results{sfx}.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md), flush=True)

    # ---- per-airfoil error (input for the active-selection draft) ----
    allidx = torch.arange(data.n_cases, device=device)
    pred, cl_pred = predict(net, data, allidx)
    err = pred - data.case_cp
    rel = (err.norm(dim=1) / data.case_cp.norm(dim=1)).cpu().numpy()
    clerr = (cl_pred - data.case_cl).abs().cpu().numpy()
    rmse_case = err.pow(2).mean(dim=1).sqrt().cpu().numpy()
    ai = data.case_ai.cpu().numpy()
    split = data.case_split
    rows = []
    for a in np.unique(ai):
        m = ai == a
        aid = data.ids[a]
        rows.append({"airfoil_id": aid, "baseline": data.baseline[aid], "split": split[m][0],
                     "n_cases": int(m.sum()), "mean_cp_rel_l2": float(rel[m].mean()),
                     "mean_cp_rmse": float(rmse_case[m].mean()),
                     "max_cp_rmse": float(rmse_case[m].max()),
                     "mean_abs_cl_err": float(clerr[m].mean())})
    rows.sort(key=lambda r: -r["mean_cp_rel_l2"])
    with open(os.path.join(OUT, f"stage1_per_airfoil_error{sfx}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # ---- plots ----
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ep = [r["epoch"] for r in log]
    ax[0].semilogy(ep, [r["train_rmse"] for r in log], label="train")
    ax[0].semilogy(ep, [r["val_cp_rmse"] for r in log], label="val")
    ax[0].axvline(best_ep, color="k", ls=":", lw=0.8)
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("Cp RMSE"); ax[0].legend(); ax[0].grid(alpha=0.3)
    ax[1].semilogy(ep, [r["lr"] for r in log]); ax[1].set_xlabel("epoch"); ax[1].set_ylabel("learning rate")
    ax[1].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, f"stage1_curves{sfx}.png"), dpi=130); plt.close(fig)

    ex_splits = [s for s in ["test", "test_unseen_family"] if s in eval_splits]
    fig, axes = plt.subplots(len(ex_splits), 3, figsize=(15, 3.5 * len(ex_splits)), squeeze=False)
    for row_i, s in enumerate(ex_splits):
        idx = data.indices(s)
        e = (pred[idx] - data.case_cp[idx]).pow(2).mean(dim=1).sqrt()
        order = torch.argsort(e)
        picks = [order[0], order[len(order) // 2], order[-1]]
        for col, (k, lab) in enumerate(zip(picks, ["best", "median", "worst"])):
            ci = idx[k]
            a = data.ids[int(data.case_ai[ci])]
            axx = axes[row_i, col]
            axx.plot(data.case_cp[ci].cpu().numpy(), label="XFOIL")
            axx.plot(pred[ci].cpu().numpy(), "--", label="model")
            axx.invert_yaxis(); axx.grid(alpha=0.3)
            axx.set_title(f"{s}: {lab}\n{a} ({data.baseline[a]}), alpha {float(data.case_alpha[ci]):g}, "
                          f"RMSE {float(e[k]):.3f}", fontsize=9)
            if col == 0:
                axx.legend(fontsize=8); axx.set_ylabel("Cp")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, f"stage1_examples{sfx}.png"), dpi=120); plt.close(fig)
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
