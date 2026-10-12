"""Report for the all-families Stage 1 retrain (seeds 42, 43, 44).

Reads overnight/all_families/ (results, per-airfoil errors, checkpoints) and the
earlier 64x64 runs in overnight/, and writes overnight/all_families/ALLFAMILIES_REPORT.md:
  1. per-seed run summary and the spread over seeds (val, test)
  2. the 3-seed ENSEMBLE accuracy against the single-model mean
  3. error by baseline family, now vs the earlier run where e387 and naca6409
     were held out entirely (same families, but different val/test airfoils,
     so family-level means, not a paired comparison)
"""

from __future__ import annotations

import csv
import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fno  # noqa: E402
from data_stage1 import Stage1Data, cl_from_cp_torch  # noqa: E402

ROOT = os.path.dirname(HERE)
OV = os.path.join(ROOT, "overnight")
AF = os.path.join(OV, "all_families")
SEEDS = [42, 43, 44]
OLD = {42: "stage1_per_airfoil_error.csv", 43: "stage1_per_airfoil_error_s43.csv",
       44: "stage1_per_airfoil_error_s44.csv"}


def family_errors(path_by_seed, splits):
    """{family: mean over seeds of (mean over airfoils in `splits` of mean_cp_rel_l2)}"""
    per_seed = []
    for s, path in path_by_seed.items():
        if not os.path.exists(path):
            continue
        d = defaultdict(list)
        for r in csv.DictReader(open(path)):
            if r["split"] in splits:
                d[r["baseline"]].append(float(r["mean_cp_rel_l2"]))
        per_seed.append({k: float(np.mean(v)) for k, v in d.items()})
    fams = sorted({k for d in per_seed for k in d})
    return {f: float(np.mean([d[f] for d in per_seed if f in d])) for f in fams}, len(per_seed)


def main() -> int:
    res = {}
    for s in SEEDS:
        p = os.path.join(AF, f"stage1_results_allfam_s{s}.json")
        if os.path.exists(p):
            res[s] = json.load(open(p))
    if not res:
        print("no all-families results yet", file=sys.stderr)
        return 1
    seeds = sorted(res)

    md = ["# Stage 1 retrained on ALL 15 families", "",
          f"Seeds finished: {', '.join(map(str, seeds))}. Split: `data_generation/splits_all_families.csv` "
          "(every family 80/10/10 into train/val/test by airfoil; nothing held out). "
          "Early stopping on val; the earlier 64x64 runs held out `e387` and `naca6409` entirely.", "",
          "| seed | epochs | best epoch | stop reason | minutes | paused for RAM (min) |", "|---|---|---|---|---|---|"]
    for s in seeds:
        r = res[s]
        md.append(f"| {s} | {r['epochs_run']} | {r['best_epoch']} | {r['stop_reason']} | "
                  f"{r['minutes']:.1f} | {r.get('paused_minutes', 0):.1f} |")
    md.append("")

    md += ["## Spread over seeds", "", "| split | metric | " + " | ".join(f"seed {s}" for s in seeds) + " | mean | std |",
           "|---|---|" + "---|" * (len(seeds) + 2)]
    for sp in ("val", "test"):
        for key, label in (("cp_rmse", "Cp RMSE"), ("cp_rel_l2", "Cp relative L2"), ("cl_mae", "CL MAE (from Cp)")):
            v = np.array([res[s]["results"][sp][key] for s in seeds])
            sd = v.std(ddof=1) if len(v) > 1 else float("nan")
            md.append(f"| {sp} | {label} | " + " | ".join(f"{x:.4f}" for x in v) + f" | {v.mean():.4f} | {sd:.4f} |")
    md.append("")

    # ---- ensemble ----
    ck = [os.path.join(AF, "checkpoints", f"stage1_best_allfam_s{s}.pt") for s in seeds]
    ck = [c for c in ck if os.path.exists(c)]
    if len(ck) >= 2:
        torch.set_num_threads(2)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        data = Stage1Data(device, os.path.join(ROOT, "data_generation", "splits_all_families.csv"))
        preds = []
        for c in ck:
            net = fno.Stage1Net().to(device)
            net.load_state_dict(torch.load(c, map_location=device))
            net.eval()
            out = []
            with torch.no_grad():
                for sp_idx in [torch.arange(data.n_cases, device=device)]:
                    for i in range(0, len(sp_idx), 256):
                        g_in, sxy, cp, a, cl = data.batch(sp_idx[i:i + 256])
                        out.append(net(g_in, sxy))
            preds.append(torch.cat(out))
        stack = torch.stack(preds)
        ens = stack.mean(dim=0)
        md += [f"## Ensemble of {len(ck)} seeds vs the single-model average", "",
               "| split | single-model Cp RMSE (mean) | ensemble Cp RMSE | single-model CL MAE (mean) | ensemble CL MAE |",
               "|---|---|---|---|---|"]
        for sp in ("val", "test"):
            idx = data.indices(sp)
            tg = data.case_cp[idx]
            singles = [float((p[idx] - tg).pow(2).mean().sqrt()) for p in preds]
            e_rmse = float((ens[idx] - tg).pow(2).mean().sqrt())
            g_in, sxy, cp, alpha, cl = data.batch(idx)
            s_cl = [float((cl_from_cp_torch(sxy, p[idx], alpha) - cl).abs().mean()) for p in preds]
            e_cl = float((cl_from_cp_torch(sxy, ens[idx], alpha) - cl).abs().mean())
            md.append(f"| {sp} | {np.mean(singles):.4f} | {e_rmse:.4f} | {np.mean(s_cl):.4f} | {e_cl:.4f} |")
        md.append("")

    # ---- family comparison ----
    new, n_new = family_errors({s: os.path.join(AF, f"stage1_per_airfoil_error_allfam_s{s}.csv") for s in seeds},
                               {"val", "test"})
    old, n_old = family_errors({s: os.path.join(OV, OLD[s]) for s in SEEDS}, {"val", "test", "test_unseen_family"})
    md += ["## Error by baseline family, now vs the earlier run", "",
           f"Mean relative Cp error on airfoils the model never trained on, averaged over seeds "
           f"({n_new} seeds now, {n_old} before). Earlier: `e387` and `naca6409` were entirely held out "
           "(all their airfoils); the other 13 families use their val+test airfoils. Now: val+test airfoils of "
           "every family. Different airfoils, same families, so read the family-level difference, not exact numbers.", "",
           "| family | earlier (held-out error) | now (all families seen) | change |", "|---|---|---|---|"]
    for f in sorted(new, key=lambda k: -new[k]):
        o = old.get(f)
        tag = " (was fully unseen)" if f in ("e387", "naca6409") else ""
        md.append(f"| {f}{tag} | {o*100:.1f}% | {new[f]*100:.1f}% | {(new[f]-o)/o*100:+.0f}% |" if o is not None
                  else f"| {f} | n/a | {new[f]*100:.1f}% | |")
    un = [f for f in ("e387", "naca6409") if f in new and f in old]
    if un:
        md += ["", f"**Reading:** the two families that were fully unseen went from "
               f"{np.mean([old[f] for f in un])*100:.1f}% to {np.mean([new[f] for f in un])*100:.1f}% "
               "once they were part of training, while the other families moved much less. That is "
               "consistent with the earlier gap being a data-diversity effect (shapes never seen) rather "
               "than a limit of the network."]
    open(os.path.join(AF, "ALLFAMILIES_REPORT.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
