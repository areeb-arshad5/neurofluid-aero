"""DRAFT active selection: which airfoils is the Stage 1 model most wrong about?

For review by Areeb only. Nothing here is sent to ANSYS.

Method: score every converged case with one or more Stage 1 checkpoints (several
seeds are averaged into an ensemble), rank airfoils by their mean relative Cp
error, and propose the top N with at most --max_per_family from any one baseline
family (so the 15-20 expensive ANSYS cases are not all copies of one hard
family). For each proposed airfoil the angle with the worst error is suggested
as the ANSYS operating point. With several checkpoints it also reports how much
the seeds disagree (a second "where is the model unsure" signal, not used for
the ranking).

Caveats written into the output:
  * Errors on `train` airfoils are optimistic (the model has fitted them), so the
    view that counts is the one restricted to airfoils held out of training.
  * A large error at high alpha on a thin airfoil may reflect XFOIL being
    unreliable near stall rather than the model failing; check before spending
    ANSYS time there.
Needs checkpoints from train_stage1.py (default overnight/checkpoints/stage1_best.pt).
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import Counter, defaultdict

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fno  # noqa: E402
from data_stage1 import Stage1Data  # noqa: E402

ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "overnight")


def predict_all(net, data, device):
    net.eval()
    with torch.no_grad():
        idx = torch.arange(data.n_cases, device=device)
        preds = []
        for i in range(0, len(idx), 256):
            g_in, sxy, cp, alpha, cl = data.batch(idx[i:i + 256])
            preds.append(net(g_in, sxy))
    return torch.cat(preds)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--max_per_family", type=int, default=3)
    p.add_argument("--min_cases", type=int, default=8,
                   help="ignore airfoils with fewer converged angles than this")
    p.add_argument("--ckpts", nargs="+",
                   default=[os.path.join(OUT, "checkpoints", "stage1_best.pt")],
                   help="one or more checkpoints; several are averaged into an ensemble")
    p.add_argument("--splits", default="", help="splits CSV the checkpoints were trained with")
    p.add_argument("--out_dir", default=OUT)
    p.add_argument("--name", default="active_selection_draft", help="output file stem")
    p.add_argument("--threads", type=int, default=2)
    args = p.parse_args()
    torch.set_num_threads(args.threads)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = Stage1Data(device, args.splits or None)
    has_unseen = bool(len(data.indices("test_unseen_family")))

    preds = []
    for ck in args.ckpts:
        net = fno.Stage1Net().to(device)
        net.load_state_dict(torch.load(ck, map_location=device))
        preds.append(predict_all(net, data, device))
    stack = torch.stack(preds)            # (models, cases, 200)
    pred = stack.mean(dim=0)
    tgt = data.case_cp
    rel = ((pred - tgt).norm(dim=1) / tgt.norm(dim=1)).cpu().numpy()
    if len(preds) > 1:
        disagree = (stack.std(dim=0, unbiased=True).norm(dim=1) / tgt.norm(dim=1)).cpu().numpy()
    else:
        disagree = np.full(data.n_cases, np.nan)
    ai = data.case_ai.cpu().numpy()
    alpha = data.case_alpha.cpu().numpy()

    per = defaultdict(list)
    for k in range(data.n_cases):
        per[int(ai[k])].append(k)
    rows = []
    for a, ks in per.items():
        if len(ks) < args.min_cases:
            continue
        aid = data.ids[a]
        worst = ks[int(np.argmax(rel[ks]))]
        rows.append({"airfoil_id": aid, "baseline": data.baseline[aid],
                     "split": data.case_split[ks[0]],
                     "n_cases": len(ks), "mean_cp_rel_l2": float(rel[ks].mean()),
                     "worst_alpha": float(alpha[worst]), "worst_case_rel_l2": float(rel[worst]),
                     "seed_disagreement": float(np.nanmean(disagree[ks])) if len(preds) > 1 else float("nan")})

    def pick(pool):
        pool = sorted(pool, key=lambda r: -r["mean_cp_rel_l2"])
        chosen, fam = [], Counter()
        for r in pool:
            if fam[r["baseline"]] >= args.max_per_family:
                continue
            chosen.append(r)
            fam[r["baseline"]] += 1
            if len(chosen) == args.n:
                break
        return chosen

    views = {"all airfoils (train airfoils included, optimistic)": rows}
    if has_unseen:
        views["held-out airfoils (val, test, unseen family)"] = [r for r in rows if r["split"] != "train"]
    views["val + test airfoils only (never trained on)"] = [r for r in rows if r["split"] in ("val", "test")]

    model_desc = (f"an ensemble of {len(preds)} checkpoints (mean prediction)" if len(preds) > 1
                  else "one checkpoint")
    md = ["# DRAFT active selection (not sent to ANSYS)", "",
          f"Scored with {model_desc}. Top {args.n} airfoils by mean relative Cp error, at most "
          f"{args.max_per_family} per baseline family, airfoils with at least {args.min_cases} "
          "converged angles. `worst alpha` is the suggested ANSYS angle (largest error)."
          + (" `seed disagreement` is how far the seeds' predictions differ, relative to the "
             "target size; it is not used for the ranking." if len(preds) > 1 else ""), "",
          "Caveats:",
          "- Train-split errors are optimistic (the model has fitted those airfoils); the "
          "ranking that counts is the last view, airfoils the model never trained on."]
    if has_unseen:
        md.append("- Airfoils from the held-out families rank high BY CONSTRUCTION: the model never saw "
                  "them. A real selection should retrain on all families first.")
    else:
        md.append("- Every family was seen in training here (all-families split); only the val and "
                  "test airfoils of each family are held out, so the candidate pool for an honest "
                  f"ranking is {len([r for r in rows if r['split'] in ('val', 'test')])} airfoils.")
    md += ["- A large error at high alpha on a thin airfoil may be XFOIL near stall rather than a "
           "model failure; check before spending ANSYS time.",
           "- Check that each suggested angle is a sensible ANSYS operating point before using it.", ""]

    out_rows = []
    for name, pool in views.items():
        chosen = pick(pool)
        md += [f"## {name}", "",
               "| rank | airfoil | baseline | split | mean Cp rel. L2 | worst alpha | worst-case rel. L2 | seed disagreement |",
               "|---|---|---|---|---|---|---|---|"]
        for i, r in enumerate(chosen, 1):
            sd = "n/a" if np.isnan(r["seed_disagreement"]) else f"{r['seed_disagreement']:.3f}"
            md.append(f"| {i} | {r['airfoil_id']} | {r['baseline']} | {r['split']} | "
                      f"{r['mean_cp_rel_l2']:.3f} | {r['worst_alpha']:g} | {r['worst_case_rel_l2']:.3f} | {sd} |")
            out_rows.append({"view": name, "rank": i, **r})
        md += ["", "Families in this list: " + ", ".join(f"{b} x{c}" for b, c in
               Counter(r["baseline"] for r in chosen).most_common()), ""]
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, f"{args.name}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    open(os.path.join(args.out_dir, f"{args.name}.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
