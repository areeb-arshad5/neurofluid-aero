"""Train / val / test split of the 500 airfoils (Phase 1, split step).

Rules:
  - The unit of splitting is the AIRFOIL, never the (airfoil, alpha) case, so
    the same shape can never sit in both train and test.
  - Within each baseline family the airfoils are split 80/10/10 at random.
    Variants of one baseline are only ~10% apart, so this "test" split measures
    interpolation within families the model has seen.
  - Two WHOLE baseline families are held out as `test_unseen_family`: the model
    never sees any variant of them. This is the honest generalisation test.
  - Fully reproducible from --seed (default 42).

Output: splits.csv with airfoil_id, baseline, split, n_converged.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import Counter, defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

N_HELD_OUT_FAMILIES = 2
VAL_FRAC = 0.10
TEST_FRAC = 0.10


def make_splits(index_rows, converged, seed: int, n_held_out: int = N_HELD_OUT_FAMILIES):
    rng = np.random.default_rng(seed)
    by_family = defaultdict(list)
    for r in index_rows:
        by_family[r["baseline"]].append(r["airfoil_id"])

    families = sorted(by_family)
    held_out = (sorted(rng.choice(families, size=n_held_out, replace=False).tolist())
                if n_held_out > 0 else [])

    assign = {}
    for fam in families:
        ids = sorted(by_family[fam])
        if fam in held_out:
            for a in ids:
                assign[a] = "test_unseen_family"
            continue
        order = rng.permutation(len(ids))
        n_test = max(1, round(TEST_FRAC * len(ids)))
        n_val = max(1, round(VAL_FRAC * len(ids)))
        for rank, k in enumerate(order):
            a = ids[k]
            assign[a] = "test" if rank < n_test else ("val" if rank < n_test + n_val else "train")

    rows = [{"airfoil_id": r["airfoil_id"], "baseline": r["baseline"],
             "split": assign[r["airfoil_id"]],
             "n_converged": converged.get(r["airfoil_id"], 0)} for r in index_rows]
    return rows, held_out


def run_checks(rows, held_out, index_rows, seed, converged, n_held_out=N_HELD_OUT_FAMILIES) -> bool:
    """Known-good properties, plus a known-bad detector for leakage."""
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
        ok &= bool(cond)

    ids = [r["airfoil_id"] for r in rows]
    check("every airfoil appears exactly once", len(ids) == len(set(ids)) == len(index_rows))
    check("every airfoil is assigned a split",
          all(r["split"] in {"train", "val", "test", "test_unseen_family"} for r in rows))
    fam_split = defaultdict(set)
    for r in rows:
        fam_split[r["baseline"]].add(r["split"])
    check("held-out families appear ONLY in test_unseen_family",
          all(fam_split[f] == {"test_unseen_family"} for f in held_out))
    check("no held-out family leaks into train/val/test",
          all(not (fam_split[f] & {"train", "val", "test"}) for f in held_out))
    check("every other family appears in train, val and test",
          all(fam_split[f] == {"train", "val", "test"} for f in fam_split if f not in held_out))
    again, _ = make_splits(index_rows, converged, seed, n_held_out)
    check("same seed reproduces the identical split", again == rows)
    other, _ = make_splits(index_rows, converged, seed + 1, n_held_out)
    check("a different seed changes the split", other != rows)

    if held_out:
        # known-bad: the leakage detector must catch a deliberately leaked family
        bad = [dict(r) for r in rows]
        for r in bad:
            if r["baseline"] == held_out[0]:
                r["split"] = "train"
                break
        leak = {r["baseline"]: set() for r in bad}
        for r in bad:
            leak[r["baseline"]].add(r["split"])
        caught = leak[held_out[0]] != {"test_unseen_family"}
        check("known-bad: a leaked held-out airfoil is detected", caught)
    else:
        check("no family is held out (all families are seen in training)",
              not any(r["split"] == "test_unseen_family" for r in rows))
        # known-bad: dropping every val airfoil of one family must be detected
        fam = rows[0]["baseline"]
        bad_splits = defaultdict(set)
        for r in rows:
            if not (r["baseline"] == fam and r["split"] == "val"):
                bad_splits[r["baseline"]].add(r["split"])
        check("known-bad: a family missing from val is detected",
              bad_splits[fam] != {"train", "val", "test"})
    return ok


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--index_csv", default=os.path.join(HERE, "airfoil_index.csv"))
    p.add_argument("--status_csv", default=os.path.join(HERE, "xfoil_data", "status.csv"))
    p.add_argument("--out", default=os.path.join(HERE, "splits.csv"))
    p.add_argument("--n_held_out", type=int, default=N_HELD_OUT_FAMILIES,
                   help="whole families held out as test_unseen_family (default 2). "
                        "0 = all families are seen (use with --out splits_all_families.csv)")
    args = p.parse_args()

    with open(args.index_csv, newline="") as f:
        index_rows = list(csv.DictReader(f))
    converged = Counter()
    with open(args.status_csv, newline="") as f:
        for r in csv.DictReader(f):
            if r["status"] == "converged":
                converged[r["airfoil_id"]] += 1

    rows, held_out = make_splits(index_rows, converged, args.seed, args.n_held_out)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["airfoil_id", "baseline", "split", "n_converged"])
        w.writeheader()
        w.writerows(rows)

    print(f"seed {args.seed}; held-out families: {held_out}\n")
    n_air, n_case = Counter(), Counter()
    for r in rows:
        n_air[r["split"]] += 1
        n_case[r["split"]] += r["n_converged"]
    print(f"{'split':22s}{'airfoils':>10s}{'converged cases':>18s}")
    for s in ("train", "val", "test", "test_unseen_family"):
        print(f"{s:22s}{n_air[s]:10d}{n_case[s]:18d}")
    zero = [r["airfoil_id"] for r in rows if r["n_converged"] == 0]
    print(f"\nairfoils with no converged case (no training data): {zero}\n")

    print("Checks:")
    return 0 if run_checks(rows, held_out, index_rows, args.seed, converged, args.n_held_out) else 1


if __name__ == "__main__":
    sys.exit(main())
