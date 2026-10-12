"""Does --resume reproduce an uninterrupted run?  Run: python model/test_resume.py

Trains 4 epochs straight (tag t_full), then 4 epochs as 2 + resume 2 (tag t_res),
in a scratch folder, and compares the per-epoch training log. The two runs use
the same seed and data; they should agree epoch by epoch. Needs about 1.4 GB of
free RAM per run (one at a time) and the GPU. Exits nonzero on a mismatch.
"""

from __future__ import annotations

import csv
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SPLITS = os.path.join(ROOT, "data_generation", "splits_all_families.csv")


def run(args, out_dir):
    cmd = [sys.executable, os.path.join(HERE, "train_stage1.py"), "--seed", "42", "--epochs", "4",
           "--splits", SPLITS, "--out_dir", out_dir, "--min_free_gb", "0", "--patience", "999"] + args
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.returncode, r.stdout


def rows(out_dir, tag):
    with open(os.path.join(out_dir, f"stage1_training_log_{tag}.csv")) as f:
        return list(csv.DictReader(f))


def main() -> int:
    out = tempfile.mkdtemp(prefix="resume_test_")
    ok = True

    code, _ = run(["--tag", "t_full"], out)
    print(f"uninterrupted run exit {code}")
    ok &= code == 0

    code, so = run(["--tag", "t_res", "--debug_exit_after", "2"], out)
    print(f"interrupted run exit {code} (expect 75); state saved: "
          f"{os.path.exists(os.path.join(out, 'checkpoints', 'stage1_resume_t_res.pt'))}")
    ok &= code == 75
    code, so = run(["--tag", "t_res", "--resume"], out)
    print(f"resumed run exit {code} (expect 0)")
    print("  " + next((ln for ln in so.splitlines() if "RESUMED" in ln), "NO RESUME LINE"))
    ok &= code == 0

    a, b = rows(out, "t_full"), rows(out, "t_res")
    print(f"epochs logged: uninterrupted {len(a)}, resumed {len(b)}")
    ok &= len(a) == len(b) == 4
    worst = 0.0
    for ra, rb in zip(a, b):
        for k in ("train_rmse", "val_cp_rmse", "val_cl_rmse"):
            worst = max(worst, abs(float(ra[k]) - float(rb[k])))
        print(f"  epoch {ra['epoch']}: val Cp RMSE {float(ra['val_cp_rmse']):.6f} vs {float(rb['val_cp_rmse']):.6f}")
    print(f"largest difference over train/val metrics: {worst:.2e}")
    ok &= worst < 5e-4   # GPU kernels are not bit-deterministic; 5e-4 is far below any real effect
    gone = not os.path.exists(os.path.join(out, "checkpoints", "stage1_resume_t_res.pt"))
    print(f"resume state removed after a finished run: {gone}")
    ok &= gone
    print("RESUME TEST " + ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
