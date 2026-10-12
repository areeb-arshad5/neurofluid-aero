"""Stage 1 dataset: converged XFOIL cases as (grid input, surface points, Cp).

Everything is held on the target device as tensors; the per-airfoil fields
(sdf, mask) are stored once and the 6-channel input for a batch is assembled on
the fly, so memory stays small (the dense 7,075 x 6 x 64 x 64 tensor would be
~700 MB).

Splits come from data_generation/splits.csv (by airfoil, never by case).
"""

from __future__ import annotations

import csv
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DG = os.path.join(ROOT, "data_generation")
sys.path.insert(0, DG)
import grid_encoding as ge  # noqa: E402

RE = 1.0e6          # all XFOIL cases use Re = 1e6 (data_spec.md)
SPLIT_NAMES = ["train", "val", "test", "test_unseen_family"]


def cl_from_cp_torch(surf_xy: torch.Tensor, cp: torch.Tensor, alpha_deg: torch.Tensor):
    """Batched lift coefficient from Cp, same formula as qa_xfoil.cl_from_cp
    (CCW Selig order, closing segment included). surf_xy (B, N, 2), cp (B, N)."""
    x, y = surf_xy[..., 0], surf_xy[..., 1]
    x2, y2, cp2 = x.roll(-1, dims=1), y.roll(-1, dims=1), cp.roll(-1, dims=1)
    cpm = 0.5 * (cp + cp2)
    cy = (cpm * (x2 - x)).sum(dim=1)
    cx = -(cpm * (y2 - y)).sum(dim=1)
    a = torch.deg2rad(alpha_deg)
    return cy * torch.cos(a) - cx * torch.sin(a)


class Stage1Data:
    def __init__(self, device: str | torch.device = "cpu", splits_file: str | None = None):
        """splits_file: CSV with airfoil_id, split (default data_generation/splits.csv)."""
        self.device = torch.device(device)
        g = np.load(os.path.join(DG, "grids", "airfoil_grids.npz"), allow_pickle=False)
        ids = [str(a) for a in g["airfoil_id"]]
        self.ids = ids
        self.sdf = torch.from_numpy(g["sdf"]).float().to(self.device)       # (A, 64, 64)
        self.mask = torch.from_numpy(g["mask"]).float().to(self.device)     # (A, 64, 64)
        self.xg = torch.from_numpy(g["x"]).float().to(self.device)
        self.yg = torch.from_numpy(g["y"]).float().to(self.device)
        self.re_value = ge.re_channel_value(RE)

        with open(splits_file or os.path.join(DG, "splits.csv"), newline="") as f:
            split_of = {r["airfoil_id"]: r["split"] for r in csv.DictReader(f)}
        with open(os.path.join(DG, "airfoil_index.csv"), newline="") as f:
            self.baseline = {r["airfoil_id"]: r["baseline"] for r in csv.DictReader(f)}

        surf = np.zeros((len(ids), 200, 2), dtype=np.float32)
        case_ai, case_alpha, case_cp, case_cl, case_split = [], [], [], [], []
        for ai, aid in enumerate(ids):
            p = os.path.join(DG, "xfoil_data", "cp", f"{aid}.npz")
            if not os.path.exists(p):
                continue  # airfoil_0192: no converged case
            d = np.load(p, allow_pickle=False)
            surf[ai, :, 0], surf[ai, :, 1] = d["x"], d["y"]
            for k in range(len(d["alpha"])):
                case_ai.append(ai)
                case_alpha.append(float(d["alpha"][k]))
                case_cp.append(d["cp"][k])
                case_cl.append(float(d["CL"][k]))
                case_split.append(split_of[aid])

        t = lambda a, dt=torch.float32: torch.as_tensor(np.asarray(a), dtype=dt, device=self.device)
        self.surf_xy = t(surf)                       # (A, 200, 2)
        self.case_ai = t(case_ai, torch.long)        # (n,)
        self.case_alpha = t(case_alpha)              # (n,) degrees
        self.case_cp = t(np.stack(case_cp))          # (n, 200)
        self.case_cl = t(case_cl)                    # (n,)
        self.case_split = np.array(case_split)
        self.n_cases = len(case_ai)

    def indices(self, split: str) -> torch.Tensor:
        return torch.as_tensor(np.where(self.case_split == split)[0], device=self.device)

    def batch(self, idx: torch.Tensor):
        """idx: 1-D tensor of case indices -> (grid_in (B,6,64,64), surf_xy (B,200,2),
        cp (B,200), alpha_deg (B,), cl (B,))."""
        ai = self.case_ai[idx]
        alpha = self.case_alpha[idx]
        b = len(idx)
        grid_in = torch.empty(b, 6, ge.GRID_N, ge.GRID_N, device=self.device)
        grid_in[:, 0] = self.sdf[ai]
        grid_in[:, 1] = self.mask[ai]
        grid_in[:, 2] = self.xg
        grid_in[:, 3] = self.yg
        grid_in[:, 4] = (alpha / ge.ALPHA_SCALE).view(b, 1, 1)
        grid_in[:, 5] = self.re_value
        return grid_in, self.surf_xy[ai], self.case_cp[idx], alpha, self.case_cl[idx]
