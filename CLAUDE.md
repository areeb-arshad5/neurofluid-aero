# NeuroFluid-Aero

## What this project is
A physics-informed Fourier Neural Operator (FNO) that predicts airflow around
2D airfoils (surface pressure, and eventually full velocity/pressure fields)
near-instantly, trained with a **multi-fidelity curriculum**: cheap/fast
XFOIL data first, then a small amount of expensive/precise ANSYS CFD data
to fine-tune. Originally an academic project, now reframed as a portfolio
piece. Team of 2.

## The core hypothesis we're testing
Pre-training on lots of cheap XFOIL data, then fine-tuning on only a
*small* number of expensive ANSYS cases, should reach the same accuracy as
training on ANSYS data alone with far more cases. The main result of the
whole project is one plot: test error vs. number of ANSYS training cases,
with-pretraining vs. from-scratch. Everything else supports that plot.

## Architecture (agreed design - don't deviate without discussion)
- Geometry is encoded as a grid, not raw coordinates: a 64x64 grid with
  channels [signed distance function (SDF), inside/outside mask, x, y,
  angle of attack, Reynolds number] (~6 input channels). This is what
  makes a 2D spectral convolution (FNO) valid - it needs a field on a
  grid, not a point list.
- Shared trunk: 4 stacked Fourier layers (SpectralConv2d + 1x1 conv skip +
  GELU), width 32, modes ~12 to start (this will be swept in Phase 3).
- Two output heads sharing the same trunk:
  - Head A: samples the trunk's latent field at ~200 surface points ->
    small MLP -> predicts Cp (pressure coefficient curve). Trained first,
    on XFOIL data.
  - Head B: MLP that takes [latent(x,y), x, y, angle of attack, Reynolds
    number] -> predicts (u, v, p) at that point. Trained second, on
    ANSYS data, fine-tuning the trunk at a low learning rate.
- Physics loss: conservation of mass (divergence-free, div(u)=0) is the
  PRIMARY physics term - it's exact for RANS mean-flow data and doesn't
  require assuming laminar flow. A momentum-residual term is SECONDARY,
  weighted an order of magnitude lower, and ramped in gradually during
  Stage 2/3 training (not applied at full strength from step 1). Physics
  residuals are computed only in the outer field, masked away from the
  wall, since a 64x64 grid can't resolve a boundary layer at Re~10^6.
- Any physics-loss function must be unit-tested against an analytic flow
  with a known-zero residual (e.g. potential flow over a cylinder) BEFORE
  being trusted on real training data.

## Data pipeline (three tiers, in order of cost)
1. **XFOIL** (cheap, ~seconds/case): 500 CST-parameterized airfoils,
   swept across multiple angles of attack (at least -4 deg to 10 deg),
   fixed Mach/Reynolds range. Produces Cp curves, CL, CD, CM, and
   transition location (Top_Xtr / Bot_Xtr).
2. **SU2 RANS** (medium cost) - optional/stretch, only if ahead of
   schedule. Not required for the core hypothesis.
3. **ANSYS Fluent RANS** (expensive, minutes/case + meshing overhead):
   small number of cases (target ~15-20), selected by ACTIVE SELECTION
   (pick the airfoils the Stage-1 model is most wrong about), not
   randomly. Exported on the same 64x64 grid as everything else.

All three tiers must use the SAME units, coordinate frame, and grid
definition - defined once in `data_spec.md` and never redefined ad hoc.

## Repo conventions
- One shared `data_spec.md` at repo root defining exact tensor shapes,
  file formats, and non-dimensionalization. Read it before writing any
  data-producing code.
- Every data-generation script takes a `--seed` argument and is fully
  reproducible.
- Any filter/validity-check function (geometry validity, physics residual,
  etc.) gets a unit test against known-good and known-bad cases before
  being used on real data.
- Overfit-a-single-sample test required before trusting any new training
  loop on the full dataset.

## Reference papers (cite by author/year, not "the literature")
- Teimourian et al. 2024 (Heliyon) - classical ML (RF/GBR/DTR/AdaBoost/LR)
  predicting lift-to-drag ratio from airfoil images. Used as a baseline
  only, not a component.
- Moin et al. 2021 - ANN on airfoil coordinates predicting CL/CD/Cm;
  found CD hardest to predict (lowest R^2) since drag is quadratic in AoA
  while CL/Cm are near-linear. Expect the same pattern here.
- Chen et al. 2020 - CNN on composite airfoil images (flow conditions
  convolved into the image) predicting Cmz/CD/CL together; beat kriging
  on training-time scaling.
- Balla et al. 2021 - multi-output NN predicting full pressure fields
  (not direct coefficients) then integrating; found this more accurate
  and more robust to hyperparameters than direct-coefficient regression,
  and found finer CFD meshes (sharper shocks) made NN prediction harder.
- Zafar (2023 dissertation) and related PRF/DCE papers from our professor
  - CNN/RNN surrogates for laminar-turbulent transition prediction.
  Precedent for: injecting scalar flow conditions separately from
  convolved/field inputs, and error-based active data selection.

## 3-week plan (high level)
- Week 1: dummy-data pipeline end-to-end, XFOIL batch data, one validated
  ANSYS case.
- Week 2: Stage 1 training (XFOIL), baselines, active selection + ANSYS
  grind.
- Week 3: Stage 2/3 fine-tuning, the core N_ANSYS-sweep experiment,
  ablations, write-up.

## Current status
Phase 1 (data generation) done except the ANSYS tier; Stage 1 prototype
trained. As of 2026-10-11 (the overnight run; branch `overnight-2026-10-11`
was merged into `main` and pushed on 2026-10-13):
- DONE: airfoil geometry (500 airfoils from 15 real UIUC baselines), XFOIL
  install + validation, batch XFOIL (7,500 cases, 94.33% converged), XFOIL data
  QA (0 flags), the 64x64 grid (domain x -0.25..1.25, y -0.75..0.75, defined in
  `data_spec.md`; ANSYS export must match), train/val/test split incl. two
  held-out families, classical-ML baselines, Stage 1 model (FNO trunk + Head A)
  with the overfit-a-single-sample gate passed, and training runs over 3 seeds
  (mean test Cp RMSE 0.035; held-out families 0.146). Results in `overnight/`.
- NOT DONE: the ANSYS tier (Fluent meshing is handled in a separate session,
  which has NOT yet been told the grid domain), Head B, physics loss, the
  core N_ANSYS-sweep experiment, real active selection (only a draft exists).
- Next: tell the ANSYS session the grid domain; one validated ANSYS case on that
  grid; review `overnight/active_selection_draft.md`.
- Live one-page status is in `STATUS.md`; the overnight log is `overnight/LOG.md`.

## Model and effort
I can't be switched automatically, so before each major step recommend in ONE line the model + effort for it:
- Sonnet / medium: reruns, sweeps, report formatting, edits to existing scripts
- Opus / medium: planning a study, new mesh or solver setup
- Opus / high: validation logic, unexplained divergence, results that look wrong
- Sonnet / low: simple questions, status checks
I switch it in the model menu. Recommend only at step boundaries (every switch re-reads the whole chat).

## STATUS.md (read by HQ)
Keep `STATUS.md` in this folder. After each major step, OVERWRITE it (never append) with at most 15 lines:
date/time, current state, last result (key numbers), next step, blocked/waiting on Areeb, recommended model/effort.
Current state only, no history. Don't message HQ; HQ reads this file when Areeb asks.
