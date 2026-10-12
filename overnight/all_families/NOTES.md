# All-families retrain: decisions made alone, and questions for the morning

Written 13 Oct 2026 during the night brief, while Areeb was asleep. Nothing here needed his answer to proceed; everything below is what I decided and what he may want to change.

## What I decided

1. **Reading of "retrain on ALL families".** A new split, `data_generation/splits_all_families.csv`: every one of the 15 families is split 80/10/10 by airfoil (410 train, 45 val, 45 test airfoils; seed 42). The two families that were held out before (`e387`, `naca6409`) are now trained on. The old `splits.csv` is untouched (verified: the default mode reproduces it byte for byte), because the earlier results depend on it.
2. **Three seeds (42, 43, 44)**, same protocol as before, so there is a spread and an ensemble to use.
3. **Active selection ranks only val+test airfoils**, using the ensemble mean. Train airfoils look better than they are, so ranking on them would hide exactly the cases ANSYS should look at. That leaves a candidate pool of about 90 airfoils, from which 15 to 20 are picked, at most 3 per family.
4. **Resource limits as the brief said:** 2 CPU threads; pause if free RAM is below 3 GB.
5. **Output location:** everything under `overnight/all_families/`, files tagged `allfam_s<seed>`, so no earlier result is overwritten. Branch: `stage1-all-families` (local, not pushed; only the overnight merge was approved for pushing).

## What I found

- **The RAM rule blocks the retrain while Fluent runs.** Free RAM is about 3.9 GB with Fluent and the Claude app open, and a CUDA training process needs roughly 2 GB. Starting it would push free RAM below the 3 GB floor, so the launcher waits for 5.2 GB free (3 GB floor + 2.2 GB for the job) and the in-training guard pauses below 3 GB. If Fluent runs all night, the retrain may not start until a Fluent job ends. Check `launcher.log` to see whether and when it did.
- **Update 01:39: the retrain did start.** Free RAM rose to 5.6 GB at 01:37 and the launcher began, but its first launch failed on every attempt within a second: PowerShell's `>` and `Start-Process -Redirect*` treat `[` `]` as wildcards, and this repo sits under `...\[2] Porfolio\[4] Personal Projects\...`. Logs now go to `%TEMP%\neurofluid_allfam` and are copied here afterwards; relaunched 01:38 and training is running. The measured footprint of a training process is about 1.4 GB (free RAM went 5.6 to 4.2 GB), so the 5.2 GB start threshold was conservative; about 4.5 GB would have kept the same 3 GB floor.
- **Update 01:43: a paused trainer was holding RAM, so I made it release it.** At 01:40 a Fluent run (Bullet's fine mesh) took free RAM to 2.7 GB, below the 3 GB floor; my trainer paused as instructed but still held its 1.3 GB. I stopped it by hand (about a minute of training lost), then added save-and-exit: every epoch saves a resumable state, and if a low-RAM pause lasts 2 minutes the trainer saves and exits with code 75; `run_all_families.ps1` then waits for RAM and resumes from the saved epoch (a pause is not counted as a failure). Start threshold is now 5.0 GB. Because resume is new, `model/test_resume.py` checks that an interrupted-and-resumed run matches an uninterrupted one, and `run_test_then_train.ps1` only starts the real training if that test passes (result in `test_resume.out`).
- **`git push origin main` fails non-interactively** because two GitHub accounts are saved (`areeeb5`, `areeeb55`) and git cannot choose without a prompt. Pushing to `https://areeeb5@github.com/areeeb5/neurofluid-aero.git` works. I used that once and did not change the stored remote.

## Questions for the morning

1. **Is the all-families split the retrain you meant?** The alternative is 5-fold cross-fitting by airfoil: five models, each scoring the airfoils it never saw, so all 500 airfoils get an honest "most wrong" score (candidate pool 500 instead of about 90), plus one final model on everything. It costs about five times the GPU time (roughly 2 hours when RAM allows).
2. **RAM floor:** keep the strict 3 GB rule, or let the retrain run after 08:00 when Fluent is idle? (The launcher can simply be re-run: `overnight\all_families\run_all_families.ps1`.)
3. **Plain `git push`:** set the stored remote to `https://areeeb5@github.com/areeeb5/neurofluid-aero.git` so it works without the URL trick? That is a small config change I did not make.
4. **The finer-grid (128x128) experiment is still paused** on branch `finer-grid-128` (plan in `experiments/finer_grid_128/PLAN.md`). Resume it, and in which order relative to this?
5. **The ANSYS Meshing session still has not been told the grid domain** (x -0.25..1.25, y -0.75..0.75, 64x64 cell centres, layout `[iy, ix]`).
6. I corrected one stale line in `CLAUDE.md` ("local, not pushed" after the push). Say if you would rather I left that file alone.
