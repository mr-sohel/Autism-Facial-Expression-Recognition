# CLAUDE.md

Undergraduate capstone research: facial emotion recognition (6 classes) for children with Autism Spectrum Disorder, aiming at a Q1 biomedical/health-informatics journal (JBHI, CBM, AIIM, BSPC). The planned contribution is a novel model (a CNN-Transformer hybrid / hemiface-asymmetry design) evaluated against strong baselines under a leakage-free protocol.

**Read `PROGRESS.md` first**: it tracks the step-by-step plan, current status and findings across sessions, and must be updated at the end of each working session.

`AGENT.md` holds the earlier agent context and `Progress_Report_ September_7.md` the summary given to the supervisor. This file supersedes `AGENT.md` where the two differ; the differences are listed under "Known gaps".

## Where the project stands

1. Started from Nora Mahmoud's Mendeley dataset, which is too small on its own.
2. Added three Kaggle datasets (FERAC, Talaat, Hasibur). All four are extracted under `data/processed/<source>/`.
3. The supervisor supplied the pipeline in `src/`.
4. `src/build_manifest.py` has been reviewed, patched and rerun: `runs/manifest.csv` (10,628 rows, 960 marked `is_photo_rep`) and `manifest_audit.json` are current.
5. The corpus is about 960 distinct photographs, not the 5,139 the pHash dedup reported.
6. Not done yet: no baseline has been trained on the real data. `runs/demo/` is synthetic output from `make_demo_run.py`, not results.

## Commands

Run everything from the repo root with the `src/` prefix. The shell is PowerShell on Windows.

```powershell
# Rebuild the manifest (needed after adding or changing any dataset)
python src/build_manifest.py --roots nora=data/processed/nora ferac=data/processed/ferac talaat=data/processed/talaat hasibur=data/processed/hasibur --source-priority talaat ferac nora hasibur --out runs/manifest.csv

# Standalone photograph audit with matcher validation (reuses the pair cache build_manifest wrote)
python src/audit_unique_images.py --manifest runs/manifest.csv --out-dir runs/unique_audit

# Benchmark: subject-independent CV + locked test, writes runs/baseline/preds/*.npz
python src/asd_fer_baseline.py --manifest runs/manifest.csv --out-dir runs/baseline --models resnet50 swin_tiny_patch4_window7_224 --seeds 0 1 2 --folds 5 --epochs 30

# Recompute every table and figure from saved predictions (seconds, no GPU)
python src/asd_fer_baseline.py --manifest runs/manifest.csv --out-dir runs/baseline --analyze-only

# Model registry
python src/asd_fer_zoo.py --list
python src/asd_fer_zoo.py --check <timm_name> --mode lora

# Pipeline smoke test on synthetic data
python src/make_demo_run.py
python src/asd_fer_baseline.py --manifest runs/demo/manifest_demo.csv --out-dir runs/demo --analyze-only
```

There is no test suite or linter.

## Code map

- `src/build_manifest.py`: scan class folders, pixel-MD5 exact duplicates, pHash clusters (`dup_cluster`, reference only), photograph IDs (`photo_id`, `is_photo_rep`, `photo_label_conflict`), FaceNet + agglomerative identity clusters per photograph (`group`), audit JSON.
- `src/audit_unique_images.py`: augmentation-robust copy detection (embedding shortlist, SIFT + RANSAC verification, filename families). Imported by `build_manifest.py`; run standalone it also scores the matcher against filename ground truth and writes `runs/unique_audit/unique_audit.json`.
- `src/asd_fer_baseline.py`: `load_manifest` (keeps `is_photo_rep` rows), `group_holdout` (locked 15% test by group, cached in `locked_test_groups.json`), `train_one`, `run_benchmark` (StratifiedGroupKFold x seeds), `analyze` (tables T2-T7, cluster bootstrap CIs, McNemar + Holm, calibration, risk-coverage), plus `source_probe`, `run_lodo`, `gradcam`.
- `src/asd_fer_zoo.py`: `MODEL_ZOO` tiers and `build_model(name, mode=full|lora|probe|linear)`.
- `src/asd_fer_figures.py`: all manuscript figures from `.npz` and history files.
- `docs/`: earlier review of `build_manifest.py`, audit cross-check, and the supervisor's planning PDFs in `docs/plan_resources/`.

## Rules that must hold

- Split by `group` (child identity) only. Never image-level or random splits.
- Leave-one-dataset-out is not a valid generalisation test here: 189 of 228 identity groups span more than one source.
- Do not train on pre-augmented copies. Use only `is_photo_rep` rows (one per `photo_id`); `dup_cluster` is too weak. Augment on the fly in PyTorch.
- All metrics and figures come from the saved `.npz` predictions via `--analyze-only`, never from inside the training loop.
- Keep the locked held-out test set and the cluster (group-level) bootstrap. The locked test is reported once and never used for tuning or model selection.
- Do not claim one model beats another without the Holm-corrected paired test supporting it.
- Figure rules in `asd_fer_figures.py` stay: no dual axes, CVD-safe palettes, small multiples per class, at most three overlaid models.
- New architectures go in `src/asd_fer_zoo.py`. Large backbones use frozen probe or LoRA, not full fine-tuning.
- Never commit images, `.npz`, or checkpoints. `data/processed/` is gitignored; the raw archives in `data/raw/` are in Git LFS.

## Dataset facts (from `manifest_audit.json`)

| | Image files | Distinct photographs | With one agreed label |
|---|---|---|---|
| Total | 10,628 | 960 | 723 |
| joy | 4,590 | 553 | 530 |
| natural | 959 | 137 | 106 |
| sadness | 2,451 | 113 | 52 |
| anger | 1,074 | 74 | 5 |
| surprise | 936 | 42 | 17 |
| fear | 618 | 41 | 13 |

| Source | Files | Photographs it contains | Found only here |
|---|---|---|---|
| ferac | 770 | 704 | 283 |
| talaat | 833 | 645 | 42 |
| hasibur | 7,600 | 586 | 0 |
| nora | 1,425 | 462 | 16 |

- Hasibur is Talaat augmented 10 times (`<stem>_aug_<k>.jpg` maps to `<stem>` in the same Talaat label folder).
- 619 photographs appear in two or more sources.
- FERAC has no `sadness` or `surprise` images; its 283 exclusive photographs are joy 194, natural 77, fear 8, anger 4.
- 237 photographs (25%) are filed under more than one emotion; 69 of the 74 anger photographs are among them.
- 176 identity groups, median 3 photographs, largest 150. They are look-alike clusters, not children.
- Folder label `Natural` maps to `natural`; each source has its own train/test folders, which the manifest ignores on purpose.

## Known gaps (check before relying on results)

These were found by reading the code and the manifest. None are fixed yet.

1. **Identity groups are look-alike clusters, not children.** FaceNet on whole images at threshold 0.55 puts 150 photographs of visibly different toddlers in one group. All copies of a photograph share a group, and over-merging is the safe direction for leakage, but fold sizes are distorted and no number of children may be quoted. Needs alignment, threshold tuning and a hand purity check (`PROGRESS.md` step 2).
2. **The photograph count is an estimate.** About 960 (951 to 976 across match thresholds), judged from contact sheets and filename ground truth, not a hand-labelled sample.
3. **`asd_fer_zoo.py` is not wired into the baseline.** `asd_fer_baseline.py` has its own `build_model` that calls timm with full fine-tuning; there is no `--mode` flag. LoRA and probe modes cannot be benchmarked until this is connected, and `param_groups` will need to handle frozen parameters.
4. **`group_holdout` uses each group's first label** to pick test groups, although 64% of groups are multi-label. Class balance of the locked test set is not controlled, and whichever class pool draws the 150-photograph group puts a sixth of the data in test.
5. **LODO and the source probe are still in the baseline** (`--lodo`, `--source-probe`), and its docstrings recommend LODO as primary generalisation evidence. That contradicts the dataset findings. If LODO numbers are ever reported, they must be labelled as leakage-affected.
6. **Manifest paths are relative to the repo root** (forward slashes). On Kaggle the data must sit at the same relative paths, or the manifest must be rebuilt there.
7. **No CUDA on this machine.** It has an Intel Arc A750 and an XPU build of torch (`torch.xpu.is_available()` is true), but `asd_fer_baseline.py` only checks `torch.cuda`, so training falls back to CPU here. Either add XPU device selection (as `build_manifest.py` now has) or run the benchmark on Kaggle.
8. **Label noise is unresolved.** 237 of about 960 photographs (25%) carry more than one emotion label across their copies; the representative copy's label is kept rather than adjudicated or dropped. Rows are flagged in `photo_label_conflict`.

## Working with the user

- Explain statistical and methodological choices plainly; the user is an undergraduate and reports to a supervisor who wrote the pipeline.
- When a number goes into the paper or a progress report, compute it from `runs/manifest.csv` or the `.npz` files rather than copying it from a markdown file. `docs/audit_comparison.md` records an earlier case where a stale count (5,709) was propagated.
- Flag anything a Q1 reviewer would challenge, and say so before running long training jobs that depend on it.
