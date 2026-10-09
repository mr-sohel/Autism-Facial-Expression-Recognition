# Progress Tracker

Living record of where the ASD facial-emotion project stands. Read this first in a new session; update it at the end of every working session.

**How to update:** change the status in the roadmap table, add findings under the step's section, and append one dated line to the session log. Numbers must come from a script output, not from memory or another markdown file; name the file they came from.

Last updated: 2026-10-09

## Roadmap

| # | Step | Status | Output |
|---|---|---|---|
| 1 | Find the true number of distinct photographs | **Done** (2026-10-09) | `src/audit_unique_images.py`, `runs/unique_audit/` |
| 1b | Use `photo_id` in the manifest and baseline dedup | **Done** (2026-10-09) | `runs/manifest.csv`, `manifest_audit.json` |
| 2 | Fix and validate the identity (child) groups | Started: clustered per photo; groups are look-alike clusters, not children | |
| 3 | Resolve label conflicts | Not started | |
| 4 | Fix the pipeline (zoo wiring, locked test split, portable paths, device) | Not started | |
| 5 | Run baselines on GPU: leaky split vs clean split | Not started | |
| 6 | Design and test the novel model | Not started | |
| — | Request access to other datasets (runs in parallel) | Not started | |

## Open decisions for the supervisor

1. **The corpus is about 960 photographs, not 5,139.** Does the paper stay a "novel model" paper, or become an audit-plus-method paper (recommended; see step 6)?
2. **A quarter of the photographs carry two or more emotion labels, and only 5 anger, 13 fear and 17 surprise photographs have an undisputed label.** Drop the conflicted ones, or re-annotate with two or three raters and report agreement? Dropping is not really viable for the small classes.
3. **Second dataset.** Is collecting data through a local centre possible (needs ethics approval), or do we rely on access requests?

## Step 1: distinct photographs (done)

**Question.** `build_manifest.py` reported 5,139 unique images after pHash dedup. Is that right?

**Answer.** No. The four sources contain about **960 distinct photographs** (951 to 976 depending on the match threshold). Source: `runs/unique_audit/unique_audit.json`.

**Why pHash was wrong.** The sources were built with mirror, crop, rotation, colour and heavy-noise augmentations, which pHash does not survive. Hasibur is Talaat augmented 10 times: every one of its 760 `<stem>_aug_<k>.jpg` families has a same-named file in the same Talaat label folder. pHash split each 10-image family into 6.4 "unique" clusters on average.

**Method** (`src/audit_unique_images.py`). ViT embedding to shortlist the 50 nearest images, then SIFT keypoints with a RANSAC similarity fit; a pair with 12 or more inliers is the same photograph. Filename families are then added as certain links. Connected components are photographs; each image gets a `photo_id` in `runs/unique_audit/photo_ids.csv`.

**How well the matcher works.** Scored against the Hasibur filenames before those links were added:
- 97.3% of Hasibur images were linked to their named Talaat parent by image content alone.
- 84% of 10-image families were recovered whole; the misses are the heaviest-noise copies, which the filename links then fix.
- By eye on sampled pairs: at 12 to 14 inliers nearly all pairs are true copies; at 10 to 11 about four in five; at 6 to 7 almost none. The wrong matches near the threshold are mostly two near-identical frames of the same child, so they do not cause identity leakage.

**Findings**

| | Value |
|---|---|
| Images in manifest | 10,628 |
| Pixel-unique images | 9,056 |
| pHash clusters (old "unique" count) | 5,139 |
| Distinct photographs | 960 |
| Mean copies per photograph | 11.1 |

| Source | Distinct photos it contains | Photos found only in this source |
|---|---|---|
| ferac | 704 | 283 |
| talaat | 645 | 42 |
| hasibur | 586 | 0 |
| nora | 462 | 16 |

- 619 of the 960 photographs appear in two or more sources; 299 appear in all four.
- Hasibur adds no new photographs. Nora adds 16. FERAC is the only source with a sizeable set of its own.
- Talaat itself holds duplicates: 833 files are 717 pixel-unique images and 645 photographs, some filed under two emotion folders.
- **237 photographs (25%) carry more than one emotion label** across their copies. Most common pairs: anger/sadness (63), natural/sadness (35), joy/natural (21), joy/surprise (17), fear/sadness (15). This is an upper bound: a few components merge adjacent frames of one child.
- **123 photographs had copies assigned to different identity groups** in the manifest as it stood before step 1b, so copies of one photo could land on both sides of a "subject-independent" split. Fixed in step 1b (now 0).
- Photographs per class: see the table under step 1b, which uses the chosen representative copy.

**Limits.** The 960 is an estimate. It slightly under-counts where near-identical frames of one child were merged, and would over-count if a noisy copy in Nora or FERAC failed to link. No hand-labelled sample has been scored; precision was judged from contact sheets only.

**Rerun.**
```powershell
python src/audit_unique_images.py --manifest runs/manifest.csv --out-dir runs/unique_audit
```
About 6 minutes on the Arc A750. Delete `runs/unique_audit/pair_inliers.csv` to force re-matching; otherwise it is reused.

## Step 1b: `photo_id` in the manifest and baseline (done)

`build_manifest.py` now calls the matcher and writes three new columns: `photo_id`, `is_photo_rep` (the one copy to train and evaluate on) and `photo_label_conflict`. Source: `manifest_audit.json`.

```powershell
python src/build_manifest.py --roots nora=data/processed/nora ferac=data/processed/ferac talaat=data/processed/talaat hasibur=data/processed/hasibur --source-priority talaat ferac nora hasibur --out runs/manifest.csv
```
About 8 minutes; pair matches are cached in `runs/unique_audit/` and reused while the image set is unchanged.

What changed in `build_manifest.py`:
- Photograph-level IDs from `audit_unique_images.py` (pHash `dup_cluster` is kept for reference only).
- Representative copy per photograph chosen by `--source-priority` (Talaat, then FERAC, then Nora; never Hasibur), then non-`_aug_` name, then path. Result: 645 from Talaat, 299 from FERAC, 16 from Nora.
- Identity clustering runs on one representative per photograph and every copy inherits the group. Before, all 10,628 files were embedded, including heavy-noise copies.
- Paths are written with forward slashes so the manifest loads on Linux/Kaggle.
- Device is auto-selected (CUDA, then Intel XPU, then CPU); EXIF orientation is applied before embedding as it already was before hashing.
- Audit JSON gained photograph-level fields. The old keys are unchanged.

What changed in `asd_fer_baseline.py`: `load_manifest` keeps `is_photo_rep` rows when that column exists (960 images) and falls back to the old pHash dedup otherwise.

| Class | Photographs (representative label) | With one agreed label |
|---|---|---|
| joy | 553 | 530 |
| natural | 137 | 106 |
| sadness | 113 | 52 |
| anger | 74 | 5 |
| surprise | 42 | 17 |
| fear | 41 | 13 |
| Total | 960 | 723 |

Checked on the new manifest: every `photo_id` maps to exactly one group; sampled paths exist; `load_manifest` returns 960 rows; `group_holdout` gives 789 dev and 171 locked-test photographs (48 groups).

## Step 2: identity groups (started)

Now: 176 groups from FaceNet (whole image, no alignment, cosine threshold 0.55, average linkage) on the 960 representatives. Median 3 photographs per group, 42 single-photo groups, largest 150.

**The groups are look-alike clusters, not children.** Contact sheets of the largest group show dozens of clearly different fair-haired toddlers; mid-sized groups collect, for example, children with glasses. Seven groups of 20 or more photographs hold 339 of the 960. This over-merging is the safe direction for leakage but unbalances folds, and no child count can be quoted. Whether the same child is ever split across two groups has not been checked.

To do:
- Face detection and alignment before embedding; try a lower threshold and report how group count and the largest group respond.
- Hand-check about 30 groups for purity, and a sample of nearest cross-group pairs for missed links.
- Consider treating near-identical frames (matcher inliers just under threshold) as same-child evidence.

## Step 3: labels

237 photographs with conflicting labels to drop or re-annotate; they are flagged in `photo_label_conflict` and currently keep the representative copy's label. 108 of the pixel-identical conflicts are inside Talaat alone (the same file in two emotion folders). By representative label the conflicted photographs are anger 69, sadness 61, natural 31, fear 28, surprise 25, joy 23. Waiting on supervisor decision 2.

## Step 4: pipeline fixes

- `asd_fer_zoo.py` is not called by `asd_fer_baseline.py` (no `--mode`; full fine-tuning only).
- `group_holdout` picks test groups by each group's first label.
- Done: manifest paths are now relative with forward slashes (run from the repo root).
- `asd_fer_baseline.py` only checks `torch.cuda`; this machine has Intel XPU. (`build_manifest.py` is fixed.)
- `--lodo` is still offered and recommended in docstrings although it leaks here.

## Step 5: baselines

Run the same models twice, with the leaky image-level split and the clean subject-level split, to measure how much published-style accuracy is leakage. Needs steps 1b to 4 first.

## Step 6: novel model

Recommended framing: (a) the audit and a cleaned benchmark, (b) expression-pretrained weights adapted with LoRA or a frozen probe, plus the hemiface-asymmetry module. With about 960 photographs, a claim that rests only on beating other architectures is unlikely to reach significance. Test the asymmetry idea first with the landmark baseline listed in the zoo.

## Other datasets

Searched 2026-10-09. **No ready-to-download, reliably labelled image dataset of autistic children's emotions was found.** Details below come from papers, repository pages and search results; nothing has been applied for, and access terms must be confirmed at the source.

Autistic children:

| Dataset | What it has | Access | Use for us |
|---|---|---|---|
| DE-ENIGMA | 128 children (UK, Serbia), aged 5 to 12, about 150 hours of video; valence/arousal annotated by experts for 49 children | Academic licence by application (db.de-enigma.eu). Source is a 2018 abstract; current status not confirmed | Best candidate for real ASD data. Supervisor would have to apply. Labels are valence/arousal, not six emotions |
| Hugging Rain Man | 66 ASD + 32 typically developing Chinese children, aged 2 to 12, 131,758 frames, expert action-unit labels | Labels and extracted features on GitHub (CC BY-NC-SA). **Images are not released** | Not usable for an image model unless the authors share images on request |
| EMBOA | Children with ASD in Poland, Turkey, North Macedonia; about 17 hours of video; 3 annotators, Ekman emotions | Not publicly available; ask the project (emboa.eu) | Only by direct request |
| CALMED | Children aged 8 to 12; 4 parent-labelled classes | Extracted features only, on request | No images |
| MMASD, Engagnition | Skeleton, optical flow, physiological signals | Open | No faces |
| NAO robot set (India, 2025) | 15 children, about 50,000 frames | Not released; labels come from models, not people | Not usable |

Typically developing children (reliable labels and consent; for pretraining or comparison, not ASD):

| Dataset | What it has | Access |
|---|---|---|
| CAFE | About 1,200 photographs, 100+ children aged 2 to 8, 7 expressions | Needs authorisation; believed to be via Databrary, where a professor must be the authorised investigator and students join as affiliates (to confirm) |
| NIMH-ChEFS | 482 images, ages 10 to 17, 4 emotions + neutral | Described as freely released; procedure not found |
| ChildEFES | Photos and videos, about 124 children aged 4 to 6 | By application to the authors |
| EmoReact | 1,102 videos, ages 4 to 14 | Not found; contact authors |
| LIRIS-CSE | 12 children, spontaneous video | Project site currently shows an empty listing |

Adult sets for expression pretraining: FER+/FER2013 (open), RAF-DB and AffectNet (request forms, usually signed by faculty).

Realistic routes to reliable data, in order of effort:
1. Re-label our 960 photographs with two or three raters (fixes labels; does not fix unknown diagnosis or consent).
2. Supervisor applies for DE-ENIGMA; email the Hugging Rain Man, EMBOA and CALMED authors.
3. Request CAFE and NIMH-ChEFS for child-face pretraining; start with FER+ now.
4. Collect our own images through a local autism centre with ethics approval. This is the only route that gives known diagnosis, consent and labels together.

## Session log

- 2026-10-09 (end): Wrote `Supervisor_Meeting_Report.md` for the meeting in the week of 12 October. Six decisions are waiting on the supervisor (framing, labels, class count, more data, compute, ethics); record the answers here after the meeting.
- 2026-10-09 (later): Reviewed and patched `build_manifest.py` (photo_id, per-photo identity clustering, portable paths, device auto-select); baseline now trains on `is_photo_rep` rows. Rebuilt `runs/manifest.csv` and `manifest_audit.json`: 960 photographs, 176 look-alike groups, 0 photos split across groups. Rewrote `Progress_Report_ September_7.md`.
- 2026-10-09: Wrote `CLAUDE.md`. Built `src/audit_unique_images.py` and ran it: about 960 distinct photographs; Hasibur shown to be Talaat x10; 237 photos with conflicting labels; 123 photos split across identity groups. Created this file.
