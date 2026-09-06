# Comparison: `manifest_audit.json` vs `Dataset_Audit_Report.md`

## Side-by-Side Fact Check

| Claim in Report | `manifest_audit.json` Value | Match? | Notes |
|---|---|---|---|
| 10,628 raw images | `n_images: 10628` | ✅ | |
| 4 source datasets (Nora 1425, FERAC 770, Talaat 833, Hasibur 7600) | `source_counts` matches exactly | ✅ | |
| Class distribution (Joy 4590, Sadness 2451, etc.) | `class_counts` matches exactly | ✅ | |
| 1,572 exact duplicates | `n_exact_duplicates: 1572` | ✅ | |
| 7,501 augmented/near-duplicates | `n_images_in_near_dup_clusters: 7501` | ✅ | |
| **"5,709 unique raw images"** | Not directly in JSON | ❌ **Wrong** | See below |
| **"5,139 unique raw images"** (also in report) | Matches `dup_cluster` count after dedup | ⚠️ **Contradicts 5,709** | Report says both numbers |
| 228 unique children | `n_groups: 228` | ✅ | FaceNet identity clustering |
| 189 children in multiple datasets | `groups_spanning_sources: 189` | ✅ | |
| FERAC missing Sadness & Surprise | `class_x_source.ferac.sadness: 0, surprise: 0` | ✅ | |

---

## 🔴 Issue 1 — Contradictory "unique image" counts in the Report

The report contains **two different numbers** for unique images:

> *Section 2:* "we caught 555 more duplicates after removing a flaw in the previous AI's code!"
> *Section 2:* "True Unique Images: We have exactly **5,139 unique raw images** to train on."

But [AGENT.md](file:///C:/Users/mrsoh/Documents/Autism-Facial-Expression-Recognition/AGENT.md) line 16 says:

> "Out of 10,628 raw images, there are only **5,709 unique raw images**"

These can't both be right. The manifest JSON doesn't have a direct "unique images" field, but:

- `n_images` (10,628) − `n_images_in_near_dup_clusters` (7,501) = **3,127** non-clustered images
- `n_near_dup_clusters_gt1` (2,012) clusters + 3,127 singletons = **5,139** unique representatives

So **5,139 is correct** (one representative per dup_cluster). The **5,709 in AGENT.md is wrong** — that number appears to have been an earlier calculation before a deduplication bug was fixed (the report itself mentions "555 more duplicates" caught after a fix: 5,709 − 555 ≈ 5,154, close to 5,139).

> [!WARNING]
> [AGENT.md](file:///C:/Users/mrsoh/Documents/Autism-Facial-Expression-Recognition/AGENT.md) line 16 should be updated from **5,709** → **5,139**.

---

## 🔴 Issue 2 — Report is missing the label conflict disclosure

Our fixed `build_manifest.py` now reveals that **147 pixel-identical image groups have conflicting labels** across source datasets. This is a significant finding that the report doesn't mention at all.

This matters because:
- The same child's face is labeled "joy" in one dataset and "natural" in another
- When `asd_fer_baseline.py` calls `.drop_duplicates("dup_cluster", keep="first")`, the surviving label depends on filesystem sort order
- A reviewer who inspects the data will find these and question your labelling protocol

The report should add a **Problem C** section disclosing this.

---

## 🟡 Issue 3 — Report claim about "healthy mix of all 6 classes" is misleading

The report Section 5 says:

> "Every single testing fold will automatically contain a healthy, randomized mix of all 6 classes from across all datasets."

But `manifest_audit.json` shows **151 of 228 identity groups (66%) span multiple labels**. `StratifiedGroupKFold` stratifies on a *single* label per group, but 66% of your groups have images from *multiple* emotion classes. The stratification is based on whichever label `.first()` returns, which means:

- Class balance in each fold is **approximate**, not guaranteed
- This is expected and acceptable (because one child showing multiple emotions is the *correct* real-world scenario), but the report should say "approximately balanced" rather than imply a guarantee

---

## ✅ What's consistent

| Aspect | Verdict |
|---|---|
| Raw counts (images, sources, classes) | ✅ Perfect match |
| Duplicate detection numbers | ✅ Perfect match |
| Identity group count (228 children) | ✅ Perfect match |
| Cross-dataset overlap (189 children) | ✅ Perfect match |
| FERAC missing classes | ✅ Correctly documented |
| LODO invalidity rationale | ✅ Sound reasoning |
| Subject-independent CV solution | ✅ Correct approach |

---

## Recommended Updates to `Dataset_Audit_Report.md`

| Priority | Fix |
|---|---|
| 🔴 | Change "5,709" → "5,139" in AGENT.md (and verify the report is consistent) |
| 🔴 | Add **Problem C: Label Conflicts** — 147 image groups have the same pixels but different emotion labels across sources |
| 🟡 | Soften "healthy, randomized mix" to "approximately balanced" and note that 66% of children have images spanning multiple emotion classes |
| 🟢 | Add `n_label_conflicts_exact_dup: 147` to the production `manifest_audit.json` (re-run with FaceNet to get the updated JSON) |
