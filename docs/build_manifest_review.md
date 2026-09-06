# Code Review: `build_manifest.py`

## Verdict: Very solid — 3 real bugs, a few hardening suggestions

Overall, this is well-designed for a Q1-grade pipeline: subject-independent grouping via FaceNet + agglomerative clustering, perceptual duplicate detection, provenance tracking, and an audit report that surfaces exactly the red flags reviewers ask about. The downstream contract with [asd_fer_baseline.py](file:///C:/Users/mrsoh/Documents/Autism-Facial-Expression-Recognition/src/asd_fer_baseline.py) is clean — the columns `path`, `label`, `source`, `group`, `dup_cluster`, `is_exact_dup` are all consumed correctly.

That said, I found **3 bugs** (one potentially paper-affecting), and a handful of robustness/correctness improvements worth making.

---

## 🐛 Bugs

### Bug 1 — `md5_of()` is defined but never called (dead code)
[`md5_of()`](file:///C:/Users/mrsoh/Documents/Autism-Facial-Expression-Recognition/src/build_manifest.py#L114-L119) computes an MD5 over the **raw file bytes** (including JPEG headers, EXIF, etc.), but it's never called. Instead, [`compute_hashes()`](file:///C:/Users/mrsoh/Documents/Autism-Facial-Expression-Recognition/src/build_manifest.py#L138) computes MD5 over **decoded pixel bytes** (`im.tobytes()`).

This is actually the *correct behavior* — pixel-level MD5 catches recompressed copies, which is what you want. But the dead `md5_of()` function is confusing and should be removed.

**Impact:** None (dead code). **Fix:** Delete lines 114-119.

---

### Bug 2 — `md5` column uses pixel hash but `flag_duplicates()` uses it for exact-dup detection — potential collision with different labels
[Line 161](file:///C:/Users/mrsoh/Documents/Autism-Facial-Expression-Recognition/src/build_manifest.py#L161): `df["is_exact_dup"] = df.duplicated(subset=["md5"], keep="first")`

This marks duplicates purely by pixel content. If the same image file appears under two *different* class folders (e.g., the same child's face labeled "joy" in one dataset and "happy" in another), the **first alphabetical path wins** and the other is flagged as an exact dup. This is probably correct behavior (the labels *are* for the same image), but:

- The "kept" row's label depends on filesystem sort order, which is non-deterministic across OS/locale.
- No warning is emitted when exact duplicates have **conflicting labels**.

**Impact:** Moderate — could silently keep the wrong label. **Fix:** Add a check after line 161:

```python
conflict = df.groupby("md5").filter(lambda g: g["label"].nunique() > 1)
if not conflict.empty:
    print(f"[warn] {conflict['md5'].nunique()} pixel-identical images have "
          f"conflicting labels — review manually", file=sys.stderr)
```

---

### Bug 3 (Paper-affecting) — `group_holdout()` in baseline is broken for multi-label groups
This isn't in `build_manifest.py` itself, but it's a direct consequence of its output. [Line 222 of asd_fer_baseline.py](file:///C:/Users/mrsoh/Documents/Autism-Facial-Expression-Recognition/src/asd_fer_baseline.py#L222) does:

```python
groups = df.groupby("group").agg(n=("path", "size"), lab=("label", "first"))
```

The audit shows **151 groups span multiple labels** (`groups_spanning_labels: 151`). Using `.first()` for the label means those groups are only counted toward one class during test-set carving. Some classes (especially `fear` with only 618 images) could end up severely underrepresented or overrepresented in the held-out set.

**Impact:** High — test set class balance is unreliable. **Fix:** This should be fixed in `asd_fer_baseline.py`, not here, but it's worth noting that `build_manifest.py` should more prominently surface this as a **warning** (not just a quiet audit counter), because 151/228 = **66%** of groups spanning labels is extremely high and affects every downstream split.

---

## ⚠️ Robustness Issues

### R1 — Near-duplicate brute force is O(n²) — scales badly
The [brute-force loop](file:///C:/Users/mrsoh/Documents/Autism-Facial-Expression-Recognition/src/build_manifest.py#L179-L185) is fine for n=10,628 (~56M pairs, ~2–5 seconds in Python). But if you ever add a fifth dataset and go to 20k+ images, this will take 10x longer. The comment says "~2 seconds for n=10k" which is optimistic — `bit_count()` in pure Python on uint64 is faster in 3.10+, but the loop itself is the bottleneck.

**Fix (optional):** Not urgent, but a NumPy vectorized approach or BK-tree would handle 100k+ images:

```python
# Vectorized chunked approach
CHUNK = 2048
for i in range(0, n, CHUNK):
    xor = bits[i:i+CHUNK, None] ^ bits[None, :]  # (chunk, n) uint64
    hamming = np.unpackbits(xor.view(np.uint8), axis=-1).sum(-1)
    # ... union pairs where hamming <= threshold
```

### R2 — No MTCNN face detection before FaceNet embedding
[`infer_identities()`](file:///C:/Users/mrsoh/Documents/Autism-Facial-Expression-Recognition/src/build_manifest.py#L240-L245) resizes the whole image to 160×160 and feeds it to InceptionResnetV1. FaceNet expects a **cropped, aligned face** — feeding uncropped images degrades embedding quality significantly.

The ASD datasets probably have pre-cropped faces, but if any have full-frame images with backgrounds, the identity clusters will be noisy. A reviewer might ask why you didn't use MTCNN.

**Fix:** Add MTCNN detection with fallback to whole-image:

```python
from facenet_pytorch import MTCNN
mtcnn = MTCNN(image_size=160, margin=14, device=device, post_process=False)
# In the loop:
face = mtcnn(im)
if face is None:
    face = torch.from_numpy(np.asarray(im.resize((160,160)))).permute(2,0,1).float()
    face = (face - 127.5) / 128.0
buf.append(face)
```

### R3 — `CANONICAL_CLASSES` is defined but never enforced
[`CANONICAL_CLASSES`](file:///C:/Users/mrsoh/Documents/Autism-Facial-Expression-Recognition/src/build_manifest.py#L65) is defined on line 65 but never referenced in any logic. The actual class validation happens implicitly through `LABEL_ALIASES`. If an alias maps to a typo (e.g., `"suprise"` instead of `"surprise"`), no error is raised.

**Fix:** Add a validation after line 77:

```python
assert set(LABEL_ALIASES.values()) == set(CANONICAL_CLASSES), \
    f"LABEL_ALIASES maps to {set(LABEL_ALIASES.values())} but CANONICAL_CLASSES is {set(CANONICAL_CLASSES)}"
```

### R4 — `compute_hashes()` MD5 is not the same as `md5_of()` — column name is misleading
The `md5` column actually contains `hashlib.md5(im.tobytes()).hexdigest()` — a hash of the decoded, EXIF-transposed, RGB pixel buffer. This is semantically a **pixel hash**, not a file hash. Calling it `md5` is fine internally, but if you ever write about it in the paper, clarify it's a pixel-level hash (which is the right choice — it catches recompressed copies).

### R5 — Missing `"disgust"` alias
Common FER datasets include "disgust" as a class. If a fifth dataset is ever added that has disgust images, they'll silently trigger the unmapped-folder error. Consider whether "disgust" should map to an existing class or be explicitly excluded.

---

## ✅ What's Done Well

| Aspect | Assessment |
|---|---|
| Subject-independent grouping | ✅ FaceNet + agglomerative clustering is the right approach |
| Duplicate detection (exact + near) | ✅ Pixel-level MD5 + pHash with union-find is clean |
| Provenance tracking (`source` column) | ✅ Enables LODO and source-probe experiments |
| Graceful FaceNet fallback | ✅ Falls back to dup_cluster with a clear warning |
| Audit report | ✅ Surfaces cross-source groups and cross-label groups |
| Label normalization | ✅ Strict — unmapped labels raise, not silently skipped |
| Downstream contract | ✅ All columns consumed correctly by `asd_fer_baseline.py` |

---

## Summary of Recommended Actions

| Priority | Issue | Effort |
|---|---|---|
| 🔴 High | Add warning for conflicting labels on pixel-identical images (Bug 2) | 5 min |
| 🔴 High | Surface `groups_spanning_labels` as a loud warning, not just audit JSON (Bug 3 context) | 5 min |
| 🟡 Medium | Add MTCNN face detection before FaceNet (R2) | 30 min |
| 🟡 Medium | Validate `LABEL_ALIASES` values against `CANONICAL_CLASSES` (R3) | 2 min |
| 🟢 Low | Remove dead `md5_of()` function (Bug 1) | 1 min |
| 🟢 Low | Vectorize near-dup loop for future scale (R1) | 20 min |
