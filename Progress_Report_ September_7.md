# Progress Summary Report

*First written 7 September 2026. Updated 9 October 2026 after a second, stricter audit of the data. Sections 2 to 5 replace the earlier version; the corrections are listed in Section 6.*

## 1. What We Did
We gathered four publicly available datasets from Kaggle and Mendeley to create one large dataset for Autism Facial Expression Recognition:
- **[Nora Mahmoud Dataset](https://data.mendeley.com/datasets/b33pf78h62/1)**
- **[FERAC Dataset](https://www.kaggle.com/datasets/rajasreechaiti/ferac-dataset)**
- **[Dr. Fatma M. Talaat Dataset](https://www.kaggle.com/datasets/fatmamtalaat/autistic-children-emotions-dr-fatma-m-talaat?select=Autistic+Children+Emotions+-+Dr.+Fatma+M.+Talaat)** *(Version 2)*
- **[Md. Hasibur Rahman Dataset](https://www.kaggle.com/datasets/hasibur013/autism-facial-emotion-recognition)**

Together, these gave us a total of **10,628 image files**.

| Dataset | Image Files | Number of Emotion Classes |
|---------|-------------|---------------------------|
| Nora Mahmoud Dataset | 1,425 | 6 |
| FERAC Dataset | 770 | 4 |
| Dr. Fatma M. Talaat Dataset | 833 | 6 |
| Md. Hasibur Rahman Dataset | 7,600 | 6 |
| **Total** | **10,628** | **-** |

We first ran `build_manifest.py`, which looks for duplicates with a perceptual hash and groups faces by identity with FaceNet. That gave the numbers in our earlier report (5,139 unique images, 228 children).

We then checked those numbers and found the duplicate detection was too weak for this data. The datasets were built with mirroring, cropping, rotation, colour changes and heavy noise, and a perceptual hash cannot see through those. We wrote a stronger check (`audit_unique_images.py`) that compares the actual facial keypoints of two images, added it to `build_manifest.py`, and ran everything again.

## 2. What We Found (`manifest_audit.json`)

**The whole collection is about 960 distinct photographs, not 5,139.**

| | Count |
|---|---|
| Image files | 10,628 |
| Pixel-unique files | 9,056 |
| "Unique" by the old perceptual hash | 5,139 |
| **Distinct photographs** | **960** |
| Average copies of each photograph | 11 |

How we know the new check is right: the Hasibur files are named like `0368_aug_7.jpg`, and every one of its 760 families has a file with the same number in the same emotion folder of Talaat. So the filenames tell us which images are copies, without anyone labelling anything. Using only the image content, our check linked 97.3% of Hasibur images to the correct Talaat original. The old hash split each family of 10 copies into about 6 "different" images.

The figure of 960 is an estimate. It stays between 951 and 976 when we make the matching stricter or looser.

### Where the photographs come from

| Dataset | Distinct photographs it contains | Photographs found only in this dataset |
|---------|----------------------------------|----------------------------------------|
| FERAC | 704 | 283 |
| Talaat | 645 | 42 |
| Hasibur | 586 | 0 |
| Nora | 462 | 16 |

- **Hasibur is Talaat augmented 10 times.** It adds no new photographs.
- **Nora adds 16 new photographs.** Its own page says it is an augmented rework of an existing dataset.
- **FERAC is the only dataset with a real set of its own** (283 photographs), but only in its four classes: joy 194, natural 77, fear 8, anger 4.
- 619 of the 960 photographs appear in two or more datasets.

### Class distribution

| Emotion | Image files | Distinct photographs | Photographs with one agreed label |
|---------|-------------|----------------------|-----------------------------------|
| Joy | 4,590 | 553 | 530 |
| Natural | 959 | 137 | 106 |
| Sadness | 2,451 | 113 | 52 |
| Anger | 1,074 | 74 | 5 |
| Surprise | 936 | 42 | 17 |
| Fear | 618 | 41 | 13 |
| **Total** | **10,628** | **960** | **723** |

The middle column uses the label of one chosen copy of each photograph. The last column counts only photographs whose copies all carry the same label.

## 3. The Problems With Our Dataset

### Problem A: The four datasets are mostly one dataset
Two thirds of the photographs are shared between datasets, and two of the four datasets add almost nothing new. The 10,628 files are about 960 photographs copied and augmented.

### Problem B: A quarter of the photographs have conflicting labels
**237 of the 960 photographs (25%) are filed under two or more emotions.** This is not only a disagreement between datasets: 108 of these cases are inside Talaat itself, where the identical file sits in two emotion folders. The most common pairs are anger/sadness, natural/sadness, joy/natural, joy/surprise and fear/sadness.

The effect on the small classes is severe. Of the 74 anger photographs, 69 are also labelled as something else, which leaves **5 anger photographs with an undisputed label**. Fear has 13 and surprise 17.

The 237 is an upper limit, because in a few cases our check merged two nearly identical frames of the same child.

### Problem C: The old manifest leaked between training and testing
In the earlier manifest, 123 photographs had copies assigned to different "children". A split by child could therefore put one copy of a photograph in training and another in testing. This is fixed in the new manifest: all copies of a photograph now share one group.

### Problem D: We do not know how many children there are
The identity grouping now gives 176 groups, but when we looked at them they are groups of **similar-looking children**, not individual children. The largest group contains 150 photographs of clearly different fair-haired toddlers. Face recognition models are trained on adults and do this poorly on young children.

Grouping look-alikes together is the safe direction (it cannot put the same child on both sides of a split), so the groups are usable for splitting. But the earlier statement that there are "228 unique children" was not correct, and we should not quote a number of children at all until the groups are checked by hand.

### Problem E: Missing classes in FERAC
FERAC has no Sadness or Surprise images. This matters more now, because FERAC is the only dataset that adds a meaningful number of new photographs.

### Problem F: The classes are very unbalanced and very small
Joy is 58% of the photographs. Fear and Surprise have about 40 photographs each before label conflicts are removed.

## 4. Why "Leave-One-Dataset-Out" (LODO) Won't Work
Training on three datasets and testing on the fourth is not a valid test here:

1. **The same photographs are on both sides.** 619 photographs appear in more than one dataset, and Hasibur is entirely contained in Talaat.
2. **Missing classes.** If FERAC is the test set, Sadness and Surprise are not tested at all.

## 5. Our Solution, and What It Does Not Solve

### What is now in place
- **One copy per photograph.** `build_manifest.py` gives every file a `photo_id` and marks one clean copy of each photograph (taken from Talaat or FERAC where possible, never a noisy augmented copy). The training script now uses only those 960 images. All augmentation is done on the fly during training.
- **Split by group, not by image.** We still use Subject-Independent Stratified Group K-Fold. All copies of a photograph are in one group, and the groups err on the side of merging similar children, so the same child should not appear in both training and testing.
- A check run on the new manifest gives 789 photographs for development and 171 in the locked test set (48 groups).

### What is still open
1. **Labels.** We must decide whether to drop the 237 conflicting photographs or have two or three people re-label them and report their agreement. Dropping them leaves almost no anger, fear or surprise data.
2. **Identity groups.** They need face alignment, a tuned threshold and a hand check of about 30 groups before we describe them in a paper.
3. **Dataset size.** With about 960 photographs, and about 40 in the smallest classes, differences between models will be hard to prove statistically. A second source of data would help a great deal. We have identified candidates to request (Hugging Rain Man, CAFE, NIMH-ChEFS, ChildEFES, LIRIS-CSE) but have not applied yet.
4. **Paper framing.** We suggest the paper present the audit itself as a contribution (public ASD emotion datasets are overlapping copies with inconsistent labels, and image-level splits inflate accuracy), alongside the model.

No model has been trained on the real data yet.

## 6. Corrections to the 7 September Version

| Earlier statement | Corrected |
|---|---|
| 5,139 unique raw images | About 960 distinct photographs. The perceptual hash missed augmented copies. |
| 7,501 images were rotated, flipped or changed versions, all caught | The hash caught only part of them; about 9,670 of the 10,628 files are copies. |
| 228 unique children | Unknown. The groups are look-alike clusters, not children. |
| 189 children appear in more than one dataset | 619 photographs appear in more than one dataset. |
| 147 label conflicts, between datasets | 237 photographs with conflicting labels; 108 of the pixel-identical cases are inside Talaat alone. |
| FERAC adds 770 extra examples | FERAC adds 283 photographs not found elsewhere. |
| Splitting by child guarantees unseen faces | True only after the fix in Problem C; the earlier manifest split 123 photographs across groups. |
