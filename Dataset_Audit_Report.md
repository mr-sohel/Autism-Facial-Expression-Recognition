# Progress Summary Report

## 1. What We Did
We gathered four publicly available datasets from Kaggle and Mendeley to create one large dataset for Autism Facial Expression Recognition:
- **[Nora Mahmoud Dataset](https://data.mendeley.com/datasets/b33pf78h62/1)**
- **[FERAC Dataset](https://www.kaggle.com/datasets/rajasreechaiti/ferac-dataset)**
- **[Dr. Fatma M. Talaat Dataset](https://www.kaggle.com/datasets/fatmamtalaat/autistic-children-emotions-dr-fatma-m-talaat?select=Autistic+Children+Emotions+-+Dr.+Fatma+M.+Talaat)** *(Version 2)*
- **[Md. Hasibur Rahman Dataset](https://www.kaggle.com/datasets/hasibur013/autism-facial-emotion-recognition)**

Together, these gave us a total of **10,628 raw + pre-augmented images**. 

| Dataset | Raw Image Count | Number of Emotion Classes |
|---------|-----------------|---------------------------|
| Nora Mahmoud Dataset | 1,425 | 6 |
| FERAC Dataset | 770 | 4 |
| Dr. Fatma M. Talaat Dataset | 833 | 6 |
| Md. Hasibur Rahman Dataset | 7,600 | 6 |
| **Total Aggregation** | **10,628** | **-** |

### Total Class Distribution (Raw)
| Emotion Class | Image Count |
|---------------|-------------|
| Joy | 4,590 |
| Sadness | 2,451 |
| Anger | 1,074 |
| Natural | 959 |
| Surprise | 936 |
| Fear | 618 |

Because researchers often copy datasets from one another on Kaggle, we ran a special script (`build_manifest.py`) to automatically find duplicates, group augmented images (like rotated or flipped faces), and track the unique identity of every child using FaceNet AI.

## 2. What We Found (`manifest_audit.json`)
The script generated an audit report that revealed the true structure of our data:
- **Exact Duplicates:** 1,572 images were exact pixel-identical copies.
- **Augmented/Near-Duplicates:** 7,501 images were just rotated, flipped, or slightly changed versions of other images, grouped into 2,012 near-duplicate clusters.
- **True Unique Images:** After collapsing each near-duplicate cluster to one representative, we have exactly **5,139 unique raw images** to train on. 
- **Total Unique Children:** Across all 10,628 images, there are actually only **228 unique children**. 
- **Label Conflicts:** 147 groups of pixel-identical images carry *different* emotion labels across source datasets (e.g., the same face labeled "joy" in one dataset and "natural" in another). Additionally, 151 of 228 identity groups (66%) contain images spanning multiple emotion classes.

## 3. The Problems With Our Dataset
The audit revealed three major issues with our merged Kaggle datasets:

### Problem A: Massive Dataset Overlap
Out of the 228 distinct children, **189 of them appear in more than one dataset**. This proves that these 4 datasets are not independent. People on Kaggle just downloaded the same original images, reshuffled them, added augmentations, and uploaded them as "new" datasets.

### Problem B: Missing Classes in FERAC
Most of our datasets have 6 emotion classes (Anger, Fear, Joy, Natural, Sadness, Surprise). However, **the FERAC dataset only has 4 classes**. It completely misses *Sadness* and *Surprise*. 
If we treat FERAC as a standalone dataset for testing, our model won't be able to properly evaluate its ability to recognize Sadness or Surprise.

### Problem C: Cross-Dataset Label Conflicts
**147 groups of pixel-identical images have conflicting emotion labels** across different source datasets. For example, the exact same child's face may be labeled "joy" in one dataset and "natural" in another. When we deduplicate by keeping one representative per cluster, the surviving label depends on filesystem sort order — meaning the "ground truth" for these images is somewhat arbitrary. Additionally, **151 of 228 identity groups (66%)** contain images spanning multiple emotion classes, which limits the precision of stratified class balancing during cross-validation.

## 4. Why "Leave-One-Dataset-Out" (LODO) Won't Work
A popular idea is to train the model on 3 datasets and test it on the 4th dataset (LODO). While this is usually a good idea, it is dangerous for our specific data.

1. **Data Leakage:** Because 189 children overlap across the datasets, if we train on 3 datasets and test on 1, the model will be tested on the exact same children it trained on. This will make our accuracy look artificially high, and Q1 journal reviewers will reject the paper because the results are not honest.
2. **Missing Evaluations:** If FERAC happens to be the dataset chosen for testing, the model will not be evaluated on Sadness or Surprise at all.

## 5. Our Solution: Subject-Independent K-Fold CV
To solve all of these problems, we are going to use **Subject-Independent K-Fold Cross-Validation**.

Instead of splitting the training and testing by *Dataset Name*, our script splits them by *Child Identity*. 
- It puts 80% of the children in the training set and 20% of the children in the testing set. 
- This mathematically guarantees that the model is tested on faces it has **never seen before** (solving Problem A).

**Why FERAC's Missing Classes Are No Longer a Problem:**
Because we pooled all the datasets together and split by child identity, FERAC is no longer isolated. 
- FERAC simply acts as a powerful "booster," providing 770 extra examples to help the model learn Anger, Fear, Joy, and Natural.
- The model naturally learns the missing Sadness and Surprise classes from the Nora, Talaat, and Hasibur datasets.
- Each testing fold will contain an approximately balanced mix of all 6 classes from across all datasets. (Note: because 66% of identity groups span multiple emotion classes, `StratifiedGroupKFold` class balancing is approximate rather than exact — this is expected and acceptable for this type of data.)

This fixes the dataset overlap problem, safely absorbs FERAC without issue, and provides the kind of rigorous, subject-independent evaluation that Q1 journals demand.
