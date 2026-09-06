# Dataset Check: Simple Summary for Supervisor

## What we did
We combined 4 public datasets of autistic children's faces into one big dataset.

Total images: **10,628**

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

Goal: teach the computer to recognize 6 emotions — anger, fear, joy, natural, sadness, surprise.

We then ran a checking script to find copies and to find which photos belong to the same child.

## What we found — the real size is much smaller

Many photos are copies of each other.

| Finding | Number | Simple meaning |
|---|---|---|
| Exact copies | 1,572 | Same photo, saved twice |
| Small changes (rotated / flipped) | 7,501 photos in 2,012 groups | Same photo, slightly edited |
| Truly different photos | **5,139** | What we can actually train on |
| Different children | **228 only** | Not thousands, only 228 kids |

Big problem: joy has 4,590 photos, but fear has only 618. The computer will learn joy easily and fear poorly.

## Challenge 1: Same children in all datasets
Out of 228 children, **189 appear in more than one dataset**.

People on Kaggle copied the same kids and uploaded them as "new" datasets. So the 4 datasets are not independent.

## Challenge 2: Wrong and mixed labels
* **147 times:** the exact same photo has a different label. Example: one dataset says "joy", another says "natural".
* **151 of 228 children (66%)** have photos with mixed labels.
* FERAC has only 4 emotions. It has **zero** sadness and **zero** surprise photos.

This means our "correct answers" are noisy and need careful handling.

## Challenge 3: Normal testing will cheat
If we train on 3 datasets and test on the 4th (LODO), the test will show the same children the computer already saw. The score will look high but it will be fake. Q1 journals will reject this.

## Our solution
We will **split by child, not by photo**.

* 80% of children for training, 20% of children for testing.
* The computer is always tested on new faces it has never seen.
* We mix all datasets together, so FERAC's missing classes are covered by the other datasets.

This is honest, safe, and accepted by Q1 journals.

## What we need next
1. Accept that real data = 5,139 photos from 228 children.
2. Manually check ~30 child groups to prove our grouping is correct.
3. Fix or remove the 147 conflicting labels.
4. Continue with child-wise testing, not dataset-wise testing.
