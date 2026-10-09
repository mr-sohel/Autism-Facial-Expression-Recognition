# Report for Supervisor Meeting

**Project:** Emotion recognition from facial images of children with autism
**Prepared:** 9 October 2026
**Purpose:** Show what we found in the data, and ask for decisions before we train any model.

## 1. Summary in five lines

1. Our four datasets are not four datasets. They are about **960 photographs**, copied and augmented into 10,628 files.
2. **One in four photographs has two different emotion labels.** After removing those, only 5 anger, 13 fear and 17 surprise photographs are left.
3. We **cannot count the children**. The automatic face grouping puts similar-looking children together, not the same child.
4. The pipeline is now fixed so that copies of a photograph can never be in both training and testing.
5. With this data alone, a paper that only says "our new model is the best" is **unlikely to pass a Q1 review**. We need a decision on labels, extra data, and how to frame the paper.

No model has been trained on the real data yet.

## 2. What we did since the last report

- Checked the earlier numbers (5,139 unique images, 228 children) and found they were wrong.
- Wrote a stronger duplicate check (`src/audit_unique_images.py`). It compares the facial keypoints of two images, so it still recognises a copy after mirroring, cropping, rotating or adding noise.
- Tested that check against filenames. Hasibur files are named like `0368_aug_7.jpg`, and the original `0368.jpg` is in Talaat. The check found the correct original for 97.3% of images using the picture alone.
- Fixed `src/build_manifest.py` and rebuilt `runs/manifest.csv` and `manifest_audit.json`.
- Changed the training script so it uses one clean copy of each photograph.

## 3. Findings

### Finding 1: 10,628 files are about 960 photographs

| | Count |
|---|---|
| Image files | 10,628 |
| "Unique" in our earlier report | 5,139 |
| Distinct photographs | about 960 |

The number stays between 951 and 976 when the check is made stricter or looser.

### Finding 2: Three of the four datasets add very little

| Dataset | Files | Photographs inside | Photographs found only here |
|---|---|---|---|
| FERAC | 770 | 704 | 283 |
| Talaat | 833 | 645 | 42 |
| Hasibur | 7,600 | 586 | 0 |
| Nora | 1,425 | 462 | 16 |

- Hasibur is Talaat augmented 10 times. It adds nothing new.
- Nora adds 16 photographs.
- FERAC adds 283, but almost all are joy (194) and natural (77). It has no sadness or surprise.

### Finding 3: The labels are not reliable

237 of the 960 photographs (25%) are stored under two or more emotions. In 108 cases the exact same file is in two emotion folders inside Talaat itself.

| Emotion | Photographs | Photographs with one agreed label |
|---|---|---|
| Joy | 553 | 530 |
| Natural | 137 | 106 |
| Sadness | 113 | 52 |
| Anger | 74 | 5 |
| Surprise | 42 | 17 |
| Fear | 41 | 13 |
| Total | 960 | 723 |

69 of the 74 anger photographs are also labelled as another emotion, mostly sadness.

### Finding 4: We do not know how many children there are

The face grouping gives 176 groups, but when we looked at them they are groups of similar-looking children. The largest group has 150 photographs of clearly different toddlers. So the "228 unique children" in our earlier report was not correct.

The groups are still safe to use for splitting, because putting similar children together cannot put the same child on both sides. But we should not write a number of children in the paper until the groups are checked by hand.

### Finding 5: The earlier pipeline had a leak, now fixed

In the old manifest, 123 photographs had copies in different groups, so the same photograph could be in training and testing. In the new manifest this number is 0.

### Finding 6: The data is small and unbalanced

Joy is 58% of the photographs. Fear and surprise have about 40 each, before removing label conflicts.

### Limits of these findings

- The 960 is an estimate. We checked it with filenames and by looking at sample pairs, not with a full hand-labelled set.
- The 237 label conflicts is an upper limit. In a few cases the check joined two almost identical frames of the same child.
- None of these datasets explain how autism was diagnosed, whether consent was given, or who chose the emotion labels.

## 4. Can we still reach our goal?

**Goal:** a Q1 journal paper with a novel model.

| Plan | Realistic? | Why |
|---|---|---|
| Novel model only, on this data as it is | Unlikely | About 40 photographs in the small classes and unreliable labels. Differences between models will not be statistically significant. |
| Audit of the public datasets + cleaned benchmark + our model | Possible | The audit is a real result: these datasets are copies with conflicting labels, and published high accuracies are probably inflated. This holds even if our model is only a little better. |
| The same, plus re-labelled data and a second dataset | Best chance | Fixes the two weakest points: labels and size. |

Our honest view: the project can still produce a publishable paper, but not the paper we first planned, and Q1 is not guaranteed.

## 5. Decisions we need from you

1. **Paper framing.** May we present the dataset audit as a main contribution, together with the model?
2. **Labels.** Should we re-label the photographs ourselves? Our proposal: two or three people label all 960 photographs independently, and we report how often they agree. Removing the conflicted photographs is not practical, because it leaves 5 anger photographs.
3. **Number of classes.** If re-labelling still leaves anger, fear and surprise very small, may we reduce to fewer classes (for example joy, natural, sadness, other)?
4. **More data.** We searched and found no ready-to-download, reliably labelled image dataset of autistic children's emotions. The options are:
   - **DE-ENIGMA** (128 autistic children, UK and Serbia, video with expert valence/arousal ratings). Academic licence by application; a faculty member must apply. This is the best candidate. We have not confirmed that applications are still open.
   - **Writing to authors** of Hugging Rain Man (66 autistic children; images are not public, only labels and features), EMBOA and CALMED.
   - **Child datasets without autism** (CAFE, NIMH-ChEFS, ChildEFES) for pretraining. These also need a faculty signature.
   - **Collecting our own images** through a local autism centre with ethics approval. This is the only way to get known diagnosis, consent and labels together.

   Which of these will you support, and will you sign the applications?
5. **Computing.** Our machine has no NVIDIA GPU. Is Kaggle acceptable for training, or is there a lab GPU?
6. **Ethics.** Is it acceptable to publish on web-collected images of children with no documented consent or diagnosis? What statement will the journal need?

## 6. What we plan to do next

| Order | Task | Depends on |
|---|---|---|
| 1 | Send dataset access requests | Decision 4 |
| 2 | Check the identity groups by hand and improve them | Nothing, can start now |
| 3 | Re-label the 960 photographs | Decision 2 |
| 4 | Finish pipeline fixes (connect the model registry, fix the locked test split) | Nothing, can start now |
| 5 | Train baseline models twice: with the wrong (leaky) split and the correct split, to show how much accuracy is inflated | Tasks 2 to 4 |
| 6 | Build and test the novel model | Task 5 |

## 7. Files to look at

- `manifest_audit.json`: all the numbers in this report.
- `Progress_Report_ September_7.md`: full updated progress report, with a table of corrections to the earlier version.
- `PROGRESS.md`: step-by-step tracker.
- `src/audit_unique_images.py`, `src/build_manifest.py`: the code.
