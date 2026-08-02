# Thesis Plan — Autism Facial Emotion Recognition with XAI

## 1. Objective
Classify the **6 emotions (anger, fear, joy, natural, sadness, surprise)** of **ASD children** from facial images, with **explainable AI**, aiming for a **Q1 journal publication** that beats prior published results.

## 2. Dataset
### 2.1 Primary dataset (teacher's choice)
**FER-Autism** — Mendeley Data, DOI `10.17632/b33pf78h62.1`
- 1,200 train / 220 test images, 6 classes (same taxonomy as ours)
- Published Oct 2025; "enhanced/restructured" version of the Kaggle-Autism lineage
- **Known caveats to verify before use:** built via 10× augmentation per original; train/test may share the same children (leak risk)

### 2.2 Verification (before any training)
Run `verify_fer_autism.py` on the downloaded set. Must check:
1. Total + per-class counts
2. Exact duplicates (md5)
3. Near-duplicates (dHash hd≤4) — clusters
4. **Cross-split leak:** clusters spanning train ↔ test
5. Within-train duplication (augmentation signature)
6. Label conflicts inside clusters

**Decision rules:**
- No leak + no conflicts → use as-is (fixed 1200/220 split, plus our own 5-fold CV for robustness)
- Leak found → regroup by face identity / dedup, and document the fix as a methodological contribution

### 2.3 Backup / secondary
Our cleaned merged set (~1,870 unique after perceptual dedup + conflict resolution) can serve as a **secondary comparison set** to show robustness across corpora. Never merge the two.

## 3. Experimental Protocol
- **Stratified 5-fold CV (seed 42)** on the autism data; every image predicted exactly once (out-of-fold)
- Report **mean ± std** across folds, plus per-class F1 and 95% CI
- Fixed train/test split reported as a secondary evaluation when required by the dataset's official protocol

## 4. Models
### 4.1 Baselines (8 curated models, all ImageNet-pretrained)
- VGG16, ResNet50, EfficientNet, MobileNet, DenseNet, InceptionV3 (299 input), ViT-B, DeiT-S
- Differential LR (backbone lr×0.1, head lr)
- CNNs: focal loss, lr=1e-3; Transformers: label smoothing, lr=1e-4

### 4.2 Proposed: Care-FER (dual-stream)
- **Spatial stream:** VGG16-BN `forward_features` + GAP → 512-d
- **Token stream:** DeiT-S CLS token → 384-d
- Dual **SE blocks** (r=16); head 896→512→256→6
- Focal loss (α per-class: joy 0.39, sadness 1.45, fear 4.57) + sadness ×2.0 / fear ×1.2 boosts
- Two stages: 160 ep (unbalanced) → 20 ep (frozen backbone, balanced sampler)
- Warmup + cosine LambdaLR; EMA (deepcopy) weights saved for eval

### 4.3 Pretraining ablation (key comparison)
- **Group A (main):** ImageNet init → fine-tune on autism data
- **Group B (ablation):** FER2013/FER+ pretrain → fine-tune on the same folds
- Report which wins; keep as honest ablation either way

## 5. XAI (quantitative, not just heatmaps)
- Grad-CAM per class, aggregated over all OOF predictions (not cherry-picked)
- ROI coverage statistics (eyes / nose / mouth regions)
- **Faithfulness metrics:** Pointing-Game / AUROC of explanation
- Compare attention maps on ASD vs. neurotypical faces if TD data is available (clinical angle)

## 6. Required Ablations (reviewer-proofing)
- Care-FER vs. each stream alone (spatial-only, token-only)
- ImageNet vs. FER2013 pretrain
- Focal loss vs. weighted CE; balanced vs. unbalanced stage-2
- Dedup-cleaned vs. raw dataset impact

## 7. Paper Positioning / Claims
> *First hybrid CNN-Transformer with SE fusion + quantitative XAI for 6-class emotion recognition in autistic children, on a leakage-checked dataset, beating prior ASD-emotion results (~73–79%).*

- Benchmark targets to beat: CvT 79.12% (face-only), Enhanced MobileNet 73.3%
- Prior 90%+ "ASD detection" papers are binary + leaky — cite as motivation for honest protocol

## 8. Reproducibility
- `kaggle/run_all_models.py` — 8 baselines + CV + figures
- `kaggle/run_proposed_model.py` — Care-FER + 5-view TTA + uncertainty guardrail + Grad-CAM
- Resumable: per-fold `.npy` OOF + `cv_metrics.json`; Kaggle GPU (T4/P100), CUDA-only

## 9. Timeline / Next Actions
1. **Now:** download FER-Autism → run verification script → decide clean/leaky
2. Add FER2013 pretrain option to notebook (`PRETRAIN = "imagenet" | "fer2013"`)
3. Build dataset-cleaning pass for secondary set
4. Run baseline sweep → Care-FER → ablations
5. Run XAI faithfulness evaluation
6. Write thesis / paper
