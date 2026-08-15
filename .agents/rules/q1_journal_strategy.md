# Q1 Journal Strategy — Proposed-Model Paper

## Baseline Performance (from final run, v3-emabn)

| Model | Acc | Macro F1 |
|---|---|---|
| **vgg16** | 0.7135 ± 0.014 | **0.6114 ± 0.017** ← best baseline |
| swin_base | 0.7201 ± 0.013 | 0.6064 ± 0.028 |
| vit_base | 0.7196 ± 0.009 | 0.6037 ± 0.026 |
| deit_small | 0.7152 ± 0.012 | 0.6015 ± 0.026 |
| maxvit_tiny | 0.7096 ± 0.009 | 0.6006 ± 0.026 |
| swin_tiny | 0.7057 ± 0.012 | 0.5985 ± 0.021 |
| densenet121 | 0.7091 ± 0.011 | 0.5944 ± 0.020 |
| efficientnet_b0 | 0.7074 ± 0.015 | 0.5863 ± 0.033 |
| mobilenetv2_100 | 0.6980 ± 0.022 | 0.5845 ± 0.034 |
| resnet50 | 0.6620 ± 0.022 | 0.5609 ± 0.031 |
| efficientformer_l1 | 0.5936 ± 0.203 | 0.5012 ± 0.184 |
| convnext_tiny | 0.2742 ± 0.165 | 0.0761 ± 0.026 ← collapsed |

**The proposed model must beat VGG16's F1 0.6114 by a clear, statistically non-overlapping margin.**

---

## Decision Gate After Cell 3 Results

```
Proposed model Macro F1 → Action
──────────────────────────────────────────────────────
≥ 0.67   → Q1 ready as-is (strong +5%+ gain over 0.6114)
0.64–0.67 → Add cross-attention fusion or one architectural upgrade
0.61–0.64 → Needs rework; clinical framing alone won't save it
< 0.61   → Architecture issue; check training stability logs
```

---

## Strengths to Highlight in Paper

- **VGG16-BN (local texture) + DeiT-S (global geometry)**: clinically justified — ASD FER needs both micro-texture AND holistic face structure.
- **Dual SE recalibration**: learned suppression of stream-specific noise before fusion.
- **FocalLoss + sadness ×2.0, fear ×1.2**: clinically-driven minority class weighting.
- **EMA + 2-stage training**: stabilization for small (1,808 image) datasets.
- **5-view TTA + uncertainty rejection guardrail**: strongest Q1 differentiator — most papers don't have this. Frame as clinical deployment safety (flags low-confidence predictions for caregiver review).

---

## Known Risks / Reviewer Red Flags

| Risk | Mitigation |
|---|---|
| Small dataset (1,808 images, fear=68) | Report mean ± std across 5-fold CV; use StratifiedKFold |
| `fear` F1 near 0 (~14 samples/fold) | Emphasize FocalLoss alpha boost; show per-class recall |
| `convnext_tiny` collapse (F1=0.076) | Explain as known training instability; exclude from mean or note |
| Dual-stream fusion "not novel enough" | Add cross-attention bridge OR emphasize clinical framing |
| No comparison to ASD-specific SOTA | Must cite existing autism FER papers and compare |

---

## If Results Are Disappointing — Upgrade Options

### Option 1: Cross-Stream Attention (highest impact, genuine novelty)
Replace simple concatenation with a cross-attention bridge between VGG16 and DeiT-S features before SE recalibration. Each stream attends to the other's context. This is novel for autism FER literature.

### Option 2: Compare Against ASD-Specific SOTA
Reviewers will ask "how does this compare to existing autism FER methods?"
Must cite: Lian et al., Bisogni et al., AffectNet/RAF-DB transfer papers.

### Option 3: Strengthen Clinical Narrative
The **uncertainty rejection guardrail** is the paper's strongest Q1 differentiator:
> *"Unlike existing methods that produce overconfident predictions, our model flags low-confidence cases (< 30%) for caregiver review, achieving X% accuracy on the high-confidence subset — directly addressing clinical deployment safety for ASD."*

---

## Target Journals (in priority order)

| Journal | Why | Key Requirement |
|---|---|---|
| **Computers in Biology and Medicine** | Q1, CiteScore ~14, clinical focus | Clinical contribution + accuracy |
| **Expert Systems with Applications** | Q1, broad ML + application | Strong ablation study |
| **IEEE Transactions on Neural Networks & Learning Systems** | Q1, prestigious | F1 gap must be clear and significant |
| **Pattern Recognition** | Q1 | Avoid if dataset is small — penalized heavily |

---

## Notes on Current Pipeline

- **Pipeline version**: `v3-emabn` — ModelEMA decays BN running stats (not frozen at init).
- **Proposed model checkpoint**: saves EMA weights, not raw model weights.
- **Fold file**: `fold_id_by_path.json` from Cell 2 (baseline run) is already in saved results — Cell 3 can run standalone.
- **Kaggle time estimate**: ~6.5 hours expected on T4 (5 folds × ~80 min). Set `PATIENCE=12` to reduce to ~5 hours if timeout is a concern.
- **Prior session resume**: `restore_from_prior_session()` handles mid-run Kaggle timeouts automatically.
