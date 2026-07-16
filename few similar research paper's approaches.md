Yes, quite a few. Here's a rundown of published work on this exact problem, organized by what they used and how they trained:

**1. Radočaj & Martinović (2025), Applied Sciences** — CNN vs. Transformer comparison
They used the FERAC dataset (770 images, four classes: natural, fear, joy, anger) and removed black-and-white and duplicate images before training, then split into 691 training and 79 testing images, resizing all images to 224×224 pixels. They tested eight models total. Transformer-based models consistently beat CNNs, with the Swin Transformer reaching 80% accuracy and an F1-score of 0.79, while CNNs struggled to detect the "fear" class at all. Good paper to cite if you want to justify trying a transformer architecture instead of a plain CNN.

**2. Real-time FER + IoT system (Neural Computing and Applications, 2023)**
Used the same 1,200 train / 220 test image split as the FER-Autism Mendeley dataset. They proposed an "enhanced deep learning" (EDL) approach using a genetic algorithm to select optimal CNN hyperparameters, feeding data through MobileNet, Xception, and ResNet via data generators, and reported 99.99% accuracy. Be cautious citing this number as a target — accuracy that close to 100% on such a small dataset is a classic sign of data leakage (e.g., augmented near-duplicates ending up in both train and test splits), which is exactly the trap I flagged earlier.

**3. Autoencoder-preprocessing + Xception/InceptionV3 (arXiv, 2025)**
This one is directly useful methodologically. Their core problem: Xception/InceptionV3 need fixed 299×299×3 inputs, but ASD facial datasets have inconsistent image sizes and lighting. Their fix — an autoencoder that maps variable-sized inputs to standardized 299×299×3 outputs while preserving facial features, trained via a two-stage protocol: 100 epochs pretraining the autoencoder alone, then end-to-end fine-tuning of the full pipeline jointly. Results: Xception improved from 72.3% baseline accuracy to 85.6% with autoencoder preprocessing (a 13.3-point gain), and InceptionV3 improved from 71.0% to 83.8%, with large effect sizes (Cohen's d of 2.66 and 2.56) confirmed via paired t-tests and McNemar's test. This is a strong, honestly-reported result you could realistically replicate or reference — the preprocessing trick (not just the backbone) is the actual contribution.

**4. Hybrid ResNet50V2 + InceptionV3 (ScienceDirect, 2025)**
Combined a Hybrid ResNet50V2+InceptionV3 model for ASD prediction with an enhanced MobileNet for six-emotion recognition, training on labeled facial images categorized as Autism/Non-Autism drawn from Kaggle. Multi-task framing (ASD detection + emotion recognition together) — relevant if your thesis scope extends beyond pure emotion classification.

**5. Hybrid ensemble framework (IIETA, Oct 2025)**
Used ResNet50 and MobileNetV3Small as an ensemble learning classifier for autism detection, plus MobileNetV2 and DeepFace for emotion recognition as an interpretability layer alongside the ASD classification. Ensemble + explainability angle — useful if your examiner cares about interpretability, not just raw accuracy.

**6. CMU Multi-PIE / AffectNet comparison study**
Tested generalization across datasets: 90% accuracy on CMU Multi-PIE database images but only 68% accuracy on AffectNet database images using the same method — a useful data point showing how much accuracy swings based on dataset difficulty, which is worth citing when you explain why your own numbers might differ from other papers.

**Patterns across all these papers, for your training design:**
- Nobody trains from scratch — all use transfer learning (ImageNet-pretrained Xception, ResNet, MobileNet, InceptionV3)
- Preprocessing matters as much as architecture — the autoencoder paper got the single biggest, most credible jump (+13pts) purely from input standardization, not from swapping models
- Honest accuracy on these datasets clusters around 70–86%; anything near 99% on a dataset this small should be treated as a leakage red flag, not a target
- Several papers report precision/recall/F1 per class, not just overall accuracy — expected in this subfield because class imbalance (fewer "fear"/"anger" samples) is common

Given your interest, I'd suggest structuring your methodology section around: transfer learning backbone (pick 2-3 to compare, as most papers do) + an explicit preprocessing/standardization step (face alignment or autoencoder-style normalization) + stratified k-fold validation + full metric reporting (not just accuracy). That combination would put your thesis in line with the strongest recent work here (#1 and #3 above) rather than the suspiciously-high-accuracy ones.