# Autism Facial Expression Recognition — Project Explanation

### 1. The Core Objective
You are building an **Autism Facial Expression Recognition** system. The goal is to accurately classify the facial expressions of children with Autism Spectrum Disorder (ASD) into **6 specific classes**: 
* Anger, Fear, Joy, Natural, Sadness, Surprise.

Children with ASD often express emotions differently or more subtly than neurotypical children, meaning standard emotion-recognition models (like those trained on FER2013) perform poorly on them. This research bridges that gap.

---

### 2. The Dataset (`dataset_clean/`)
You are using a highly refined dataset of **1,808 images**. 
* **Data Sourcing:** The data was merged from 4 different sources (including Kaggle and Mendeley) to ensure enough diversity, especially since classes like "Fear" are very rare.
* **Data Cleaning (Crucial Step):** You ran a rigorous cleaning process using **dHash (Difference Hashing)**. You found and removed 88 near-duplicate images and 86 conflicting labels (e.g., the exact same face labeled as "Sadness" in one dataset and "Fear" in another). This prevents "Data Leakage" and artificially inflated results, making your methodology highly honest and academically rigorous.
* **Class Imbalance:** The dataset is heavily imbalanced (e.g., 843 images of Joy vs. only 68 images of Fear). 

---

### 3. The Baseline Sweep (`kaggle/run_all_models.py`)
Instead of just guessing which architecture works best, you conducted a **Comparative Study** across state-of-the-art models. You trained both **CNNs** (which are great at picking up local textures like wrinkles or eye crinkles) and **Transformers** (which are great at understanding the global geometry of the face). 

The models you tested include:
* **CNNs:** VGG-16, ResNet-50, InceptionV3, DenseNet-121, EfficientNet-B0, MobileNetV2
* **Transformers/Hybrids:** ViT-Base, DeiT-Small, Swin-Base, Swin-Tiny

---

### 4. Training Methodology & Handling Imbalance
To ensure fair and accurate training, you implemented several best practices:
* **Stratified 5-Fold Cross-Validation (The Great Merger):** Even though your `dataset_clean/` directory contains physical `train`, `valid`, and `test` folders, the training script completely ignores these boundaries. First, it merges every single image into one giant pool of 1,808 images in RAM. Then, it mathematically chops this pool into 5 equal chunks (folds). For each fold, it uses 4 chunks (80%) to train, and 1 chunk (20%) to evaluate. 
  * **Why no separate Test set?** Because you only have 68 "Fear" images, an 80/20 test split would leave you with just ~13 test images for Fear (which isn't statistically valid). Instead, the 5 evaluation folds combined become your ultimate Test set. By the time all 5 folds are finished, every single image was evaluated as an unseen "Out-of-Fold" (OOF) prediction exactly once. These 1,808 OOF predictions are glued together to calculate your final defensible score without needing a separate Test folder.
* **Weighted Random Sampler:** To combat the class imbalance, the training loader samples the rare classes (like Fear and Surprise) much more frequently than common classes (like Joy). This forces the model to pay equal attention to all emotions.
* **RAW Images:** You tested using MTCNN (face cropping) and CLAHE (contrast enhancement), but found they actually *hurt* accuracy because ASD faces sometimes have unusual poses that confuse crop algorithms. Therefore, you train directly on the RAW images.

---

### 5. Evaluation & Results
You do not judge these models on basic "Accuracy". In an imbalanced dataset where 46% of the data is "Joy", a model could just guess "Joy" every time and get 46% accuracy without learning anything.

Instead, you use **Macro F1-Score**, which calculates the performance for each class individually and averages them, treating "Fear" as equally important as "Joy".

**The Key Finding so far:** 
**VGG-16** proved to be the strongest baseline model (achieving ~72% Accuracy and ~62% Macro F1). Because VGG-16 is heavily texture-focused, it is particularly good at picking up the subtle pixel-level muscle twitches present in ASD facial expressions compared to deeper models like ResNet-50 which can overfit on small datasets.
