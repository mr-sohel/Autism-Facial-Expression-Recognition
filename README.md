# Autism Facial Expression Recognition (ASD FER)

This repository contains the publication-grade model implementation for the Q1 journal submission on Autism Facial Expression Recognition (ASD FER). The codebase is designed to enforce rigorous evaluation standards, such as subject-independent cross-validation, locked test sets, cluster bootstrapping, and statistical significance testing.

## Overview

The primary goal of this project is to develop and evaluate deep learning architectures (CNNs, Transformers, and Hybrids) for the complex task of classifying facial emotions in children with Autism Spectrum Disorder (ASD). Since standard image-level splitting leads to significant data leakage in merged face corpora, this project explicitly tracks face identities and groups them accordingly to prevent overlapping subjects between train and test splits.

## Project Structure

```text
Autism-Facial-Expression-Recognition/
├── data/
│   ├── raw/                 # Raw dataset zip files and unprocessed inputs
│   └── processed/           # Extracted datasets ready for training
├── src/                     # Source code for the project
│   ├── build_manifest.py    # Deduplicates and creates subject-independent cross-validation splits
│   ├── asd_fer_baseline.py  # Main pipeline: models, significance testing, CI bounds, etc.
│   ├── asd_fer_figures.py   # Code to generate the required 18 paper figures
│   ├── asd_fer_zoo.py       # Registry for Q1 models (Hybrids, Frozen DINOv3, etc.)
│   └── make_demo_run.py     # Script to generate a synthetic test run
├── notebooks/               # Jupyter notebooks for exploratory data analysis
├── runs/                    # Execution logs, output models, tables, and generated figures
├── results/                 # Consolidates final result tables and figures for the paper
├── docs/                    # Documentation and planning materials
│   └── plan_resources/      # Consultation transcripts and planning docs
├── requirements.txt         # Project dependencies
└── README.md                # Project overview and setup instructions
```

## Setup & Installation

It is recommended to use a virtual environment.

1. **Clone and enter the directory**:
   ```bash
   git clone <your-repo-url>
   cd Autism-Facial-Expression-Recognition
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   *(Note: For GPU support, ensure that you have installed the correct version of PyTorch corresponding to your CUDA setup prior to running the requirements file.)*

## Usage Guide

### 1. Preparing the Data (Manifest Building)
Run `build_manifest.py` on your merged datasets. This script handles exact/near-duplicate detection and infers identity groups using FaceNet.
```bash
python src/build_manifest.py --roots my_dataset=data/processed/dataset_clean --out runs/manifest.csv
```

### 2. Training and Evaluation (The Baseline Benchmark)
Run the full benchmarking suite across multiple model architectures. This computes subject-independent K-Fold CV, evaluates against a locked held-out test set, runs multiple seeds, and generates prediction `.npz` files.
```bash
python src/asd_fer_baseline.py --manifest runs/manifest.csv --out-dir runs/baseline \
    --models resnet50 swin_tiny_patch4_window7_224 \
    --seeds 0 1 2 --folds 5 --epochs 30
```

### 3. Metric and Figure Generation
Once the `.npz` predictions are saved, you can rerun the analysis at any time (which takes seconds) to recreate all figures and tables required by the journal:
```bash
python src/asd_fer_baseline.py --manifest runs/manifest.csv --out-dir runs/baseline --analyze-only
```
Check `runs/baseline/tables` and `runs/baseline/figures` for the outputs.

### 4. Synthetic Demo (Testing the Pipeline)
Generate a synthetic dataset to verify that the pipeline runs correctly without needing a GPU:
```bash
python src/make_demo_run.py
python src/asd_fer_baseline.py --manifest runs/demo/manifest_demo.csv --out-dir runs/demo --analyze-only
```

## Guidelines for Q1 Journal Submissions
- **Avoid leakage:** Always run `build_manifest.py` to prevent frames of the same child appearing in both train and test sets.
- **Statistical Significance:** The pipeline includes McNemar and Wilcoxon signed-rank testing. Pay attention to the adjusted p-values before claiming a model "outperforms" another.
- **Calibration & Confidence:** The pipeline records calibration curves and Expected Calibration Error (ECE), and evaluates the model's Selective Prediction (risk-coverage curve). These are critical metrics for clinical tools.
