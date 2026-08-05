# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) or other AI agents when working with code in this repository.

## Important Rule: Source of Truth
**Always refer to `AGENTS.md`** as the ultimate source of truth for the project's current architecture, file structure, and workflow.

## Commands

**Training & Execution (Local & Kaggle)**
- All real training code is completely self-contained in the `kaggle/` folder.
- Run the 10-model baseline sweep: `python kaggle/run_all_models.py`
- Run the dual-stream proposed model: `python kaggle/run_proposed_model.py`
- *Note: `src/train.py`, `run_experiments.py`, and `kaggle/SETUP.md` no longer exist. Do not attempt to run them.*

## Architecture & Structure

This repository is an evaluation framework comparing 10 curated baseline deep learning architectures (CNNs, Vision Transformers) against a Proposed-Model dual-stream architecture for Facial Expression Recognition in individuals with Autism Spectrum Disorder (ASD). 

**Data & Imbalance Handling**
- The canonical dataset is `dataset_clean/` (1,808 deduplicated images).
- The dataset suffers from severe class imbalance. This is primarily handled via a single weighting strategy:
  - Baselines (`run_all_models.py`): Handled purely via `WeightedRandomSampler` (loss is unweighted).
  - Proposed Model (`run_proposed_model.py`): Handled via `FocalLoss` alpha parameters (sadness x2.0, fear x1.2).

**Code Structure**
- There is NO shared module (`src/` has been deleted). `kaggle/run_all_models.py` and `kaggle/run_proposed_model.py` are monoliths that inline their own datasets, losses, models, and plotting logic. If a cross-cutting change is required (e.g., changing the dataset path or a transform), it must be manually applied to BOTH scripts.

**Validation Strategy**
- The project strictly uses **Stratified 5-Fold Cross Validation**. There are no explicit train/valid/test directories used during runtime; the script dynamically merges all splits from the disk and partitions them on the fly. 

**Resumability**
- The training loops write out-of-fold predictions to `.npy` files and use a `cv_metrics.json` / `cv_done.json` marker to track progress. If a run crashes or times out on Kaggle, re-running the script will instantly skip completed folds and resume from where it left off.
