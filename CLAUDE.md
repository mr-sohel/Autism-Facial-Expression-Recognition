# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

**Project Setup**
- Install dependencies: `pip install -r requirements.txt` (requires Python 3.10+, PyTorch 2.0+, and `timm`)

**Training & Execution**
- Train a single model locally: `python src/train.py --model <model_name> --loss ce_smooth --epochs 80 --batch-size 16 --mixup --ema`
- Run the full pipeline (all 20 models sequentially): `python run_experiments.py`
- Kaggle pipeline: The script `kaggle/run_all_models.py` is intended to be run in a Kaggle notebook environment with the dataset uploaded there (see `kaggle/SETUP.md` for details).

## Architecture & Structure

This repository is an evaluation framework comparing 20 deep learning architectures (CNNs, Vision Transformers, and hybrid models) for Facial Expression Recognition in individuals with Autism Spectrum Disorder (ASD). 

**Data & Imbalance Handling**
- The dataset (`dataset/`) suffers from severe class imbalance. This is primarily handled in `src/dataset.py` using a custom `WeightedRandomSampler` (loss is unweighted to avoid double-penalizing).

**Code Modularity (`src/`)**
- `src/models.py`: Centralized model definition. It contains `MODEL_CONFIGS` (a dictionary mapping model names to their architecture configs) and the factory function `get_model()`.
- `src/dataset.py`: Handles data loading and augmentations (random flip, rotation, affine, color jitter, blur, random erasing), as well as the calculation of class weights for the sampler.
- `src/train.py`: The main CLI entry point containing the PyTorch training loop for a single model.
- `src/utils.py` & `src/losses.py`: Implements advanced training regularization components like MixUp, Exponential Moving Average (EMA), FocalLoss, LabelSmoothingCrossEntropy, and training metrics calculation. 
- `src/evaluate.py`: Generates all outputs saved to the `results/` folder, including evaluation metrics JSONs, normalized confusion matrices, and comparison plots across models.

**Outputs**
- All training and evaluation artifacts are saved in the `results/` directory, structured as `results/<model_name>/` for individual model metrics, weights, and plots, and `results/comparisons/` for global summary charts across all tested architectures.
