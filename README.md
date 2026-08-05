# Autism Facial Expression Recognition

A comprehensive deep learning framework designed to accurately classify six fundamental facial emotions (anger, fear, joy, natural, sadness, surprise) in children with Autism Spectrum Disorder (ASD). 

This research evaluates 10 baseline CNN and transformer architectures to establish a strict benchmark. Based on the findings from these baselines, this project aims to ultimately propose a novel hybrid architecture specifically designed to capture the globally distributed and asynchronous facial cues unique to autistic expression.

## 1. Abstract & Motivation
Emotion recognition deficits in ASD are correlated with measures of adaptive functioning, social disability, and long-term mental health outcomes. Early recognition tools can help bridge this communication gap. In many regions, such as Bangladesh, there is a severe shortage of trained clinicians. An accurate, smartphone-deployable emotion recognition system can become a low-cost, accessible companion to traditional therapy.

This framework was developed to address specific challenges in ASD facial emotion recognition:
- **Atypical facial dynamics:** Children with ASD exhibit reduced facial symmetry and asynchronous activation.
- **Data scarcity & imbalance:** Minority emotions (fear, surprise, anger, sadness) are systematically under-represented in ASD datasets.

## 2. Dataset

The **primary dataset** for this research is the **Nora Mahmoud Mendeley Dataset** (ASD/Non-ASD labeled faces). Because ASD datasets are inherently small, we supplemented the primary dataset with carefully curated images from FERAC and other public Kaggle repositories to ensure robustness.

To prevent data leakage and ensure real-world clinical reliability:
- **Test Set:** 220 pristine, unaugmented images (approx. 3%) were completely isolated as a held-out test set.
- **Training/Validation:** The remaining data was systematically augmented to construct a perfectly balanced, robust foundation of 6,000 training images and 1,500 validation images. 

*Note: For the strict baseline evaluation script (`kaggle/run_all_models.py`), a Stratified 5-Fold Cross-Validation approach on 1,808 raw images is used to ensure robust comparative metrics.*

## 3. Workflow & Execution

All training code is self-contained in Kaggle scripts under `kaggle/`, designed to run on Kaggle GPUs (T4/P100) or locally.

1. **`kaggle/run_all_models.py`**: Trains the 10 curated baselines using Stratified 5-fold CV to establish the benchmark.
2. **`kaggle/run_proposed_model.py`**: *(Pending Development)* The script designed to train the future proposed hybrid architecture.

### Running Locally
```bash
# Run the 10-model baseline sweep
python kaggle/run_all_models.py
```
