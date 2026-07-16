# Kaggle Setup Instructions

## 1. Upload Dataset to Kaggle
1. Go to [kaggle.com/datasets](https://www.kaggle.com/datasets) → "New Dataset"
2. Upload your `dataset/` folder as a zip
3. Note the dataset URL slug (e.g., `username/autism-fer-dataset`)

## 2. Update the Notebook
Open `kaggle/run_all_models.py` and change this line:
```python
DATA_DIR = "/kaggle/input/<YOUR-DATASET-NAME>/dataset"
```
to your actual dataset path (e.g., `/kaggle/input/autism-fer-dataset/dataset`)

## 3. Create Kaggle Notebook
1. Go to [kaggle.com/code](https://www.kaggle.com/code) → "New Notebook"
2. Upload `run_all_models.py` as a notebook (Kaggle auto-converts `.py`)
3. Or paste the contents into a new notebook cell-by-cell

## 4. Enable GPU
1. In the notebook editor, click "Settings" (right sidebar)
2. Under "Accelerator", select **GPU T4 x2** (free tier: 30hrs/week)
3. Ensure Internet is ON (for downloading pretrained weights)

## 5. Attach Dataset
1. In the right sidebar, click "Add Data"
2. Search for your uploaded dataset and attach it

## 6. Run
- Click "Run All" — all 14 models train sequentially
- Each model saves to `/kaggle/working/results/<model_name>/`
- Download results from the "Output" tab when done

## GPU Time Estimate
| Model Type | ~Time/Model | Total (14 models) |
|-----------|-------------|-------------------|
| MobileNet | ~5-10 min | ~1-2 hrs |
| EfficientNet | ~10-15 min | ~2-3 hrs |
| ViT/Swin | ~15-25 min | ~3-5 hrs |
| **Total** | | **~6-10 hrs** |

Fits within Kaggle's 30hr/week GPU quota with margin.

## 7. Download Results
After training, download the entire `/kaggle/working/results/` folder from the Output tab. It contains:
- `comparison.json` — all metrics in one file
- `f1_comparison.png` — visual ranking chart
- `comparisons/summary_table.txt` — formatted table
- Per-model folders with confusion matrices, training curves, F1 bar charts

## Troubleshooting
- **OOM (Out of Memory)**: Reduce `BATCH_SIZE` to 16 or 8 in the config
- **Slow training**: Check GPU is enabled (Settings → Accelerator → GPU)
- **Dataset not found**: Verify the `DATA_DIR` path matches your uploaded dataset
