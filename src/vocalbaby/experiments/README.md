# XGBoost Feature Comparison Experiments

Reproducible pipeline for comparing XGBoost model performance across four feature sets with systematic hyperparameter tuning.

## Overview

This module implements a complete end-to-end pipeline for:
- Extracting features from four different representations
- Tuning hyperparameters independently for each feature set using Optuna
- Training XGBoost classifiers with best parameters
- Evaluating on validation and test sets
- Generating confusion matrices and comprehensive metrics
- Aggregating results for cross-feature comparison

**Methodology**: Exact replication of the workflow from notebook `06__xgboost_egemaps_smote_optuna_experiment.ipynb`

## Feature Sets

1. **eGeMAPS** (88-dim): Extended Geneva Minimalistic Acoustic Parameter Set
2. **MFCC** (20-dim): Mel-Frequency Cepstral Coefficients with temporal pooling
3. **HuBERT SSL** (768-dim): Self-supervised embeddings from `arunps/hubert-home-hindibabynet-ssl`
4. **Wav2Vec2 SSL** (768-dim): Self-supervised embeddings from `arunps/wav2vec2-home-hindibabynet-ssl`

## Architecture

```
vocalbaby/experiments/
├── __init__.py
├── data_loader.py              # Load train/valid/test splits and labels
├── feature_extractors.py       # Feature extraction for all four sets
├── hyperparameter_tuning.py    # Optuna tuning (replicates notebook 06)
├── training.py                 # XGBoost training with best params
├── evaluation.py               # Evaluation + confusion matrix generation
├── run_comparison_all.py       # Main orchestration script
└── scripts/
    ├── generate_features.py    # Feature cache generation
    ├── tune_hyperparams.py     # Hyperparameter tuning
    ├── train_model.py          # Model training
    └── evaluate_model.py       # Model evaluation
```

## Output Structure

```
artifacts/
├── features/
│   ├── egemaps/
│   │   ├── train/features.npy
│   │   ├── valid/features.npy
│   │   └── test/features.npy
│   ├── mfcc/
│   ├── hubert_ssl/
│   └── wav2vec2_ssl/
├── models/
│   ├── egemaps/
│   │   ├── best_params.json
│   │   ├── xgb_model.pkl
│   │   ├── imputer.pkl
│   │   └── label_encoder.pkl
│   ├── mfcc/
│   ├── hubert_ssl/
│   └── wav2vec2_ssl/
├── eval/
│   ├── egemaps/
│   │   ├── confusion_matrix_valid.png
│   │   ├── confusion_matrix_valid.csv
│   │   ├── confusion_matrix_test.png
│   │   ├── confusion_matrix_test.csv
│   │   ├── metrics_valid.json
│   │   ├── metrics_test.json
│   │   ├── classification_report_valid.json
│   │   ├── classification_report_test.json
│   │   └── labels.json
│   ├── mfcc/
│   ├── hubert_ssl/
│   └── wav2vec2_ssl/
└── results/
    └── results_summary.csv      # Aggregated comparison table
```

## Usage

### Quick Start: Run Complete Pipeline

```bash
# Run everything for all four feature sets
python -m vocalbaby.experiments.run_comparison_all

# Run for specific feature set only
python -m vocalbaby.experiments.run_comparison_all --feature-set mfcc

# Skip steps that are already complete (incremental execution)
python -m vocalbaby.experiments.run_comparison_all --skip-features --skip-tuning
```

### Step-by-Step Execution

#### 1. Generate Feature Caches

```bash
# Generate all feature sets
python -m vocalbaby.experiments.scripts.generate_features --feature-set all

# Or generate individually
python -m vocalbaby.experiments.scripts.generate_features --feature-set egemaps
python -m vocalbaby.experiments.scripts.generate_features --feature-set mfcc
python -m vocalbaby.experiments.scripts.generate_features --feature-set hubert_ssl
python -m vocalbaby.experiments.scripts.generate_features --feature-set wav2vec2_ssl
```

#### 2. Tune Hyperparameters

```bash
# Tune all feature sets (40 trials each, matching notebook 06)
python -m vocalbaby.experiments.scripts.tune_hyperparams --feature-set all

# Or tune individually
python -m vocalbaby.experiments.scripts.tune_hyperparams --feature-set egemaps --n-trials 40

# Quick test with fewer trials
python -m vocalbaby.experiments.scripts.tune_hyperparams --feature-set mfcc --n-trials 10
```

#### 3. Train Models

```bash
# Train all models
python -m vocalbaby.experiments.scripts.train_model --feature-set all

# Or train individually
python -m vocalbaby.experiments.scripts.train_model --feature-set egemaps
```

#### 4. Evaluate and Generate Confusion Matrices

```bash
# Evaluate all models and aggregate results
python -m vocalbaby.experiments.scripts.evaluate_model --feature-set all

# Evaluate individual model
python -m vocalbaby.experiments.scripts.evaluate_model --feature-set egemaps

# Evaluate and force aggregation
python -m vocalbaby.experiments.scripts.evaluate_model --feature-set mfcc --aggregate
```

## Hyperparameter Tuning Workflow

Exact replication of notebook 06:

1. **Imputation**: Fit `SimpleImputer(strategy="median")` on train, transform all splits
2. **SMOTE**: Apply `SMOTE(random_state=42)` to training set only
3. **Optuna Search**:
   - 40 trials (default, configurable)
   - Multi-objective: maximize (UAR, F1)
   - Search space:
     ```python
     {
       "max_depth": [3, 12],
       "learning_rate": [0.01, 0.3] (log scale),
       "n_estimators": [100, 1500],
       "subsample": [0.5, 1.0],
       "colsample_bytree": [0.5, 1.0],
       "gamma": [0.0, 5.0],
       "min_child_weight": [1, 10],
       "reg_lambda": [0.0, 5.0],
       "reg_alpha": [0.0, 5.0]
     }
     ```
4. **Selection**: Best trial = `max(trial.values[0])` (UAR)

## Training Workflow

1. Load best hyperparameters from tuning
2. Apply same preprocessing: imputation + SMOTE
3. Train XGBoost with best params on SMOTE-resampled training set
4. Save model + imputer artifacts

## Evaluation Workflow

For **both validation and test** splits:
1. Load trained model + imputer
2. Preprocess features (imputation only, no SMOTE)
3. Predict and compute metrics: UAR, F1, Precision, Recall
4. Generate confusion matrix (PNG + CSV)
5. Save classification report (JSON)
6. Save metrics (JSON)

## Data Splits

- Uses existing child-disjoint splits from data ingestion
- Labels loaded from `artifacts/<timestamp>/data_transformation/features/`
- LabelEncoder fitted on train, applied consistently across all splits

## Requirements

### Core Dependencies
- `xgboost`
- `optuna`
- `scikit-learn`
- `imbalanced-learn` (SMOTE)
- `transformers` (HuggingFace)
- `torch`
- `librosa`
- `opensmile`
- `pandas`, `numpy`
- `matplotlib`, `seaborn`

### Installation

```bash
# Install all dependencies (already in requirements.txt)
pip install -r requirements.txt

# Or install specific packages
pip install optuna transformers torch librosa opensmile
```

## Reproducibility

- **Random seed**: 42 (all steps)
- **Deterministic splits**: Child-disjoint from data ingestion
- **Label encoding**: Fitted once on train, reused consistently
- **Imputation**: Fitted on train only, transforms valid/test
- **SMOTE**: Applied to train only during tuning and training

## Results Interpretation

### Results Summary Table

`artifacts/results/results_summary.csv` contains:

| feature_set | valid_uar | valid_f1 | valid_precision | valid_recall | test_uar | test_f1 | test_precision | test_recall |
|-------------|-----------|----------|------------------|--------------|----------|---------|-----------------|-------------|
| egemaps     | 0.xxxx    | 0.xxxx   | 0.xxxx           | 0.xxxx       | 0.xxxx   | 0.xxxx  | 0.xxxx          | 0.xxxx      |
| mfcc        | 0.xxxx    | 0.xxxx   | 0.xxxx           | 0.xxxx       | 0.xxxx   | 0.xxxx  | 0.xxxx          | 0.xxxx      |
| hubert_ssl  | 0.xxxx    | 0.xxxx   | 0.xxxx           | 0.xxxx       | 0.xxxx   | 0.xxxx  | 0.xxxx          | 0.xxxx      |
| wav2vec2_ssl| 0.xxxx    | 0.xxxx   | 0.xxxx           | 0.xxxx       | 0.xxxx   | 0.xxxx  | 0.xxxx          | 0.xxxx      |

### Confusion Matrices

- **PNG**: High-resolution visualizations at `artifacts/eval/<feature_set>/confusion_matrix_{valid|test}.png`
- **CSV**: Raw matrices for further analysis
- **Consistent labels**: Same class order across all feature sets (saved in `labels.json`)

## Comparison with Notebook 06

| Aspect | Notebook 06 | This Pipeline |
|--------|-------------|---------------|
| Feature set | eGeMAPS only | 4 feature sets |
| Optuna trials | 40 | 40 (configurable) |
| Selection criterion | max UAR | max UAR (same) |
| SMOTE | Yes | Yes (same) |
| Imputation | Median | Median (same) |
| Eval splits | Valid + Test | Valid + Test (same) |
| Confusion matrices | Both splits | Both splits (same) |
| Reproducibility | Ad-hoc | Fully scripted |

## Troubleshooting

### CUDA/GPU Issues

If you encounter GPU memory issues with SSL models:

```python
# In feature_extractors.py, change:
DEVICE = "cpu"  # Force CPU
```

### Missing Artifact Directory

```bash
# Ensure data pipeline has been run first
python -m vocalbaby.pipeline.training_pipeline
```

### Missing Best Params

```bash
# Run tuning before training
python -m vocalbaby.experiments.scripts.tune_hyperparams --feature-set <name>
```

## Development Notes

### Adding a New Feature Set

1. Add extractor function to `feature_extractors.py`
2. Add config to `FEATURE_CONFIGS` in `generate_features.py`
3. Add to `FEATURE_SETS` list in all scripts
4. Run pipeline

### Modifying Tuning Search Space

Edit `run_optuna_tuning()` in `hyperparameter_tuning.py`:

```python
params = {
    "max_depth": trial.suggest_int("max_depth", 3, 15),  # Changed range
    # ... other params
}
```

## References

- **Baseline notebook**: `06__xgboost_egemaps_smote_optuna_experiment.ipynb`
- **eGeMAPS**: Eyben et al., "The Geneva Minimalistic Acoustic Parameter Set (GeMAPS)"
- **HuBERT**: Facebook AI's HuBERT model fine-tuned on infant cry data
- **Wav2Vec2**: Facebook AI's Wav2Vec2 model fine-tuned on infant cry data
- **Optuna**: Akiba et al., "Optuna: A Next-generation Hyperparameter Optimization Framework"

## License

Same as project root.
