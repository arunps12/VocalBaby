# VocalBaby — Feature Comparison Pipeline for Infant Vocalization Classification

[![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![uv](https://img.shields.io/badge/uv-Astral-DE5FE9?logo=astral&logoColor=white)](https://docs.astral.sh/uv/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7-FF6600?logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Container-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![Prometheus](https://img.shields.io/badge/Prometheus-Monitoring-E6522C?logo=prometheus&logoColor=white)](https://prometheus.io/)
[![Grafana](https://img.shields.io/badge/Grafana-Dashboards-F46800?logo=grafana&logoColor=white)](https://grafana.com/)
[![Terraform](https://img.shields.io/badge/Terraform-IaC-7B42BC?logo=terraform&logoColor=white)](https://www.terraform.io/)
[![DVC](https://img.shields.io/badge/DVC-Pipeline-13ADC7?logo=dvc&logoColor=white)](https://dvc.org/)
[![GitHub Actions](https://img.shields.io/badge/GitHub_Actions-CI/CD-2088FF?logo=githubactions&logoColor=white)](https://github.com/features/actions)
[![AWS](https://img.shields.io/badge/AWS-ECR%20%7C%20EC2%20%7C%20S3-FF9900?logo=amazonaws&logoColor=white)](https://aws.amazon.com/)
[![Evidently](https://img.shields.io/badge/Evidently-Drift_Detection-4B32C3)](https://www.evidentlyai.com/)
[![openSMILE](https://img.shields.io/badge/openSMILE-eGeMAPS-blue)](https://audeering.github.io/opensmile-python/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![Optuna](https://img.shields.io/badge/Optuna-HPO-1674B1)](https://optuna.org/)
[![Ruff](https://img.shields.io/badge/Ruff-Linter-D7FF64?logo=ruff&logoColor=black)](https://docs.astral.sh/ruff/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**A reproducible ML pipeline for comparing XGBoost model performance across multiple acoustic feature sets for infant vocalization classification.**

---

## Project Purpose

This project provides a **clean, reproducible pipeline** to:

1. **Compare XGBoost performance** across 4 acoustic feature sets:
   - **eGeMAPS** (88-dim openSMILE features)
   - **MFCC** (20/40-dim librosa features)
   - **HuBERT SSL** (768-dim embeddings from [arunps/hubert-home-hindibabynet-ssl](https://huggingface.co/arunps/hubert-home-hindibabynet-ssl))
   - **Wav2Vec2 SSL** (768-dim embeddings from [arunps/wav2vec2-home-hindibabynet-ssl](https://huggingface.co/arunps/wav2vec2-home-hindibabynet-ssl))

2. **Find optimal hyperparameters** for each feature set independently using Optuna (40 trials, multi-objective: UAR + F1)

3. **Generate comprehensive evaluation artifacts**:
   - Confusion matrices for **both validation and test** splits
   - Classification reports with per-class metrics
   - Aggregated comparison table across all feature sets

4. **Maintain full reproducibility** via DVC pipeline and `params.yaml` configuration

---

## Overview

**VocalBaby** is an end-to-end multi-class audio classification system designed for child language acquisition research. It classifies short infant audio segments into **5 vocalization categories**:

| Class | Description | Samples |
|-------|-------------|--------:|
| **Non-canonical** | Non-canonical infant vocalizations (e.g., squeals, growls, vowel-like sounds) | 5,606 |
| **Junk** | Background noise, silence, or non-speech artifacts | 4,974 |
| **Canonical** | Canonical babbling (consonant–vowel syllables like "ba", "da") | 1,826 |
| **Crying** | Infant cry episodes | 823 |
| **Laughing** | Infant laughter | 241 |

The system focuses on infant and adult vocalizations in naturalistic interaction recordings.

---

## Quick Start

### Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) package manager (recommended) or pip
- DVC (for pipeline orchestration)
- Git (for version control)

### Installation

```bash
# Clone repository
git clone https://github.com/your-username/VisionInfantNet.git
cd VisionInfantNet

# Create virtual environment and install dependencies (using uv)
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e .

# Or using pip
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### Running the Pipeline

#### Option 1: DVC Pipeline (Recommended)

Run the entire 7-stage pipeline with DVC:

```bash
# Run complete pipeline (all 7 stages)
dvc repro

# Run specific stage and its dependencies
dvc repro ingest      # Stage 01: Data ingestion only
dvc repro validate    # Stages 01-02: Through validation
dvc repro evaluate    # Stages 01-06: Through evaluation
dvc repro aggregate   # Full pipeline: All 7 stages
```

#### Option 2: Manual Stage Execution

Run individual stages via bash scripts:

```bash
# Stage 01: Data Ingestion
bash scripts/01_ingest.sh

# Stage 02: Data Validation
bash scripts/02_validate.sh

# Stage 03: Feature Extraction (all feature sets from params.yaml)
bash scripts/03_transform.sh

# Or extract specific feature set(s)
bash scripts/03_transform.sh --feature-sets mfcc

# Stage 04: Hyperparameter Tuning
bash scripts/04_tune.sh

# Or tune specific feature set(s) with custom trial count
bash scripts/04_tune.sh --feature-sets egemaps mfcc --n-trials 20

# Stage 05: Model Training
bash scripts/05_train.sh

# Stage 06: Model Evaluation
bash scripts/06_evaluate.sh

# Stage 07: Results Aggregation
bash scripts/07_aggregate.sh
```

#### Option 3: Python Module Execution

Run stages directly as Python modules:

```bash
python -m vocalbaby.pipeline.stage_01_ingest
python -m vocalbaby.pipeline.stage_02_validate
python -m vocalbaby.pipeline.stage_03_transform --feature-sets mfcc --force
python -m vocalbaby.pipeline.stage_04_tune --n-trials 10
python -m vocalbaby.pipeline.stage_05_train
python -m vocalbaby.pipeline.stage_06_evaluate
python -m vocalbaby.pipeline.stage_07_aggregate
```

---

## Pipeline Architecture

### 7-Stage ML Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 01: DATA INGESTION                                       │
│  • Loads raw audio + metadata                                   │
│  • Creates child-disjoint train/valid/test splits               │
│  Output: artifacts/latest/data_ingestion/                       │
└───────────────────────┬─────────────────────────────────────────┘
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 02: DATA VALIDATION                                      │
│  • Schema validation                                            │
│  • Data quality checks                                          │
│  • Drift detection                                              │
│  Output: artifacts/latest/data_validation/                      │
└───────────────────────┬─────────────────────────────────────────┘
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 03: FEATURE EXTRACTION                                   │
│  • eGeMAPS (88-dim openSMILE)                                   │
│  • MFCC (20/40-dim librosa)                                     │
│  • HuBERT SSL (768-dim transformers)                            │
│  • Wav2Vec2 SSL (768-dim transformers)                          │
│  Output: artifacts/features/<feature_set>/<split>/              │
└───────────────────────┬─────────────────────────────────────────┘
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 04: HYPERPARAMETER TUNING                                │
│  • Optuna optimization (40 trials)                              │
│  • Multi-objective: UAR + Macro F1                              │
│  • Independent tuning per feature set                           │
│  Output: artifacts/models/<feature_set>/best_params.json        │
└───────────────────────┬─────────────────────────────────────────┘
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 05: MODEL TRAINING                                       │
│  • XGBoost training with best params                            │
│  • SMOTE oversampling (k=5, random_state=42)                    │
│  • Label encoding + median imputation                           │
│  Output: artifacts/models/<feature_set>/xgb_model.pkl           │
└───────────────────────┬─────────────────────────────────────────┘
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 06: MODEL EVALUATION                                     │
│  • Evaluation on valid + test splits                            │
│  • Confusion matrices (PNG + CSV)                               │
│  • Classification reports                                       │
│  • Metrics: Accuracy, UAR, F1, Precision, Recall                │
│  Output: artifacts/eval/<feature_set>/                          │
└───────────────────────┬─────────────────────────────────────────┘
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 07: RESULTS AGGREGATION                                  │
│  • Comparison table across all feature sets                     │
│  • Metrics for both valid and test splits                       │
│  Output: artifacts/results/results_summary.csv                  │
└─────────────────────────────────────────────────────────────────┘
```

### Feature Sets

| Feature Set | Dimension | Extractor | Description |
|-------------|-----------|-----------|-------------|
| **egemaps** | 88 | openSMILE | Extended Geneva Minimalistic Acoustic Parameter Set |
| **mfcc** | 40 | librosa | Mel-Frequency Cepstral Coefficients (mean+std pooling) |
| **hubert_ssl** | 768 | HuggingFace Transformers | HuBERT SSL embeddings from [arunps/hubert-home-hindibabynet-ssl](https://huggingface.co/arunps/hubert-home-hindibabynet-ssl) |
| **wav2vec2_ssl** | 768 | HuggingFace Transformers | Wav2Vec2 SSL embeddings from [arunps/wav2vec2-home-hindibabynet-ssl](https://huggingface.co/arunps/wav2vec2-home-hindibabynet-ssl) |

#### Self-Supervised Learning (SSL) Models

The **HuBERT** and **Wav2Vec2** models used in this pipeline are **custom fine-tuned versions** trained on infant vocalization data:

- **[arunps/hubert-home-hindibabynet-ssl](https://huggingface.co/arunps/hubert-home-hindibabynet-ssl)** - HuBERT model fine-tuned on home environment infant speech recordings from the HindiBabyNet corpus. This model is specifically adapted for infant vocalization patterns and home recording conditions.

- **[arunps/wav2vec2-home-hindibabynet-ssl](https://huggingface.co/arunps/wav2vec2-home-hindibabynet-ssl)** - Wav2Vec2 model fine-tuned on the same infant speech data. It provides complementary self-supervised representations optimized for infant cry and babbling classification.

**Why SSL models?** Self-supervised learning models pre-trained on large speech corpora and fine-tuned on domain-specific data (infant vocalizations) often capture richer acoustic-phonetic representations than traditional hand-crafted features. These 768-dimensional embeddings encode temporal dynamics, prosodic patterns, and spectral characteristics that are particularly useful for distinguishing between canonical babbling, non-canonical vocalizations, crying, and laughing.

**Model Architecture:**
- Base architecture: HuBERT-base / Wav2Vec2-base (12 transformer layers)
- Hidden size: 768 dimensions
- Fine-tuning: Infant vocalization data from naturalistic home recordings
- Inference: Mean-pooled temporal embeddings (768-dim fixed-length vectors)

---

## Project Structure

```
.
├── params.yaml                    # Central configuration file
├── dvc.yaml                       # DVC pipeline definition
├── pyproject.toml                 # Package metadata & dependencies
├── README.md                      # This file
│
├── data/                          # Raw data (not versioned)
│   ├── audio/raw/                 # Raw .wav files
│   └── metadata/                  # Metadata CSV files
│
├── data_schema/                   # Data validation schemas
│   └── schema.yaml
│
├── scripts/                       # Bash runner scripts
│   ├── 01_ingest.sh              # Stage 01: Data ingestion
│   ├── 02_validate.sh            # Stage 02: Data validation
│   ├── 03_transform.sh           # Stage 03: Feature extraction
│   ├── 04_tune.sh                # Stage 04: Hyperparameter tuning
│   ├── 05_train.sh               # Stage 05: Model training
│   ├── 06_evaluate.sh            # Stage 06: Model evaluation
│   └── 07_aggregate.sh           # Stage 07: Results aggregation
│
├── src/vocalbaby/                 # Main package (src-layout)
│   ├── config/                    # Configuration management
│   │   └── schemas.py
│   ├── data/                      # Data ingestion & validation
│   │   ├── ingest.py
│   │   └── validate.py
│   ├── features/                  # Feature extractors
│   │   ├── base.py
│   │   ├── egemaps.py
│   │   ├── mfcc.py
│   │   ├── hubert.py
│   │   └── wav2vec2.py
│   ├── models/                    # Hyperparameter tuning & training
│   ├── eval/                      # Evaluation & metrics
│   ├── pipeline/                  # 7 pipeline stage modules
│   │   ├── stage_01_ingest.py
│   │   ├── stage_02_validate.py
│   │   ├── stage_03_transform.py
│   │   ├── stage_04_tune.py
│   │   ├── stage_05_train.py
│   │   ├── stage_06_evaluate.py
│   │   └── stage_07_aggregate.py
│   ├── components/                # Legacy components (kept for compatibility)
│   ├── utils/                     # Utility functions
│   ├── logging/                   # Logging configuration
│   ├── exception/                 # Exception handling
│   └── cli.py                     # Command-line interface
│
├── artifacts/                     # Pipeline outputs (DVC-tracked)
│   ├── data/                      # Ingested + validated data
│   ├── features/                  # Feature arrays
│   │   └── <feature_set>/
│   │       ├── train/
│   │       ├── valid/
│   │       └── test/
│   ├── models/                    # Trained models
│   │   └── <feature_set>/
│   │       ├── xgb_model.pkl
│   │       ├── best_params.json
│   │       ├── label_encoder.pkl
│   │       └── imputer.pkl
│   ├── eval/                      # Evaluation results
│   │   └── <feature_set>/
│   │       ├── confusion_matrix_valid.png
│   │       ├── confusion_matrix_test.png
│   │       ├── metrics_valid.json
│   │       └── metrics_test.json
│   └── results/                   # Aggregated results
│       └── results_summary.csv    # Main comparison table
│
├── notebooks/                     # Exploratory notebooks (not part of pipeline)
│   ├── 01_EDA.ipynb
│   ├── 02__feature_and_model_selection_experiments.ipynb
│   ├── ...
│   └── 06__xgboost_egemaps_smote_optuna_experiment.ipynb
│
└── tests/                         # Unit tests
```

---

## Configuration

All pipeline parameters are centralized in [`params.yaml`](params.yaml). Key sections:

### Data Configuration
```yaml
data:
  raw_audio_dir: data/audio/raw
  raw_metadata_file: data/metadata/private_metadata.csv
  seed: 42
```

### Feature Sets
```yaml
features:
  sets:
    - egemaps
    - mfcc
    - hubert_ssl
    - wav2vec2_ssl
  
  mfcc:
    n_mfcc: 20
    sample_rate: 16000
    pool: mean_std  # 40-dim output (mean+std)
```

### Hyperparameter Tuning
```yaml
tuning:
  framework: optuna
  n_trials: 40
  objectives:
    - uar        # Unweighted Average Recall
    - macro_f1   # Macro F1-score
  
  xgb_search_space:
    max_depth: [3, 10]
    learning_rate: [0.001, 0.3]
    n_estimators: [100, 500]
    # ... more parameters
```

### Evaluation
```yaml
evaluation:
  metrics:
    - accuracy
    - balanced_accuracy  # UAR
    - macro_f1
    - weighted_f1
  
  confusion_matrix:
    normalize: false
    save_csv: true
    save_png: true
```

Modify `params.yaml` and re-run `dvc repro` to update the pipeline.

---

## Results & Artifacts

### Key Output Files

1. **`artifacts/results/results_summary.csv`** - Main comparison table
   ```csv
   feature_set,split,n_samples,accuracy,balanced_accuracy,macro_f1,...
   egemaps,test,1000,0.82,0.78,0.75,...
   mfcc,test,1000,0.79,0.75,0.72,...
   hubert_ssl,test,1000,0.84,0.81,0.78,...
   wav2vec2_ssl,test,1000,0.85,0.82,0.79,...
   ```

2. **Confusion Matrices** - PNG + CSV for each feature set × split
   - `artifacts/eval/egemaps/confusion_matrix_valid.png`
   - `artifacts/eval/egemaps/confusion_matrix_test.png`
   - ... (same for mfcc, hubert_ssl, wav2vec2_ssl)

3. **Best Hyperparameters** - JSON for each feature set
   - `artifacts/models/egemaps/best_params.json`
   - `artifacts/models/mfcc/best_params.json`
   - ... etc

4. **Trained Models** - Pickled XGBoost models
   - `artifacts/models/<feature_set>/xgb_model.pkl`
   - `artifacts/models/<feature_set>/label_encoder.pkl`
   - `artifacts/models/<feature_set>/imputer.pkl`

### Viewing Results

```bash
# View aggregated comparison table
cat artifacts/results/results_summary.csv

# View confusion matrices (if on remote server, use scp/rsync to download)
open artifacts/eval/egemaps/confusion_matrix_test.png

# View best hyperparameters
cat artifacts/models/egemaps/best_params.json | python -m json.tool

# View evaluation metrics
cat artifacts/eval/egemaps/metrics_test.json | python -m json.tool
```

---

## Methodology

### Data Preprocessing

1. **Child-Disjoint Splits** - Train/valid/test splits ensure no child appears in multiple splits
2. **Label Encoding** - Categorical labels converted to integers (0-4)
3. **Missing Value Imputation** - Median imputation for any missing features
4. **Class Imbalance Handling** - SMOTE oversampling (k=5) applied to training set only

### Hyperparameter Optimization

- **Framework**: Optuna with TPESampler
- **Trials**: 40 per feature set
- **Objectives**: Unweighted Average Recall (UAR) + Macro F1-score
- **Search Space**: Independent for each feature set (see `params.yaml`)
- **Validation**: Best params selected based on validation set performance

### Model Training

- **Algorithm**: XGBoost (multi:softmax objective)
- **Early Stopping**: 50 rounds on validation set
- **Random State**: 42 (for reproducibility)
- **Preprocessing**: Applied consistently (imputation → SMOTE → label encoding)

### Evaluation

- **Splits**: Both validation and test (ensures no overfitting)
- **Metrics**: Accuracy, Balanced Accuracy (UAR), Macro/Weighted F1, Precision, Recall
- **Confusion Matrices**: Raw counts (not normalized) saved as PNG + CSV

---

## System Architecture & Workflow

```mermaid
flowchart TB
    subgraph DATA["Data Layer"]
        direction LR
        RAW["Raw .wav Audio<br/><i>Naturalistic Recordings</i>"]
        META["Metadata CSV<br/><i>child_ID · age · gender · Answer · corpus</i>"]
    end

    subgraph DVC_PIPELINE["DVC Pipeline  <i>(dvc repro)</i>"]
        direction TB
        ING["<b>1 — Data Ingestion</b><br/>Train / Valid / Test split"]
        VAL["<b>2 — Data Validation</b><br/>Schema check · Drift guard"]
        TRANS["<b>3 — Feature Extraction</b><br/>openSMILE eGeMAPS → .npy"]
        TRAIN["<b>4 — Model Training</b><br/>SMOTE · XGBoost · Optuna"]
    end

    subgraph ML_TOOLS["ML & Feature Stack"]
        direction LR
        SMILE["openSMILE<br/><i>eGeMAPS features</i>"]
        SMOTE["imbalanced-learn<br/><i>SMOTE oversampling</i>"]
        XGB["XGBoost<br/><i>Classifier</i>"]
        OPTUNA["Optuna<br/><i>Hyperparameter tuning</i>"]
        SKLEARN["scikit-learn<br/><i>Preprocessing · Metrics</i>"]
    end

    subgraph ARTIFACTS["Model Artifacts"]
        direction LR
        MODEL["xgb_egemaps_smote_optuna.pkl"]
        PREPROC["preprocessing.pkl"]
        ENCODER["label_encoder.pkl"]
    end

    subgraph SERVING["Serving Layer"]
        direction TB
        API["<b>FastAPI Server</b><br/><i>vocalbaby-serve · port 8000</i>"]
        PREDICT["Prediction Pipeline<br/><i>/predict · /predict_zip</i>"]
        METRICS_EP["/metrics endpoint"]
    end

    subgraph MONITORING["Monitoring Stack"]
        direction LR
        PROM["Prometheus<br/><i>Scrape metrics · Alerts</i>"]
        GRAF["Grafana<br/><i>Dashboards · Visualization</i>"]
        EVID["Evidently<br/><i>Data Drift Detection</i>"]
    end

    subgraph CICD["CI/CD  <i>(GitHub Actions)</i>"]
        direction TB
        LINT["<b>Lint & Test</b><br/>Ruff · pytest"]
        BUILD["<b>Build & Push</b><br/>Docker → ECR"]
        DEPLOY["<b>Deploy</b><br/>EC2 pull & restart"]
        DRIFT_CRON["<b>Nightly Drift</b><br/>Scheduled cron job"]
    end

    subgraph INFRA["AWS Infrastructure  <i>(Terraform)</i>"]
        direction LR
        VPC["VPC / Subnets<br/><i>Networking module</i>"]
        ECR["ECR<br/><i>Container Registry</i>"]
        EC2["EC2<br/><i>Compute Instance</i>"]
        S3["S3<br/><i>Artifact Storage</i>"]
        IAM["IAM<br/><i>Roles & Policies</i>"]
    end

    subgraph PKG["Packaging & Tooling"]
        direction LR
        UV["uv  <i>(Astral)</i><br/>Dependency management"]
        PYPROJ["pyproject.toml<br/><i>PEP 621 · Hatchling</i>"]
        DOCKER["Docker<br/><i>Container build</i>"]
        RUFF["Ruff<br/><i>Lint & Format</i>"]
    end

    %% ── Data flows ──
    RAW --> ING
    META --> ING
    ING --> VAL --> TRANS --> TRAIN

    %% ── ML tool connections ──
    TRANS -.-> SMILE
    TRAIN -.-> SMOTE
    TRAIN -.-> XGB
    TRAIN -.-> OPTUNA
    TRAIN -.-> SKLEARN

    %% ── Artifacts ──
    TRAIN --> MODEL
    TRAIN --> PREPROC
    TRAIN --> ENCODER

    %% ── Serving ──
    MODEL --> API
    PREPROC --> API
    ENCODER --> API
    API --> PREDICT
    API --> METRICS_EP

    %% ── Monitoring ──
    METRICS_EP --> PROM
    PROM --> GRAF
    EVID --> PROM
    TRANS -.->|reference data| EVID

    %% ── CI/CD ──
    LINT --> BUILD --> DEPLOY
    DRIFT_CRON -.-> EVID

    %% ── Infrastructure ──
    BUILD --> ECR
    DEPLOY --> EC2
    TRAIN --> S3
    EC2 -.-> ECR
    EC2 -.-> S3

    %% ── Packaging ──
    UV -.-> PYPROJ
    PYPROJ -.-> DOCKER

    %% ── Styling ──
    classDef dataStyle fill:#e3f2fd,stroke:#1565c0,color:#0d47a1
    classDef pipeStyle fill:#e8f5e9,stroke:#2e7d32,color:#1b5e20
    classDef mlStyle fill:#fff3e0,stroke:#e65100,color:#bf360c
    classDef artifactStyle fill:#f3e5f5,stroke:#6a1b9a,color:#4a148c
    classDef serveStyle fill:#e0f7fa,stroke:#00838f,color:#006064
    classDef monStyle fill:#fce4ec,stroke:#c62828,color:#b71c1c
    classDef cicdStyle fill:#e8eaf6,stroke:#283593,color:#1a237e
    classDef infraStyle fill:#fff8e1,stroke:#f57f17,color:#e65100
    classDef pkgStyle fill:#f1f8e9,stroke:#558b2f,color:#33691e

    class RAW,META dataStyle
    class ING,VAL,TRANS,TRAIN pipeStyle
    class SMILE,SMOTE,XGB,OPTUNA,SKLEARN mlStyle
    class MODEL,PREPROC,ENCODER artifactStyle
    class API,PREDICT,METRICS_EP serveStyle
    class PROM,GRAF,EVID monStyle
    class LINT,BUILD,DEPLOY,DRIFT_CRON cicdStyle
    class VPC,ECR,EC2,S3,IAM infraStyle
    class UV,PYPROJ,DOCKER,RUFF pkgStyle
```

---

## Quick Start

### Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) (Astral package manager)

### Setup

```bash
# Install uv (if not already)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and install
git clone https://github.com/arunps12/VocalBaby.git
cd VocalBaby
uv sync

# Verify installation
uv run python -c "import vocalbaby; print(vocalbaby.__version__)"
```

### Run the Prediction Server

```bash
# Via console entry point
uv run vocalbaby-serve

# Or via uvicorn directly
uv run uvicorn vocalbaby.api.app:app --host 0.0.0.0 --port 8000
```

The server runs at `http://localhost:8000`:
- **Swagger docs:** http://localhost:8000/docs
- **Prometheus metrics:** http://localhost:8000/metrics
- **Health check:** http://localhost:8000/health

### Run the Training Pipeline

```bash
uv run vocalbaby-train
```

---

## Project Structure

```
VocalBaby/
├── src/vocalbaby/           # Main package (src layout)
│   ├── api/                 # FastAPI prediction server
│   │   └── app.py
│   ├── cli.py               # Console entry points
│   ├── components/          # Pipeline components
│   │   ├── data_ingestion.py
│   │   ├── data_validation.py
│   │   ├── data_transformation.py
│   │   └── model_trainer.py
│   ├── pipeline/            # Training & prediction pipelines
│   ├── monitoring/          # Prometheus metrics + Evidently drift
│   │   ├── metrics.py
│   │   └── drift.py
│   ├── entity/              # Data classes & configs
│   ├── exception/           # Custom exceptions
│   ├── logging/             # Logging setup
│   ├── cloud/               # S3 sync
│   ├── constant/            # Pipeline constants
│   └── utils/               # ML utilities
├── configs/                 # YAML configuration files
├── scripts/                 # DVC stage runner scripts
├── tests/                   # Test suite
├── monitoring/              # Grafana & Prometheus configs
│   ├── grafana/
│   └── prometheus/
├── infra/terraform/         # Infrastructure as Code
│   ├── modules/
│   └── envs/production/
├── pyproject.toml           # PEP 621 package definition
├── uv.lock                  # Lockfile (uv)
├── dvc.yaml                 # DVC pipeline stages
├── Dockerfile               # uv-based container build
└── docker-compose.monitoring.yml
```

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Redirect to Swagger docs |
| `/health` | GET | Health check |
| `/predict` | POST | Classify uploaded .wav file(s) into vocalization categories |
| `/predict_zip` | POST | Classify a ZIP of .wav files into vocalization categories |
| `/metrics` | GET | Prometheus metrics |

### Example: Predict

```bash
curl -X POST http://localhost:8000/predict \
  -F "files=@segment.wav"
```

---

## Machine Learning Model

The current production model uses:

- **eGeMAPS features** extracted using openSMILE
- **XGBoost classifier** tuned with Optuna
- **SMOTE** oversampling (best performer in experiments)

### Trained objects

```
final_model/
├── xgb_egemaps_smote_optuna.pkl
├── preprocessing.pkl
└── label_encoder.pkl
```

---

## Prediction Pipeline

```python
from vocalbaby.pipeline.prediction_pipeline import PredictionPipeline

pipe = PredictionPipeline(model_trainer_dir="final_model")

# Single file
y_enc, y_dec, paths = pipe.predict_from_audio("samples/test.wav")

# Whole directory
y_enc, y_dec, paths = pipe.predict_from_audio("samples/test_clips/")

# List of files
y_enc, y_dec, paths = pipe.predict_from_audio(["a.wav", "b.wav", "c.wav"])
```

| Output | Type | Meaning |
|--------|------|---------|
| `y_pred_encoded` | `np.ndarray` | Encoded class indices |
| `y_pred_decoded` | `np.ndarray` | Human-readable class labels |
| `audio_paths` | `List[str]` | Files used for prediction |

---

## DVC Pipeline

```bash
# Run full pipeline
dvc repro

# Run specific stages
dvc repro ingestion
dvc repro training
dvc repro drift
```

---

## Monitoring Stack

```bash
# Start VocalBaby API + Prometheus + Grafana
docker compose -f docker-compose.monitoring.yml up -d
```

| Service | URL | Credentials |
|---------|-----|-------------|
| API | http://localhost:8000 | — |
| Prometheus | http://localhost:9090 | — |
| Grafana | http://localhost:3000 | admin / vocalbaby |

### Prometheus Metrics

- `vocalbaby_request_total` — Request counter by endpoint/status
- `vocalbaby_request_latency_seconds` — Latency histogram
- `vocalbaby_prediction_errors_total` — Error counter
- `vocalbaby_model_info` — Model version metadata
- `vocalbaby_drift_score` — Drift detection score

---

## Drift Detection

```bash
# Run drift detection
bash scripts/run_drift.sh

# Or directly
uv run python -c "from vocalbaby.monitoring.drift import run_drift_report; run_drift_report()"
```

Reports are saved to `artifacts/drift/`.

---

## Docker

```bash
# Build
docker build -t vocalbaby .

# Run
docker run -p 8000:8000 -v ./final_model:/app/final_model:ro vocalbaby
```

---

## Infrastructure (Terraform)

```bash
cd infra/terraform/envs/production
cp terraform.tfvars.example terraform.tfvars
# Edit terraform.tfvars with your values

terraform init
terraform plan
terraform apply
```

See `infra/terraform/README.md` for detailed IaC documentation.

---

## Development

```bash
# Install with dev dependencies
uv sync --dev

# Run tests
uv run pytest tests/ -v

# Lint
uv run ruff check src/

# Format
uv run ruff format src/
```

---

## CI/CD

The GitHub Actions workflow (`.github/workflows/main.yml`) handles:

1. **Lint & Test** — `uv sync` → `ruff check` → `pytest`
2. **Build & Push** — Docker image → ECR
3. **Deploy** — Pull & run on EC2
4. **Nightly Drift** — Scheduled drift detection (cron)

---

## Future Enhancements

- CNN models over mel-spectrogram images
- ResNet50 over mel-spectrogram images
- wav2vec2 embeddings
- Hybrid prosody + embedding features
- Temporal models (LSTMs, Transformers)
- Fine-grained sub-class detection within canonical/non-canonical categories
- Multi-corpus generalization and cross-lingual transfer

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Acknowledgements

This project is part of research at the **University of Oslo (UiO)**
studying infant speech development and multimodal learning.

---

## About Me

Hi there! I'm **Arun Prakash Singh**, a **Marie Curie Research Fellow at the University of Oslo (UiO)**.
My research focuses on **speech technology, data engineering, and machine learning**, with an emphasis on building intelligent, data-driven systems that model human communication and learning.
I am passionate about integrating **AI, analytics, and large-scale data pipelines** to advance our understanding of how humans process and acquire language.
