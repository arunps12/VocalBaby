# VisionInfantNet  
A Modular Audio Feature Extraction + Machine Learning Pipeline  
for Infant Vocalization Classification

---

## 📌 Overview

**VisionInfantNet** is a complete machine-learning framework designed for 
automatic classification of infant and adult vocalizations.  
It combines:

- 🔹 **eGeMAPS acoustic feature extraction**  
- 🔹 **SMOTE / SMOTE-ENN balancing**  
- 🔹 **XGBoost-based classification**  
- 🔹 **A reusable prediction pipeline**  
- 🔹 **Support for future multimodal models (MFCCs, wav2vec, spectrogram CNNs, audio and image embeddings)**  

The system is structured using a clean, extensible, MLOps-friendly design  
with components housed in the `visioninfantnet/` package.

---

# ⚙️ Installation

### 1️⃣ Clone the repository
```bash
git clone https://github.com/arunps12/VisionInfantNet.git
cd VisionInfantNet
```

### 2️⃣ Create & activate a virtual environment
```bash
python -m venv venv
source venv/bin/activate
# Windows: venv\Scripts\activate
```

### 3️⃣ Install requirements
```bash
pip install -r requirements.txt
```


---

# 🤖 Machine Learning Model (Current Version)

The current production model uses:

- **eGeMAPS features** extracted using openSMILE  
- **XGBoost classifier** tuned with Optuna  
- **SMOTE** oversampling (best performer in your experiments)

### All trained objects live here:

```
projectfolder/final_model/
├── xgb_egemaps_smote_optuna.pkl
├── preprocessing.pkl
└── label_encoder.pkl
```

---

# 🔮 Prediction Pipeline (Production Inference)

The prediction system is implemented in:

```
visioninfantnet/pipeline/prediction_pipeline.py
```

It supports:

- Single `.wav` file  
- List of `.wav` files  
- Entire directory containing several `.wav` files  

---

## ✅ How to Use the Prediction Pipeline

### 1️⃣ Import and initialize the pipeline

```python
from visioninfantnet.pipeline.prediction_pipeline import PredictionPipeline

MODEL_DIR = "final_model"   # Path containing the .pkl files

pipe = PredictionPipeline(model_trainer_dir=MODEL_DIR)
```

---

## 2️⃣ Predict from a Single `.wav`

```python
y_enc, y_dec, paths = pipe.predict_from_audio("samples/test.wav")

print("File:", paths[0])
print("Predicted class index:", int(y_enc[0]))
print("Predicted label:", y_dec[0])
```

---

## 3️⃣ Predict a Whole Directory

```python
y_enc, y_dec, paths = pipe.predict_from_audio("samples/test_clips/")

for p, enc, dec in zip(paths, y_enc, y_dec):
    print(f"{p} -> {dec} ({int(enc)})")
```

---

## 4️⃣ Predict from a List of Files

```python
files = ["a.wav", "b.wav", "c.wav"]
y_enc, y_dec, paths = pipe.predict_from_audio(files)
```

---

# 📤 What the Pipeline Returns

| Output | Type | Meaning |
|--------|-------|---------|
| `y_pred_encoded` | `np.ndarray` | Encoded class indices |
| `y_pred_decoded` | `np.ndarray` | Human-readable class labels |
| `audio_paths` | `List[str]` | Files used for prediction |

---

# 🔧 Requirements

- Python 3.10
- openSMILE (for eGeMAPS)
- XGBoost
- NumPy + SciPy + Scikit-learn

---

# 🚀 Future Enhancements

You can extend VisionInfantNet to:

- CNN models over mel-spectrogram images 
- Other image model like ResNet50 over mel-spectrogram images
- wav2vec2 embeddings  
- Hybrid prosody + embedding features  
- Temporal models (LSTMs, Transformers)  
- Real-time prediction service  

---

# 📄 License

MIT License — free to use, modify, and distribute.

---

# 🙏 Acknowledgements

This project is part of research at the **University of Oslo (UiO)**  
studying infant speech development and multimodal learning.

---

## 🌟 About Me

Hi there! I'm **Arun Prakash Singh**, a **Marie Curie Research Fellow at the University of Oslo (UiO)**.  
My research focuses on **speech technology, data engineering, and machine learning**, with an emphasis on building intelligent, data-driven systems that model human communication and learning.  
I am passionate about integrating **AI, analytics, and large-scale data pipelines** to advance our understanding of how humans process and acquire language.
