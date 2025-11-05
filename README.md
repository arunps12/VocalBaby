# 🧠 VisionInfantNet

**VisionInfantNet** is a deep learning pipeline for **visual classification of infant vocalizations** using image representations of short audio segments (e.g., Mel-spectrograms).  
The model fine-tunes a **ConvNeXt** image architecture (from the TIMM library) using **FastAI** to classify five types of infant vocalizations:

> **Classes:** `Junk`, `Non-canonical`, `Canonical`, `Cry`, and `Laugh`

The repository also includes data preparation scripts that show how to extract infant vocalization clips from long audio recordings and convert them into Mel-spectrogram images for model training.  
Although the **BabbleCor** corpus used for original training is not publicly shared, this repository allows anyone to use the same training pipeline on any open-source or private dataset **without errors**.

---

## 🚀 Pretrained Model Demo

You can test the pretrained ConvNeXt model (trained on the **BabbleCor** corpus) directly on Hugging Face Spaces:

🔗 **Live Demo:** [https://huggingface.co/spaces/arunps12/VisionInfantNet](https://huggingface.co/spaces/arunps12/VisionInfantNet)

Upload a Mel-spectrogram image and get the predicted vocalization class instantly.

---

## ⚙️ Prerequisites

Ensure you have the following installed:
- Python **3.10+**
- `pip` or `conda` package manager

It is recommended to create and use a **virtual environment** before installing dependencies.

### Create a virtual environment

**Using conda**
```bash
conda create -n visioninfantnet python=3.10
conda activate visioninfantnet
```

**Or using venv**
```bash
python -m venv venv
source venv/bin/activate     # On Windows: venv\Scripts\activate
```

### Install the required packages
```bash
pip install -r requirements.txt
```

This installs all the dependencies, including **FastAI 2.7**, **Torch 2.4**, and **TIMM**.

---

## 🧩 Repository Structure

```
VisionInfantNet/
│
├── pipeline.py                  # Train / evaluate ConvNeXt model using FastAI
├── requirements.txt             # All required Python packages
│
├── scripts/
│   ├── 01_create_data_and_metadata.py    # Extract infant vocal clips + create metadata.csv
│   ├── 02_create_spectral_images.py      # Convert audio clips into Mel-spectrogram images
│
├── data/
│   ├── raw_audio/                # Continuous audio recordings (input)
│   ├── voc_chunks/               # Extracted 1-sec vocalization clips
│   ├── metadata.csv              # Generated metadata file
│   ├── train_imgs/               # Generated training spectrogram images
│   ├── test_imgs/                # Generated test spectrogram images
│
├── models/                       # Saved model weights and reports
│   ├── ConvNext.pkl
│   ├── best.pth
│   ├── val_report.json
│   ├── val_confusion_matrix.csv
│
└── README.md
```

---

## 🎧 Step 1 — Extract Infant Vocalization Segments

When you have **continuous recordings** but know the **start and stop times** of each infant vocalization (e.g., from annotation text files),  
use the script `01_create_data_and_metadata.py` to segment the audio and create the metadata CSV.

### Command
```bash
python scripts/01_create_data_and_metadata.py <raw_audio_dir> <voc_chunks_dir> <metadata_dir>
```

### Example
```bash
python scripts/01_create_data_and_metadata.py     data/raw_audio/     data/voc_chunks/     data/
```

### What it does
1. Reads `.txt` files containing start and end times of each vocalization.
2. Extracts short clips from continuous `.wav` files using **pydub**.
3. Maps annotation codes (e.g., `C`, `X`, `R`) to class names:
   - `C` → Canonical  
   - `X` → Non-canonical  
   - `R` → LaughCry (re-mapped to `Other`)
4. Saves each clip as a separate `.wav` file in `data/voc_chunks/`.
5. Creates a `metadata.csv` file inside `data/` with two columns:
   ```
   file_path,label
   child01_0001_Canonical.wav,Canonical
   child01_0002_Cry.wav,Cry
   child01_0003_Laugh.wav,Laugh
   ...
   ```

---

## 🖼️ Step 2 — Generate Mel-Spectrogram Images

After extracting clips and generating `data/metadata.csv`,  
convert each audio clip into a colorized **Mel-spectrogram image** using `02_create_spectral_images.py`.

### Command
```bash
python scripts/02_create_spectral_images.py <voc_dir> <metadata_df_file_path> <train_dir> <test_dir>
```

### Example
```bash
python scripts/02_create_spectral_images.py     data/voc_chunks/     data/metadata.csv     data/train_imgs/     data/test_imgs/
```

### What it does
1. Loads `.wav` clips listed in `metadata.csv`.
2. Applies **pre-emphasis filtering**, converts to **mono**, and resamples to **16 kHz**.
3. Converts each clip to a **Mel-spectrogram** and colorizes it with the **JET** colormap.
4. Normalizes all clips to 1-second duration:
   - Loops short clips until 1 second.
   - Splits longer clips into 1-second chunks.
5. Automatically splits the dataset into **train/test** sets.
6. Saves spectrogram images (224×224 px) into:
   ```
   data/train_imgs/
   data/test_imgs/
   ```

---

## 🧠 Step 3 — Train and Evaluate the Model

Once image data is ready, train or fine-tune the ConvNeXt classifier using `pipeline.py`.

### 🔹 Train a model
```bash
python pipeline.py train   --data_dir data   --arch convnext_tiny   --epochs 10   --pretrained
```

**Key arguments:**

| Argument | Description |
|-----------|-------------|
| `--data_dir` | Directory containing `train_imgs/` and `test_imgs/` |
| `--arch` | Model architecture (default: `convnext_tiny`) |
| `--epochs` | Number of training epochs |
| `--pretrained` | Use pretrained ImageNet weights |
| `--bs` | Batch size (default: 32) |
| `--classes` | (Optional) Comma-separated list of class names |

The script:
- Builds a FastAI `ImageDataLoaders` from folders or metadata CSV.  
- Fine-tunes the ConvNeXt model.  
- Saves outputs in `models/`:
  ```
  ConvNext.pkl
  best.pth
  val_report.json
  val_confusion_matrix.csv
  ```

---

### 🔹 Evaluate a trained model
```bash
python pipeline.py eval --data_dir data
```

Generates validation metrics including accuracy, F1, precision, recall, and a confusion matrix CSV.

---

## 📊 Results & Reuse

The pretrained **ConvNeXt** model trained on **BabbleCor** accurately classifies infant vocalizations into five categories.  
While **BabbleCor** cannot be shared publicly, the provided pipeline allows **retraining on your own data** using identical steps — no code modifications required.

---

## 🧰 Reusing the Pipeline on Your Data

1. Collect and segment your own infant vocalization clips.  
2. Create a `metadata.csv` with two columns:
   ```csv
   file_path,label
   clip1.wav,Canonical
   clip2.wav,Cry
   ```
3. Generate spectrograms:
   ```bash
   python scripts/02_create_spectral_images.py data/voc_chunks/ data/metadata.csv data/train_imgs/ data/test_imgs/
   ```
4. Train:
   ```bash
   python pipeline.py train --data_dir data --classes "Junk,Non-canonical,Canonical,Cry,Laugh"
   ```
5. Evaluate:
   ```bash
   python pipeline.py eval --data_dir data
   ```

---

## 🧾 Citation

If you use or adapt this repository, please cite:

```
@software{arun_visioninfantnet_2025,
  author       = {Arun Singh},
  title        = {VisionInfantNet: Visual Classification of Infant Vocalizations},
  year         = {2025},
  url          = {https://github.com/arunps12/VisionInfantNet}
}
```

---

## 📬 Contact

**Author:** Arun Singh  
**Affiliation:** University of Oslo, Norway  
**Email:** [arunps@uio.no](mailto:arunps@uio.no)  
**Hugging Face Space:** [https://huggingface.co/spaces/arunps12/VisionInfantNet](https://huggingface.co/spaces/arunps12/VisionInfantNet)

---

## 🪪 License

This project is released under the **MIT License**.  
You are free to use, modify, and distribute it with proper attribution.

---
