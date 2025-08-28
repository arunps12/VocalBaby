# **VocalBaby Audio Segmentation and Classification**

VocalBaby is a tool designed to classify audio segments as "Infant" or "Other" using fine-tuned Wav2Vec2 models. It processes `.wav` files, detects voiced segments, classifies them, and generates `.TextGrid` files for analysis, as well as a CSV summary.

---

## **Table of Contents**
1. [Features](#features)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Input and Output Format](#input-and-output-format)
5. [Configuration](#configuration)
6. [Project Structure](#project-structure)
7. [Contributing](#contributing)
8. [License](#license)

---

## **Features**
- Detects voiced segments using RMS energy and bandpass filtering.
- Classifies audio chunks into "Infant" or "Other".
- Outputs a `.TextGrid` file for each audio file with labeled segments.
- Generates a CSV summary of the classification results.

---

## **Installation**

### **Clone the Repository**
```bash
git clone https://github.com/arunps12/VocalBaby.git
cd VocalBaby
```

### **Install Dependencies**
It is recommended to create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # or "venv\Scripts\activate" on Windows
```

Install the dependencies:
```bash
pip install -r requirements.txt
```

---

## **Usage**

### **Step 1: Update Paths in `config.py`**
- **`INPUT_test_audio_DIR`**: Path to the directory containing your `.wav` files.

Open `src/config.py` and update the following:
```python
# src/config.py

INPUT_test_audio_DIR = r"your/path/to/audio_dir"
```

### **Step 2: Add `.wav` Files**
Place the `.wav` files you want to classify inside the directory specified in `INPUT_test_audio_DIR`.

---

## **Run the Audio Segmentation Pipeline**

To classify and segment audio files, run the following command:
```bash
python src/infant_voice_segmentation_pipeline.py
```

### **What Happens:**
- **TextGrid Files**: `.TextGrid` files are created in the same directory as the audio files.
- **CSV Summary**: `classification_summary.csv` is generated with details for each audio file.

---

## **Input and Output Format**

### **Input Format**
- `.wav` audio files in the directory specified by `INPUT_test_audio_DIR`.

### **Output Format**
1. **TextGrid files**:
   - Format: `.TextGrid`
   - Contains labeled intervals of "Infant" or "Other".
2. **CSV Summary (`classification_summary.csv`)**:
   | audio_file_name | infant_count | other_count | total_chunks |
   |-----------------|--------------|-------------|--------------|

---

## **Configuration**
- `config.py` contains the default paths used in the pipeline. You can modify these paths to match your local setup:
  - `INPUT_test_audio_DIR`: Path to your audio files.
  - `MODELS_DIR`: Path where the fine-tuned model is saved.

---

## **Project Structure**
```
VocalBaby/
├── data/
│   ├── raw/                              # Raw data directory
│   ├── processed/                        # Processed data directory
├── docs/                                 # Documentation files
├── models/                               # Directory for model files
│   ├── config.json
│   ├── model.safetensors
│   └── preprocessor_config.json
├── notebooks/                            # Jupyter notebooks for testing
│   └── test.ipynb
├── results/                              # Directory to save results
├── src/                                  # Source code directory
│   ├── __init__.py                       # Package initialization
│   ├── config.py                         # Configuration file
│   ├── infant_voice_segmentation_pipeline.py  # Main pipeline script
│   ├── path.py                           # Script to create project structure
│   ├── preprocess_audio.py               # Audio preprocessing module
│   ├── process_metadata.py               # Metadata processing module
│   └── validate_pipeline.py              # Script to validate the pipeline
├── tests/                                # Test files
├── .gitignore                            # Git ignore file
├── classification_summary.csv            # Example CSV summary (output)
├── LICENSE                               # License file
├── README.md                             # Project documentation
├── requirements.txt                      # Python dependencies
└── setup.py                              # Setup script for package installation
```

---

## **Contributing**
Contributions are welcome! Feel free to open issues or submit pull requests:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-name`).
3. Commit your changes (`git commit -am 'Add new feature'`).
4. Push to the branch (`git push origin feature-name`).
5. Open a Pull Request.

---

## **License**
This project is licensed under the MIT License - see the `LICENSE` file for details.

---

## **`requirements.txt`**
```plaintext
torch>=1.9.0
torchaudio>=0.9.0
transformers>=4.24.0
numpy>=1.21.0
librosa>=0.9.1
textgrid>=1.6.1
scipy>=1.5.0
tqdm>=4.64.0
```

