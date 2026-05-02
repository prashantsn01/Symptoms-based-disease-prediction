# 🩺 Symptoms-Based Disease Prediction

An AI-powered web application that predicts diseases from user-selected symptoms using an **8-model hybrid ensemble** combining Deep Learning and classical Machine Learning.

> **Academic Project** — Dept. of ISE, DSCE (AY 2025–26)

---

## ✨ Highlights

| Metric | Target |
|--------|--------|
| Accuracy | ≥ 99% |
| F1-Score | ≥ 0.95 |
| Precision | ≥ 0.96 |
| Inference Latency | < 1 sec |

- **773 diseases** · **377 symptoms** · **246,945 training samples**
- 3 Deep Learning models + 5 ML classifiers in a voting ensemble
- Memory-optimised sequential DL training (fits limited VRAM)
- Responsive dark-themed web UI with searchable symptom picker
- Optional NLP mode — describe symptoms in plain text

---

## 🏗️ Architecture

```
Symptoms Vector
      │
      ├─── Dense NN (256 → 128)
      ├─── Conv1D (64 filters)
      ├─── Attention Network
      ├─── Logistic Regression
      ├─── Naive Bayes
      ├─── Decision Tree
      ├─── Random Forest (100 trees)
      └─── XGBoost
              │
        Probability Averaging
              │
        Predicted Disease
```

---

## 📁 Project Structure

```
symptoms-based-disease-prediction/
│
├── app.py                  # Flask entry point
├── utils.py                # Backwards-compatibility shim
├── requirements.txt        # Pinned dependencies
│
├── src/
│   ├── data.py             # Dataset loading & label encoding
│   ├── model.py            # DL architectures + HybridDeepModel
│   └── nlp/
│       ├── base.py         # Extractor interface & factory
│       ├── transformer_extractor.py
│       └── llm_extractor.py
│
├── scripts/
│   ├── train.py            # Train the ensemble
│   └── evaluate.py         # Evaluate on test split
│
├── data/                   # ⚠️ Add dataset here (not tracked in git)
├── models/                 # ⚠️ Populated after training (not tracked)
├── templates/
│   └── index.html
└── static/
    └── css/style.css
```

---

## 🚀 Quick Start

### 1. Clone & Setup

```bash
git clone https://github.com/<your-username>/symptoms-based-disease-prediction.git
cd symptoms-based-disease-prediction

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Add the Dataset

```
data/Final_Augmented_dataset_Diseases_and_Symptoms.csv
```

### 3. Train

```bash
python -m scripts.train
```

~30–40 min on CPU/low-end GPU.

### 4. Run

```bash
python app.py
# Open http://127.0.0.1:5000
```

### 5. Evaluate (optional)

```bash
python -m scripts.evaluate
```

---

## 🌐 NLP Mode (Optional)

| Backend | Description | Requirement |
|---------|-------------|-------------|
| `transformer` | Offline semantic similarity | None |
| `llm` | Google Gemini API | `GEMINI_API_KEY` env var |

```bash
export GEMINI_API_KEY=your_key_here
export NLP_BACKEND=llm
python app.py
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.10 |
| Deep Learning | TensorFlow 2.10, Keras |
| Machine Learning | scikit-learn, XGBoost |
| Web Framework | Flask 3.1 |
| NLP (optional) | sentence-transformers / Google Gemini |

---

## ⚠️ Disclaimer

For **educational purposes only**. Not a substitute for professional medical advice.
