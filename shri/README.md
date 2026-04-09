# 🔬 RetinaAI — Diabetic Retinopathy Detection

A complete Flask web application for detecting and grading diabetic retinopathy (DR) from retinal fundus photographs using a **custom lightweight CNN built entirely from scratch** — no pretrained weights, no external model downloads.

---

## 📁 Project Structure

```
dr_detection/
├── app.py                  # Flask backend (routes, prediction, model loading)
├── train_model.py          # Full training pipeline
├── requirements.txt        # Python dependencies
├── models/
│   └── dr_model.h5         # Saved after training (auto-loaded on startup)
├── static/
│   ├── css/style.css
│   ├── js/app.js
│   └── uploads/            # Auto-created on first run
└── templates/
    └── index.html
```

---

## 🚀 Quick Start in VS Code

### 1. Create & activate a virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the app
```bash
python app.py
```
Visit **http://localhost:5000** — works immediately in **demo mode** (no training needed).

---

## 🧠 Model Architecture

Custom Lightweight CNN — built from scratch, ~2.1 M parameters, no pretrained weights.

```
Input: 128 × 128 × 3

Block 1 │ Conv(32) → BN → ReLU → Conv(32) → BN → ReLU → MaxPool(2) → Dropout(0.25)
         │                                                          [→ 64 × 64]
Block 2 │ Conv(64) → BN → ReLU → Conv(64) → BN → ReLU → MaxPool(2) → Dropout(0.25)
         │                                                          [→ 32 × 32]
Block 3 │ Conv(128)→ BN → ReLU → Conv(128)→ BN → ReLU → MaxPool(2) → Dropout(0.30)
         │                                                          [→ 16 × 16]
Block 4 │ Conv(256)→ BN → ReLU → GlobalAveragePooling             [→ 256-dim]

Head    │ Dense(256) → BN → ReLU → Dropout(0.50) → Dense(5, softmax)

Output  : [No DR | Mild DR | Moderate DR | Severe DR | Proliferative DR]
```

### Why this architecture works well:
| Technique | Purpose |
|---|---|
| Double conv per block | Richer feature extraction before pooling |
| BatchNorm after every conv | Stable, fast training from scratch |
| GlobalAveragePooling | Spatial invariance, fewer params than Flatten |
| Dropout (0.25 → 0.50) | Progressively stronger regularisation |
| L2 weight decay | Prevents co-adaptation in conv filters |
| Class-weighted loss | Handles severe dataset imbalance |
| 360° rotation augmentation | Retinal images have no fixed orientation |

---

## 🏋️ Training on Real Data

### Step 1 — Download APTOS 2019 dataset
1. Sign in to Kaggle: https://www.kaggle.com/c/aptos2019-blindness-detection/data
2. Download and extract to `./data/`:
```
data/
├── train.csv          ← columns: id_code, diagnosis (0–4)
└── train_images/
    ├── 000c1434d8d7.png
    └── ...
```

### Step 2 — Train
```bash
python train_model.py --data_dir ./data --epochs 40 --batch_size 32
```

Training automatically:
- Splits 85% train / 15% validation (stratified)
- Applies class weighting for imbalanced grades
- Saves the best checkpoint to `models/dr_model.h5` (monitored by val AUC)
- Saves training plots to `plots/training_history.png`

### Step 3 — Restart the server
```bash
python app.py
```
The server auto-loads `models/dr_model.h5` on startup.

---

## 🌐 API Endpoints

| Method | URL | Description |
|---|---|---|
| GET | `/` | Web UI |
| POST | `/api/predict` | Upload image → DR grade + probabilities |
| GET | `/api/model-info` | Architecture details + param count |
| GET | `/api/health` | Health check |

### POST /api/predict — example response
```json
{
  "success": true,
  "predicted_class": "Moderate DR",
  "confidence": 74.3,
  "risk_level": "moderate",
  "color": "#f59e0b",
  "description": "Dot hemorrhages and exudates visible...",
  "recommendation": "Ophthalmologist within 3–6 months.",
  "probabilities": {
    "No DR": 6.1,
    "Mild DR": 9.2,
    "Moderate DR": 74.3,
    "Severe DR": 7.8,
    "Proliferative DR": 2.6
  },
  "image_url": "/static/uploads/abc123_retina.png"
}
```

---

## 🔧 requirements.txt

```
flask>=2.3.0
werkzeug>=2.3.0
pillow>=10.0.0
numpy>=1.24.0
tensorflow>=2.13.0
scikit-learn>=1.3.0
pandas>=2.0.0
matplotlib>=3.7.0
```

---

## ⚠️ Disclaimer

For **educational and research use only**. Not validated for clinical diagnosis. Always consult a qualified ophthalmologist for medical assessment.
