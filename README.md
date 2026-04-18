# RetinaAI — Diabetic Retinopathy Detection

Flask + TensorFlow app for diabetic retinopathy screening from retinal fundus images.

## Project Structure

```text
retina-ai/
├── api/                     # Vercel serverless entrypoint
├── app.py                   # Flask backend app
├── data/                    # Dataset + generated submissions
│   ├── train.csv
│   ├── test.csv
│   ├── train_images/
│   ├── test_images/
│   ├── submission.csv
│   └── submission_labeled.csv
├── logs/                    # Runtime training logs
├── models/                  # Saved model and training artifacts
│   ├── dr_model.h5
│   ├── history.csv
│   └── training_curves.png
├── scripts/
│   └── train_model.py       # Model training pipeline
├── static/
├── templates/
├── requirements.txt
└── vercel.json
```

## Local Setup

1. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Start app

```bash
python app.py
```

Open `http://localhost:5000`.

## Train Model

```bash
python scripts/train_model.py --data_dir ./data --variant small --epochs 20 --batch_size 8
```

The trained checkpoint is saved to `models/dr_model.h5`.

## Notes

- `api/index.py` is used by Vercel and imports `app.py`.
- Large dataset folders are ignored in Git and Vercel deployment.
- If no trained model is found, the app falls back to demo mode.
