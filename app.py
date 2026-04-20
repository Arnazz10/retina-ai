"""
Diabetic Retinopathy Detection — Flask Backend
Model: Custom Lightweight CNN (4 conv blocks, built from scratch)
No pretrained weights. No external model downloads.
"""

import os
import tempfile
import uuid
import csv
import numpy as np
from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
SCRIPTS_DIR = os.path.join(BASE_DIR, 'scripts')
LOGS_DIR = os.path.join(BASE_DIR, 'logs')
TEMPLATE_DIR = os.path.join(BASE_DIR, 'templates')
STATIC_DIR = os.path.join(BASE_DIR, 'static')
UPLOAD_DIR = (
    os.path.join(tempfile.gettempdir(), 'retina_uploads')
    if os.environ.get('VERCEL')
    else os.path.join(STATIC_DIR, 'uploads')
)

app = Flask(__name__, template_folder=TEMPLATE_DIR, static_folder=STATIC_DIR)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['UPLOAD_FOLDER']      = UPLOAD_DIR
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Dataset paths
TRAIN_IMAGES_DIR = os.path.join(DATA_DIR, 'train_images')
TEST_IMAGES_DIR  = os.path.join(DATA_DIR, 'test_images')
TRAIN_CSV = os.path.join(DATA_DIR, 'train.csv')
TEST_CSV  = os.path.join(DATA_DIR, 'test.csv')

os.makedirs(LOGS_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOGS_DIR, 'training.log')

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'dr_model.h5')
HISTORY_PATH = os.path.join(BASE_DIR, 'models', 'history.csv')
IMG_SIZE = 160
BATCH_SIZE = 4
EPOCHS = 10
NUM_CLASSES = 5

DR_CLASSES = ["No DR", "Mild DR", "Moderate DR", "Severe DR", "Proliferative DR"]

DR_DESCRIPTIONS = {
    "No DR":            "No signs of diabetic retinopathy. Continue routine annual eye exams.",
    "Mild DR":          "Microaneurysms only. Early stage — annual follow-up recommended.",
    "Moderate DR":      "Dot hemorrhages and exudates visible. Consult an ophthalmologist soon.",
    "Severe DR":        "Extensive hemorrhages in multiple quadrants. Urgent referral needed.",
    "Proliferative DR": "Neovascularization detected. Immediate treatment required.",
}
DR_RISK = {
    "No DR": "low", "Mild DR": "low",
    "Moderate DR": "moderate", "Severe DR": "high", "Proliferative DR": "critical",
}
DR_COLORS = {
    "No DR": "#22c55e", "Mild DR": "#84cc16",
    "Moderate DR": "#f59e0b", "Severe DR": "#f97316", "Proliferative DR": "#ef4444",
}

 
model_input_size = IMG_SIZE
model_mtime = None


# ── Architecture ──────────────────────────────────────────────────────────────
def build_small_cnn():
    """Smaller default model matching the safer trainer profile."""
    import tensorflow as tf
    from tensorflow.keras import layers, regularizers

    reg = regularizers.l2(3e-4)
    inp = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))

    x = layers.Conv2D(24, 3, padding="same", kernel_regularizer=reg)(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(2)(x)

    x = layers.Conv2D(48, 3, padding="same", kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(2)(x)

    x = layers.Conv2D(96, 3, padding="same", kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(160, 3, padding="same", kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.35)(x)
    out = layers.Dense(NUM_CLASSES, activation='softmax')(x)

    m = tf.keras.Model(inp, out, name='DR_SmallCNN')
    m.compile(optimizer=tf.keras.optimizers.Adam(3e-4), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return m


def build_residual_cnn():
    """Residual CNN kept for compatibility with older checkpoints."""
    import tensorflow as tf
    from tensorflow.keras import layers, regularizers

    reg = regularizers.l2(5e-4)

    def res_block(x, filters, dropout):
        shortcut = x
        if shortcut.shape[-1] != filters:
            shortcut = layers.Conv2D(filters, 1, padding="same", kernel_regularizer=reg)(shortcut)
            shortcut = layers.BatchNormalization()(shortcut)
        x = layers.Conv2D(filters, 3, padding="same", kernel_regularizer=reg)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(filters, 3, padding="same", kernel_regularizer=reg)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Add()([x, shortcut])
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D(2)(x)
        x = layers.Dropout(dropout)(x)
        return x

    inp = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = layers.Conv2D(32, 7, strides=2, padding="same")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

    x = res_block(x, 64,  0.2)
    x = res_block(x, 128, 0.25)
    x = res_block(x, 256, 0.3)
    x = res_block(x, 512, 0.4)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.50)(x)
    out = layers.Dense(5, activation='softmax')(x)

    m = tf.keras.Model(inp, out, name='DR_ResidualNet')
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
    return m


# ── Model loading ─────────────────────────────────────────────────────────────
 
    global model, model_input_size, model_mtime
    try:
        import tensorflow as tf
        if os.path.exists(MODEL_PATH):
            model = tf.keras.models.load_model(MODEL_PATH)
            input_shape = getattr(model, "input_shape", None)
            if input_shape and len(input_shape) >= 3 and input_shape[1]:
                model_input_size = int(input_shape[1])
            model_mtime = os.path.getmtime(MODEL_PATH)
            print(f"✅ Loaded trained model  →  {MODEL_PATH}")
            print(f"   Params: {model.count_params():,}")
            print(f"   Input size: {model_input_size}×{model_input_size}")
        else:
            print("⚙️  No saved model found — building SmallCNN structure (untrained)")
            print("   Run: python scripts/train_model.py")
            model = build_small_cnn()
            model_mtime = None
            print(f"   Params: {model.count_params():,}")
    except ImportError:
        print("⚠️  TensorFlow not installed — demo mode active")
        model = None
        model_mtime = None


def ensure_latest_model():
    global model_mtime
    if not os.path.exists(MODEL_PATH):
        return
    current_mtime = os.path.getmtime(MODEL_PATH)
    if model is None or model_mtime is None or current_mtime > model_mtime:
 


# ── Preprocessing ─────────────────────────────────────────────────────────────
def preprocess_image(path: str) -> np.ndarray:
    img = Image.open(path).convert('RGB').resize((model_input_size, model_input_size), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, 0)


# ── Prediction ────────────────────────────────────────────────────────────────
def predict_dr(path: str) -> dict:
    global model
    ensure_latest_model()
    if model is None:
        return _demo(path)
    try:
        probs = model.predict(preprocess_image(path), verbose=0)[0]
        idx   = int(np.argmax(probs))
        cls   = DR_CLASSES[idx]
        return {
            "success":         True,
            "predicted_class": cls,
            "confidence":      round(float(probs[idx]) * 100, 2),
            "risk_level":      DR_RISK[cls],
            "description":     DR_DESCRIPTIONS[cls],
            "color":           DR_COLORS[cls],
            "probabilities":   {DR_CLASSES[i]: round(float(probs[i]) * 100, 2) for i in range(5)},
            "recommendation":  _recommendation(cls),
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def _demo(path: str) -> dict:
    import hashlib
    h   = int(hashlib.md5(os.path.basename(path).encode()).hexdigest(), 16)
    idx = h % 5
    rng = np.random.default_rng(h % (2**32))
    p   = rng.dirichlet(np.ones(5) * 0.4)
    p[idx] = max(float(p[idx]) + 0.45, 0.55)
    p /= p.sum()
    cls = DR_CLASSES[idx]
    return {
        "success": True, "predicted_class": cls,
        "confidence": round(float(p[idx]) * 100, 2),
        "risk_level": DR_RISK[cls], "description": DR_DESCRIPTIONS[cls],
        "color": DR_COLORS[cls],
        "probabilities": {DR_CLASSES[i]: round(float(p[i]) * 100, 2) for i in range(5)},
        "recommendation": _recommendation(cls), "demo_mode": True,
    }


def _recommendation(cls: str) -> str:
    return {
        "No DR":            "Maintain glycemic control. Annual dilated eye exam.",
        "Mild DR":          "Optimise blood sugar & BP. Follow-up in 9–12 months.",
        "Moderate DR":      "Ophthalmologist within 3–6 months. Intensify diabetes management.",
        "Severe DR":        "Urgent ophthalmology referral within 1 month.",
        "Proliferative DR": "Emergency ophthalmology visit. Anti-VEGF or vitrectomy may be required.",
    }.get(cls, "Consult a specialist.")


def _allowed(filename: str) -> bool:
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


# ── Background Training ──────────────────────────────────────────────────────
import subprocess
import threading
import time

training_process = None

def run_training():
    global training_process
    with open(LOG_FILE, 'w') as f:
        f.write(f"--- Training started at {time.ctime()} ---\n")
        f.flush()
        import sys
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        training_process = subprocess.Popen(
            [
                sys.executable,
                os.path.join(SCRIPTS_DIR, 'train_model.py'),
                '--data_dir', DATA_DIR,
                '--variant', 'small',
                '--resume',
                '--img_size', str(IMG_SIZE),
                '--batch_size', str(BATCH_SIZE),
                '--epochs', str(EPOCHS),
            ],
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )
        training_process.wait()
        f.write(f"\n--- Training finished at {time.ctime()} ---\n")
    if training_process.returncode == 0:
 



# ── Routes ────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file uploaded'}), 400
    f = request.files['file']
    if not f.filename:
        return jsonify({'success': False, 'error': 'No file selected'}), 400
    if not _allowed(f.filename):
        return jsonify({'success': False, 'error': 'Use PNG / JPG / BMP / TIFF'}), 400
    try:
        fname = f"{uuid.uuid4().hex}_{secure_filename(f.filename)}"
        fpath = os.path.join(app.config['UPLOAD_FOLDER'], fname)
        f.save(fpath)
        result = predict_dr(fpath)
        result['image_url'] = f"/static/uploads/{fname}"
        return jsonify(result)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/model-info')
def model_info():
    params = model.count_params() if model else 0
    return jsonify({
        'model_loaded':  model is not None,
        'architecture':  getattr(model, 'name', 'DR_SmallCNN') if model else 'DR_SmallCNN',
        'input_size':    f'{model_input_size}×{model_input_size}',
        'total_params':  f'{params:,}',
        'classes':       DR_CLASSES,
        'framework':     'TensorFlow / Keras',
    })


@app.route('/api/training-summary')
def training_summary():
    history_rows = []
    if os.path.exists(HISTORY_PATH):
        with open(HISTORY_PATH, 'r', newline='') as csv_file:
            reader = csv.DictReader(csv_file)
            history_rows = list(reader)

    epochs_completed = len(history_rows)
    best_val_accuracy = None
    best_epoch = None
    last_train_accuracy = None
    last_val_accuracy = None

    if history_rows:
        val_scores = []
        for index, row in enumerate(history_rows):
            try:
                val_scores.append((float(row.get('val_accuracy', 0.0)), index + 1))
            except (TypeError, ValueError):
                continue

        if val_scores:
            best_val_accuracy, best_epoch = max(val_scores, key=lambda item: item[0])

        try:
            last_train_accuracy = float(history_rows[-1].get('accuracy', 0.0))
            last_val_accuracy = float(history_rows[-1].get('val_accuracy', 0.0))
        except (TypeError, ValueError):
            last_train_accuracy = None
            last_val_accuracy = None

    progress_pct = 0
    if EPOCHS > 0 and epochs_completed > 0:
        progress_pct = min(100, round((epochs_completed / EPOCHS) * 100))

    return jsonify({
        'success': True,
        'configured_epochs': EPOCHS,
        'configured_batch_size': BATCH_SIZE,
        'configured_img_size': IMG_SIZE,
        'epochs_completed': epochs_completed,
        'progress_percent': progress_pct,
        'model_file_exists': os.path.exists(MODEL_PATH),
        'best_val_accuracy': round(best_val_accuracy * 100, 2) if best_val_accuracy is not None else None,
        'best_epoch': best_epoch,
        'last_train_accuracy': round(last_train_accuracy * 100, 2) if last_train_accuracy is not None else None,
        'last_val_accuracy': round(last_val_accuracy * 100, 2) if last_val_accuracy is not None else None,
        'results_graph_url': '/static/training_results.png' if os.path.exists(os.path.join(STATIC_DIR, 'training_results.png')) else None
    })


@app.route('/api/health')
def health():
    return jsonify({'status': 'ok', 'model_ready': model is not None})


@app.route('/static/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/api/dataset')
def get_dataset():
    import pandas as pd
    page = int(request.args.get('page', 1))
    limit = int(request.args.get('limit', 20))
    split = request.args.get('split', 'train') # 'train' or 'test'
    
    csv_path = TRAIN_CSV if split == 'train' else TEST_CSV
    subset_path = os.path.join(DATA_DIR, f'{split}_labeled_subset.csv')
    
    # Use labeled subset if it exists and we are on the first page
    current_csv = subset_path if os.path.exists(subset_path) and page == 1 else csv_path
    
    img_dir = TRAIN_IMAGES_DIR if split == 'train' else TEST_IMAGES_DIR
    
    if not os.path.exists(current_csv):
        return jsonify({'success': False, 'error': f'{split}.csv not found'}), 404
        
    df = pd.read_csv(current_csv)
    total = len(df)
    start = (page - 1) * limit
    end = start + limit
    
    subset = df.iloc[start:end].copy()
    
    results = []
    for _, row in subset.iterrows():
        id_code = str(row['id_code'])
        # Try finding image
        img_name = None
        for ext in ['.png', '.jpg', '.jpeg']:
            if os.path.exists(os.path.join(img_dir, id_code + ext)):
                img_name = id_code + ext
                break
        
        results.append({
            'id_code': id_code,
            'diagnosis': int(row['diagnosis']) if 'diagnosis' in df.columns else (int(row['predicted_diagnosis']) if 'predicted_diagnosis' in df.columns else None),
            'label': row['predicted_label'] if 'predicted_label' in df.columns else (DR_CLASSES[int(row['diagnosis'])] if 'diagnosis' in df.columns else None),
            'confidence': row['confidence'] if 'confidence' in df.columns else None,
            'image_url': f'/api/images/{split}/{img_name}' if img_name else None
        })
        
    return jsonify({
        'success': True,
        'data': results,
        'total': total,
        'page': page,
        'limit': limit
    })


@app.route('/api/images/<split>/<filename>')
def serve_dataset_image(split, filename):
    directory = TRAIN_IMAGES_DIR if split == 'train' else TEST_IMAGES_DIR
    return send_from_directory(directory, filename)


@app.route('/api/train', methods=['POST'])
def start_train():
    global training_process
    if training_process and training_process.poll() is None:
        return jsonify({'success': False, 'error': 'Training already in progress'})
    
    thread = threading.Thread(target=run_training)
    thread.start()
    return jsonify({'success': True, 'message': 'Training started'})


@app.route('/api/train-status')
def train_status():
    global training_process
    status = "idle"
    if training_process:
        if training_process.poll() is None:
            status = "running"
        else:
            status = "finished" if training_process.returncode == 0 else "failed"
            
    return jsonify({
        'status': status,
        'log_exists': os.path.exists(LOG_FILE)
    })


@app.route('/api/train-logs')
def train_logs():
    def generate():
        if not os.path.exists(LOG_FILE):
            yield "data: No logs yet\n\n"
            return
            
        with open(LOG_FILE, 'r') as f:
            for line in f:
                clean = line.replace('\x00', '').rstrip('\n')
                yield f"data: {clean}\n\n"
            
            while training_process and training_process.poll() is None:
                line = f.readline()
                if line:
                    clean = line.replace('\x00', '').rstrip('\n')
                    yield f"data: {clean}\n\n"
                else:
                    time.sleep(0.5)
            
            line = f.readline()
            if line:
                clean = line.replace('\x00', '').rstrip('\n')
                yield f"data: {clean}\n\n"

    from flask import Response
    return Response(generate(), mimetype='text/event-stream')


# ── Entry ─────────────────────────────────────────────────────────────────────

load_model()

if __name__ == '__main__':
    app.run(
        debug=os.environ.get('FLASK_DEBUG', '0') == '1',
        host=os.environ.get('FLASK_HOST', '127.0.0.1'),
        port=int(os.environ.get('PORT', '5000')),
    )
