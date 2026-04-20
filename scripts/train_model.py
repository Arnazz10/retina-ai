"""
Diabetic Retinopathy Detection trainer.

The original trainer loaded the entire image dataset into RAM and trained a
fairly heavy 224px residual network. That is fragile on CPU-only machines and
can crash mid-run. This version streams images with tf.data and defaults to a
smaller model profile.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import callbacks, layers, models, regularizers, mixed_precision

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "dr_model.h5"

DEFAULT_IMG_SIZE = 160
DEFAULT_BATCH_SIZE = 8
DEFAULT_EPOCHS = 20
DEFAULT_VARIANT = "small"
NUM_CLASSES = 5
SEED = 42
AUTOTUNE = tf.data.AUTOTUNE

DR_LABELS = {
    0: "No DR",
    1: "Mild DR",
    2: "Moderate DR",
    3: "Severe DR",
    4: "Proliferative DR",
}


def find_image_file(image_dir: Path, id_code: str) -> str | None:
    for ext in (".png", ".jpeg", ".jpg"):
        path = image_dir / f"{id_code}{ext}"
        if path.exists():
            return str(path)
    return None


def load_and_verify_train(data_dir: Path) -> pd.DataFrame:
    csv_path = data_dir / "train.csv"
    image_dir = data_dir / "train_images"

    assert csv_path.exists(), f"train.csv not found at {csv_path}"
    assert image_dir.exists(), f"train_images not found at {image_dir}"

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip().str.lower()

    if "id_code" not in df.columns:
        for alias in ("image", "filename", "image_name"):
            if alias in df.columns:
                df.rename(columns={alias: "id_code"}, inplace=True)
                break
        else:
            raise ValueError(f"Missing image id column in train.csv: {list(df.columns)}")

    if "diagnosis" not in df.columns:
        for alias in ("label", "class", "target", "level"):
            if alias in df.columns:
                df.rename(columns={alias: "diagnosis"}, inplace=True)
                break
        else:
            raise ValueError(f"Missing label column in train.csv: {list(df.columns)}")

    df["id_code"] = (
        df["id_code"]
        .astype(str)
        .str.replace(r"\.(png|jpg|jpeg)$", "", regex=True)
        .str.strip()
    )
    df["filepath"] = df["id_code"].apply(lambda value: find_image_file(image_dir, value))

    missing = df["filepath"].isna()
    if missing.any():
        print(f"Skipping {missing.sum()} missing train images")
        df = df.loc[~missing].copy()

    df["diagnosis"] = pd.to_numeric(df["diagnosis"], errors="coerce")
    valid = df["diagnosis"].isin(range(NUM_CLASSES))
    if (~valid).any():
        print(f"Dropping {(~valid).sum()} rows with invalid labels")
        df = df.loc[valid].copy()

    df["diagnosis"] = df["diagnosis"].astype(int)
    df["label_name"] = df["diagnosis"].map(DR_LABELS)
    df.reset_index(drop=True, inplace=True)

    print("\n" + "=" * 52)
    print("Train label distribution")
    print("=" * 52)
    dist = df["diagnosis"].value_counts().sort_index()
    for grade, count in dist.items():
        pct = count / len(df) * 100
        print(f"Grade {grade} {DR_LABELS[grade]:<20} {count:>5}  {pct:>5.1f}%")
    print(f"Total train samples: {len(df)}")
    print("=" * 52 + "\n")

    return df


def load_test(data_dir: Path) -> pd.DataFrame:
    csv_path = data_dir / "test.csv"
    image_dir = data_dir / "test_images"

    if not csv_path.exists() or not image_dir.exists():
        return pd.DataFrame(columns=["id_code", "filepath"])

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip().str.lower()

    if "id_code" not in df.columns:
        for alias in ("image", "filename", "image_name"):
            if alias in df.columns:
                df.rename(columns={alias: "id_code"}, inplace=True)
                break

    df["id_code"] = (
        df["id_code"]
        .astype(str)
        .str.replace(r"\.(png|jpg|jpeg)$", "", regex=True)
        .str.strip()
    )
    df["filepath"] = df["id_code"].apply(lambda value: find_image_file(image_dir, value))
    df = df.loc[df["filepath"].notna()].copy().reset_index(drop=True)
    print(f"Test samples loaded: {len(df)}")
    return df


def decode_and_resize(path: tf.Tensor, label: tf.Tensor, img_size: int) -> tuple[tf.Tensor, tf.Tensor]:
    image_bytes = tf.io.read_file(path)
    image = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
    image = tf.image.resize(image, [img_size, img_size], method=tf.image.ResizeMethod.BILINEAR)
    image = tf.cast(image, tf.float32) / 255.0
    image.set_shape((img_size, img_size, 3))
    label = tf.cast(label, tf.int32)
    return image, label


def augment(image: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_brightness(image, max_delta=0.12)
    image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image, label


def make_dataset(
    df: pd.DataFrame,
    img_size: int,
    batch_size: int,
    training: bool,
) -> tf.data.Dataset:
    dataset = tf.data.Dataset.from_tensor_slices(
        (df["filepath"].astype(str).to_numpy(), df["diagnosis"].to_numpy(dtype=np.int32))
    )
    if training:
        dataset = dataset.shuffle(len(df), seed=SEED, reshuffle_each_iteration=True)
    dataset = dataset.map(
        lambda path, label: decode_and_resize(path, label, img_size),
        num_parallel_calls=AUTOTUNE,
    )
    if training:
        dataset = dataset.map(augment, num_parallel_calls=AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(1)
    return dataset


def residual_block(x: tf.Tensor, filters: int, dropout: float) -> tf.Tensor:
    reg = regularizers.l2(5e-4)
    shortcut = x
    if shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, padding="same", kernel_regularizer=reg)(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    x = layers.Conv2D(filters, 3, padding="same", kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(filters, 3, padding="same", kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([x, shortcut])
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(dropout)(x)
    return x


def build_small_model(input_shape: tuple[int, int, int]) -> tf.keras.Model:
    reg = regularizers.l2(3e-4)
    inp = layers.Input(shape=input_shape)

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
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.35)(x)
    # Mixed precision requires float32 for the final output layer
    out = layers.Dense(NUM_CLASSES, activation="softmax", dtype="float32")(x)

    model = models.Model(inp, out, name="DR_SmallCNN")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_residual_model(input_shape: tuple[int, int, int]) -> tf.keras.Model:
    inp = layers.Input(shape=input_shape)

    x = layers.Conv2D(32, 7, strides=2, padding="same")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(3, strides=2, padding="same")(x)
    x = residual_block(x, 64, 0.20)
    x = residual_block(x, 128, 0.25)
    x = residual_block(x, 256, 0.30)
    x = residual_block(x, 384, 0.35)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.45)(x)
    # Mixed precision requires float32 for the final output layer
    out = layers.Dense(NUM_CLASSES, activation="softmax", dtype="float32")(x)

    model = models.Model(inp, out, name="DR_ResidualNet")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_model(input_shape: tuple[int, int, int], variant: str) -> tf.keras.Model:
    if variant == "residual":
        return build_residual_model(input_shape)
    return build_small_model(input_shape)


def compile_model(model: tf.keras.Model, variant: str) -> None:
    if variant == "residual":
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
    else:
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )


def get_class_weights(labels: np.ndarray) -> dict[int, float]:
    weights = compute_class_weight("balanced", classes=np.arange(NUM_CLASSES), y=labels)
    return {int(index): float(weight) for index, weight in enumerate(weights)}


def plot_history(history: tf.keras.callbacks.History) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history["accuracy"], label="Train Acc")
    axes[0].plot(history.history["val_accuracy"], label="Val Acc")
    axes[0].set_title("Accuracy")
    axes[0].legend()

    axes[1].plot(history.history["loss"], label="Train Loss")
    axes[1].plot(history.history["val_loss"], label="Val Loss")
    axes[1].set_title("Loss")
    axes[1].legend()

    plt.tight_layout()
    curves_path = MODELS_DIR / "training_curves.png"
    plt.savefig(curves_path)
    plt.close()
    print(f"Training curves saved to {curves_path}")


def predict_test(model: tf.keras.Model, test_df: pd.DataFrame, data_dir: Path, img_size: int) -> None:
    if test_df.empty:
        return

    print("\nRunning inference on test set")
    test_ds = tf.data.Dataset.from_tensor_slices(test_df["filepath"].astype(str).to_numpy())
    test_ds = test_ds.map(
        lambda path: decode_and_resize(path, tf.constant(0), img_size)[0],
        num_parallel_calls=AUTOTUNE,
    ).batch(DEFAULT_BATCH_SIZE)

    probs = model.predict(test_ds, verbose=0)
    preds = np.argmax(probs, axis=1)

    out_df = test_df.copy()
    out_df["diagnosis"] = preds
    out_df["label_name"] = out_df["diagnosis"].map(DR_LABELS)

    out_df[["id_code", "diagnosis"]].to_csv(data_dir / "submission.csv", index=False)
    out_df[["id_code", "diagnosis", "label_name"]].to_csv(
        data_dir / "submission_labeled.csv",
        index=False,
    )
    print(f"Predictions saved to {data_dir / 'submission.csv'}")


def verify_only(data_dir: str) -> None:
    data_dir_path = Path(data_dir)
    print("\nVERIFY MODE\n")
    train_df = load_and_verify_train(data_dir_path)
    test_df = load_test(data_dir_path)
    print(train_df[["id_code", "diagnosis", "label_name", "filepath"]].head(5).to_string(index=False))
    if not test_df.empty:
        print(test_df[["id_code", "filepath"]].head(5).to_string(index=False))
    print("\nVerification complete")


def train(
    data_dir: str,
    epochs: int,
    batch_size: int,
    img_size: int,
    variant: str,
    resume: bool,
) -> tuple[tf.keras.Model, tf.keras.callbacks.History]:
    # Enable mixed precision training
    try:
        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_global_policy(policy)
        print(f"Mixed precision policy set to: {policy.name}")
    except Exception as e:
        print(f"Mixed precision not available: {e}")

    tf.keras.utils.set_random_seed(SEED)
    data_dir_path = Path(data_dir)

    train_df = load_and_verify_train(data_dir_path)
    test_df = load_test(data_dir_path)

    train_split, val_split = train_test_split(
        train_df,
        test_size=0.15,
        random_state=SEED,
        stratify=train_df["diagnosis"],
    )
    train_split = train_split.reset_index(drop=True)
    val_split = val_split.reset_index(drop=True)

    print(f"Train split: {len(train_split)}")
    print(f"Val split:   {len(val_split)}")
    print(f"Variant:     {variant}")
    print(f"Image size:  {img_size}")
    print(f"Batch size:  {batch_size}")
    print(f"Epochs:      {epochs}\n")

    train_ds = make_dataset(train_split, img_size=img_size, batch_size=batch_size, training=True)
    val_ds = make_dataset(val_split, img_size=img_size, batch_size=batch_size, training=False)

    class_weights = get_class_weights(train_split["diagnosis"].to_numpy(dtype=np.int32))
    print("Class weights:", {DR_LABELS[key]: round(value, 2) for key, value in class_weights.items()})

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    if resume and MODEL_PATH.exists():
        print(f"Resuming from checkpoint: {MODEL_PATH}")
        try:
            model = tf.keras.models.load_model(str(MODEL_PATH))
            compile_model(model, variant)
        except Exception as err:
            print(f"Could not load checkpoint, rebuilding model: {err}")
            model = build_model((img_size, img_size, 3), variant)
    else:
        model = build_model((img_size, img_size, 3), variant)
    model.summary()

    cb_list = [
        callbacks.ModelCheckpoint(
            str(MODEL_PATH),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1,
        ),
        callbacks.CSVLogger(str(MODELS_DIR / "history.csv")),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        class_weight=class_weights,
        callbacks=cb_list,
        verbose=2,
    )

    plot_history(history)
    print(f"\nTraining complete. Best model saved to {MODEL_PATH}")

    # --- Quantization and TFLite Export ---
    print("\nStarting model quantization and TFLite export...")
    try:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        # For full integer quantization, a representative dataset is needed.
        # Here we do dynamic range quantization as a starting point.
        tflite_quant_model = converter.convert()
        tflite_path = MODELS_DIR / "dr_model_quantized.tflite"
        tflite_path.write_bytes(tflite_quant_model)
        print(f"✅ Quantized model saved to {tflite_path}")
    except Exception as e:
        print(f"⚠️ Quantization failed: {e}")

    predict_test(model, test_df, data_dir_path, img_size)
    return model, history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DR Detection trainer")
    parser.add_argument("--data_dir", default=str(DATA_DIR), help="Dataset directory")
    parser.add_argument("--verify_only", action="store_true", help="Validate files and labels only")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size")
    parser.add_argument("--img_size", type=int, default=DEFAULT_IMG_SIZE, help="Square image size")
    parser.add_argument(
        "--variant",
        choices=("small", "residual"),
        default=DEFAULT_VARIANT,
        help="Model size profile",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from models/dr_model.h5 if available",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.verify_only:
        verify_only(args.data_dir)
    else:
        train(
            data_dir=args.data_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            img_size=args.img_size,
            variant=args.variant,
            resume=args.resume,
        )
