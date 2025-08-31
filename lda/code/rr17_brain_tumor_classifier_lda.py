from __future__ import annotations
import zipfile
import argparse
import csv
import datetime
import json
import logging
import os
import random
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib
# Use Agg backend for headless operation (no display needed)
matplotlib.use('Agg')  # type: ignore
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

from sklearn.metrics import (
    accuracy_score,
    precision_recall_curve,
    roc_curve,
    auc,
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks

try:
    import kaggle
except ImportError:
    kaggle = None  # type: ignore

# add near other optional imports
try:
    import gradio as gr
except ImportError:
    gr = None  # type: ignore

def setup_logging(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / 'log.txt'
    logger = logging.getLogger('rr17_logger')
    logger.setLevel(logging.INFO)
    # Remove default handlers to avoid duplicate logs when run multiple times
    if logger.handlers:
        for h in logger.handlers:
            logger.removeHandler(h)
    formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')
    # File handler
    fh = logging.FileHandler(log_file, mode='w')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.info('Logging initialised. Log file at %s', log_file)
    return logger


def set_seeds(seed: int) -> None:
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    try:
        tf.keras.utils.set_random_seed(seed)
    except Exception:
        pass
    # Enable deterministic operations if available
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        # On some TF versions this may not be present
        pass


def snapshot_environment(output_dir: Path, logger: logging.Logger) -> None:
    req_file = output_dir / 'requirements.txt'
    try:
        import subprocess
        with open(req_file, 'w') as f:
            subprocess.run([sys.executable, '-m', 'pip', 'freeze'], stdout=f, check=True)
        logger.info('Saved pip freeze to %s', req_file)
    except Exception as e:
        logger.error('Failed to save pip requirements: %s', e)
    # Record runtime information
    runtime_info = {
        'python_version': sys.version,
        'tensorflow_version': tf.__version__,
        'numpy_version': np.__version__,
        'pandas_version': pd.__version__,
    }
    try:
        gpus = tf.config.list_physical_devices('GPU')
        runtime_info['gpus'] = [gpu.name for gpu in gpus]
    except Exception:
        runtime_info['gpus'] = []
    info_file = output_dir / 'runtime_info.json'
    with open(info_file, 'w') as f:
        json.dump(runtime_info, f, indent=2)
    logger.info('Runtime information saved to %s', info_file)


def download_kaggle_dataset(dataset_slug: str, download_dir: Path, logger: logging.Logger) -> Path:
    if kaggle is None:
        logger.warning('Kaggle package is not installed. Cannot download dataset.')
        raise RuntimeError('Kaggle not available')
    # Ensure Kaggle credentials exist
    kaggle_dir = Path.home() / '.kaggle'
    token = kaggle_dir / 'kaggle.json'
    if not token.exists() and not (
        os.environ.get('KAGGLE_USERNAME') and os.environ.get('KAGGLE_KEY')):
        logger.error('Kaggle credentials not found. Please place kaggle.json in ~/.kaggle or set KAGGLE_USERNAME/KAGGLE_KEY.')
        raise RuntimeError('Missing Kaggle credentials')
    download_dir.mkdir(parents=True, exist_ok=True)
    zip_name = dataset_slug.split('/')[-1] + '.zip'
    zip_path = download_dir / zip_name
    if not zip_path.exists():
        logger.info('Downloading Kaggle dataset %s ...', dataset_slug)
        kaggle.api.dataset_download_files(dataset_slug, path=str(download_dir), quiet=False, force=True)
    else:
        logger.info('Dataset archive already present at %s', zip_path)
    extract_dir = download_dir / dataset_slug.split('/')[-1]
    if not extract_dir.exists():
        logger.info('Extracting %s to %s ...', zip_path, extract_dir)
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(extract_dir)
        logger.info('Extraction complete.')
    else:
        logger.info('Dataset already extracted to %s', extract_dir)
    return extract_dir


def find_image_files(base_dir: Path, logger: Optional[logging.Logger] = None) -> Tuple[List[str], List[int], List[str]]:
    if not base_dir.exists():
        raise FileNotFoundError(f'Dataset directory {base_dir} does not exist.')
    class_names = sorted([d.name for d in base_dir.iterdir() if d.is_dir()])
    if logger:
        logger.info('Found classes: %s', class_names)
    files: List[str] = []
    labels: List[int] = []
    for idx, cls in enumerate(class_names):
        cls_dir = base_dir / cls
        for ext in ('*.jpg', '*.jpeg', '*.png', '*.bmp'):  # support common formats
            for p in cls_dir.rglob(ext):
                files.append(str(p))
                labels.append(idx)
    if logger:
        logger.info('Collected %d images.', len(files))
    return files, labels, class_names


def load_image(path: str, img_size: Tuple[int, int]) -> np.ndarray:
    img = tf.keras.preprocessing.image.load_img(path, target_size=img_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = img_array / 255.0
    return img_array.astype(np.float32)


def build_rr17_model(input_shape: Tuple[int, int, int], num_classes: int, learning_rate: float = 1e-4,
                     dropout_rate: float = 0.5) -> models.Model:
    inputs = layers.Input(shape=input_shape, name="input_layer")
    x = inputs
    # Convolutional Block 1
    x = layers.Conv2D(32, (3, 3), padding="same", kernel_initializer="he_normal")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(32, (3, 3), padding="same", kernel_initializer="he_normal")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)
    # Convolutional Block 2
    x = layers.Conv2D(64, (3, 3), padding="same", kernel_initializer="he_normal")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(64, (3, 3), padding="same", kernel_initializer="he_normal")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)
    # Convolutional Block 3
    x = layers.Conv2D(128, (3, 3), padding="same", kernel_initializer="he_normal")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(128, (3, 3), padding="same", kernel_initializer="he_normal")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)
    # Convolutional Block 4
    x = layers.Conv2D(256, (3, 3), padding="same", kernel_initializer="he_normal")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(256, (3, 3), padding="same", kernel_initializer="he_normal")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)
    # Flatten and dense layers
    x = layers.Flatten()(x)
    x = layers.Dense(512, kernel_initializer="he_normal")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = models.Model(inputs=inputs, outputs=outputs, name="RR17")
    optimizer = optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def compute_ece(y_true: np.ndarray, y_pred: np.ndarray, n_bins: int = 10) -> Tuple[float, List[float], List[float], List[float]]:
    # Convert one‑hot to class index
    true_labels = np.argmax(y_true, axis=1)
    confidences = np.max(y_pred, axis=1)
    predictions = np.argmax(y_pred, axis=1)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(confidences, bin_edges[:-1], right=True) - 1
    bin_acc: List[float] = []
    bin_conf: List[float] = []
    bin_count: List[float] = []
    ece = 0.0
    for i in range(n_bins):
        idx = bin_indices == i
        if np.sum(idx) > 0:
            acc = np.mean(true_labels[idx] == predictions[idx])
            conf = np.mean(confidences[idx])
            prop = np.mean(idx)
            ece += np.abs(acc - conf) * prop
            bin_acc.append(acc)
            bin_conf.append(conf)
            bin_count.append(prop)
        else:
            bin_acc.append(0.0)
            bin_conf.append(0.0)
            bin_count.append(0.0)
    return ece, bin_acc, bin_conf, bin_count


def create_output_dirs(root: Path) -> Tuple[Path, Path, Path]:
    root.mkdir(parents=True, exist_ok=True)
    plots_dir = root / 'plots'
    tables_dir = root / 'tables'
    plots_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    return root, plots_dir, tables_dir


def save_dataframe(df: pd.DataFrame, csv_path: Path, logger: Optional[logging.Logger] = None) -> None:
    df.to_csv(csv_path, index=False)
    if logger:
        logger.info('Saved table to %s', csv_path)
    # Print to console in a friendly way
    with pd.option_context('display.max_rows', 10, 'display.max_columns', None):
        print(df)


def plot_and_save(fig: plt.Figure, save_path: Path, logger: Optional[logging.Logger] = None) -> None:
    fig.savefig(save_path, bbox_inches='tight')
    if logger:
        logger.info('Saved plot to %s', save_path)
    plt.close(fig)


def last_conv_layer(model: tf.keras.Model) -> Optional[tf.keras.layers.Layer]:
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer
    return None

def gradcam_heatmap(img_array: np.ndarray, model: tf.keras.Model) -> Optional[np.ndarray]:
    lc = last_conv_layer(model)
    if lc is None:
        return None
    gm = tf.keras.Model(model.inputs, [lc.output, model.output])
    with tf.GradientTape() as tape:
        conv_out, preds = gm(tf.expand_dims(img_array, 0))
        top = tf.argmax(preds[0])
        top_score = preds[:, top]
    grads = tape.gradient(top_score, conv_out)[0]          # (H,W,C)
    conv = conv_out[0]                                     # (H,W,C)
    weights = tf.reduce_mean(grads, axis=(0, 1))           # (C,)
    cam = tf.tensordot(conv, weights, axes=([2], [0]))     # (H,W)
    cam = tf.maximum(cam, 0)
    cam = cam / (tf.reduce_max(cam) + 1e-8)
    return cam.numpy()

def _load_best_checkpoint(output_root: Path) -> Path:
    ckpts = sorted(output_root.glob('fold*_best_model.h5'))
    if not ckpts:
        raise FileNotFoundError("No checkpoints found in outputs/")
    return ckpts[0]

def build_predict_fn(model: tf.keras.Model, class_names: List[str], img_size: Tuple[int, int]):
    def _preprocess(img: np.ndarray) -> np.ndarray:
        # img: HWC uint8 -> float32 [0,1], resize
        img_tf = tf.convert_to_tensor(img)
        img_tf = tf.image.resize(img_tf, img_size)
        img_tf = tf.cast(img_tf, tf.float32) / 255.0
        return img_tf.numpy()

    def _predict(image: np.ndarray, explain: bool):
        if image is None:
            return {cn: 0.0 for cn in class_names}, None
        x = _preprocess(image)
        probs = model.predict(x[None, ...], verbose=0)[0]
        # label dict for nice probability bars
        label_dict = {cn: float(probs[i]) for i, cn in enumerate(class_names)}
        cam_img = None
        if explain:
            heat = gradcam_heatmap(x, model)
            if heat is not None:
                heat_resized = tf.image.resize(heat[..., None], img_size).numpy().squeeze()
                heat_resized = (heat_resized - heat_resized.min()) / (heat_resized.max() - heat_resized.min() + 1e-8)
                # overlay on original resized image
                base = x
                cmap = plt.cm.jet((heat_resized * 255).astype(np.uint8))[:, :, :3]
                blend = (0.6 * base + 0.4 * cmap)
                cam_img = (np.clip(blend, 0, 1) * 255).astype(np.uint8)
        return label_dict, cam_img
    return _predict

def launch_gradio(output_root: Path, class_names: List[str], img_size: Tuple[int, int]):
    if gr is None:
        print("⚠️ Gradio not installed. Run: pip install gradio")
        return
    model_path = _load_best_checkpoint(output_root)
    model = tf.keras.models.load_model(model_path)
    predict_fn = build_predict_fn(model, class_names, img_size)

    with gr.Blocks(theme=gr.themes.Soft(primary_hue="indigo")) as demo:
        gr.Markdown("## 🧠 RR17 Brain Tumor Classifier — Demo")
        gr.Markdown("Upload an MRI image and (optionally) show Grad-CAM explanation.")
        with gr.Row():
            with gr.Column(scale=1):
                inp_img = gr.Image(type="numpy", label="MRI Image (RGB or Grayscale)", height=320)
                inp_cam = gr.Checkbox(value=True, label="Show Grad-CAM")
                btn = gr.Button("Predict", variant="primary")
            with gr.Column(scale=1):
                out_label = gr.Label(num_top_classes=len(class_names), label="Predicted probabilities")
                out_cam = gr.Image(label="Grad-CAM", height=320)

        btn.click(fn=predict_fn, inputs=[inp_img, inp_cam], outputs=[out_label, out_cam])

    # share=True for public link (Colab-friendly)
    demo.launch(share=True)

@dataclass
class DatasetInfo:
    files: List[str]
    labels: List[int]
    class_names: List[str]


def create_tf_dataset(file_paths: List[str], labels: List[int], img_size: Tuple[int, int], batch_size: int,
                      shuffle: bool = True) -> tf.data.Dataset:
    num_classes = len(set(labels))
    def _load(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_image(img, channels=3, expand_animations=False)
        img = tf.image.resize(img, img_size)
        img = tf.cast(img, tf.float32) / 255.0
        return img, tf.one_hot(label, num_classes)
    ds = tf.data.Dataset.from_tensor_slices((file_paths, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(file_paths), seed=42, reshuffle_each_iteration=True)
    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


@dataclass
class FoldResults:
    fold_index: int
    history: Dict[str, List[float]]
    y_true: np.ndarray
    y_pred_probs: np.ndarray
    y_pred_labels: np.ndarray
    metrics: Dict[str, float] = field(default_factory=dict)
    per_class_metrics: pd.DataFrame = field(default_factory=pd.DataFrame)
    classification_report_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    confusion_matrix: np.ndarray = field(default_factory=lambda: np.array([]))
    ece: float = 0.0
    calibration_bins: Optional[Dict[str, Any]] = None
    threshold_metrics: Optional[pd.DataFrame] = None
    inference_time: Optional[float] = None
    misclassified_df: Optional[pd.DataFrame] = None


def train_and_evaluate_one_fold(fold_idx: int, train_idx: np.ndarray, val_idx: np.ndarray, test_indices: Optional[np.ndarray],
                                dataset: DatasetInfo, img_size: Tuple[int, int], batch_size: int, epochs: int,
                                logger: logging.Logger, output_dirs: Tuple[Path, Path, Path],
                                quick: bool = False, learning_rate: float = 1e-4, dropout_rate: float = 0.5) -> FoldResults:
    output_root, plots_dir, tables_dir = output_dirs
    fold_tag = f'fold{fold_idx + 1}'
    logger.info('--- Starting fold %d ---', fold_idx + 1)
    # Prepare datasets for this fold
    # Optionally shrink dataset in quick mode
    train_indices = train_idx.copy()
    val_indices = val_idx.copy()
    if quick:
        # Use only a fraction of data for quick testing
        max_samples = max(10, int(0.2 * len(train_indices)))
        train_indices = train_indices[:max_samples]
        val_indices = val_indices[:max(10, int(0.2 * len(val_indices)))]
        logger.info('Quick mode: using %d training and %d validation samples', len(train_indices), len(val_indices))
    train_files = [dataset.files[i] for i in train_indices]
    train_labels = [dataset.labels[i] for i in train_indices]
    val_files = [dataset.files[i] for i in val_indices]
    val_labels = [dataset.labels[i] for i in val_indices]
    train_ds = create_tf_dataset(train_files, train_labels, img_size, batch_size, shuffle=True)
    val_ds = create_tf_dataset(val_files, val_labels, img_size, batch_size, shuffle=False)
    # Optionally prepare hold‑out test dataset for this fold
    test_ds = None
    test_files: Optional[List[str]] = None
    test_labels: Optional[List[int]] = None
    if test_indices is not None:
        test_files = [dataset.files[i] for i in test_indices]
        test_labels = [dataset.labels[i] for i in test_indices]
        test_ds = create_tf_dataset(test_files, test_labels, img_size, batch_size, shuffle=False)
    # Build a new model
    input_shape = img_size + (3,)
    model = build_rr17_model(input_shape, num_classes=len(dataset.class_names), learning_rate=learning_rate,
                             dropout_rate=dropout_rate)
    # Early stopping and checkpointing
    checkpoint_path = output_root / f'{fold_tag}_best_model.h5'
    es = callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    ckpt = callbacks.ModelCheckpoint(filepath=str(checkpoint_path), monitor='val_accuracy',
                                     save_best_only=True, save_weights_only=False, verbose=0)
    # Train
    start_time = time.time()
    history = model.fit(train_ds, epochs=epochs, validation_data=val_ds, callbacks=[es, ckpt], verbose=0)
    train_time = time.time() - start_time
    logger.info('Fold %d: training finished in %.2f seconds over %d epochs', fold_idx + 1, train_time, len(history.history['loss']))
    # Evaluate on validation set
    val_true = np.concatenate([y.numpy() for _, y in val_ds], axis=0)
    val_true_labels = np.argmax(val_true, axis=1)
    val_pred_probs = model.predict(val_ds, verbose=0)
    val_pred_labels = np.argmax(val_pred_probs, axis=1)
    # Optionally evaluate on test set
    test_true_labels: Optional[np.ndarray] = None
    test_pred_probs: Optional[np.ndarray] = None
    test_pred_labels: Optional[np.ndarray] = None
    if test_ds is not None:
        tt = np.concatenate([y.numpy() for _, y in test_ds], axis=0)
        test_true_labels = np.argmax(tt, axis=1)
        test_pred_probs = model.predict(test_ds, verbose=0)
        test_pred_labels = np.argmax(test_pred_probs, axis=1)
    # Collect results
    fold_results = FoldResults(
        fold_index=fold_idx,
        history=history.history,
        y_true=val_true,
        y_pred_probs=val_pred_probs,
        y_pred_labels=val_pred_labels,
    )
    # Compute metrics
    fold_results.metrics['accuracy'] = float(accuracy_score(val_true_labels, val_pred_labels))
    fold_results.metrics['precision'] = float(precision_score(val_true_labels, val_pred_labels, average='weighted', zero_division=0))
    fold_results.metrics['recall'] = float(recall_score(val_true_labels, val_pred_labels, average='weighted', zero_division=0))
    fold_results.metrics['f1'] = float(f1_score(val_true_labels, val_pred_labels, average='weighted', zero_division=0))
    fold_results.metrics['roc_auc_macro'] = float(auc(*roc_curve(val_true.ravel(), val_pred_probs.ravel())[:2]))
    fold_results.metrics['pr_auc_macro'] = float(average_precision_score(val_true, val_pred_probs, average='macro'))
    # Classification report and per class metrics
    cls_report = classification_report(val_true_labels, val_pred_labels, target_names=dataset.class_names, output_dict=True)
    fold_results.classification_report_df = pd.DataFrame(cls_report).transpose()
    # Confusion matrix
    cm = confusion_matrix(val_true_labels, val_pred_labels)
    fold_results.confusion_matrix = cm
    # Per class metrics table
    per_class = []
    for idx, cls_name in enumerate(dataset.class_names):
        true_idx = val_true_labels == idx
        acc = np.mean(val_pred_labels[true_idx] == idx) if np.sum(true_idx) > 0 else 0.0
        per_class.append({'Class': cls_name, 'Accuracy': acc})
    fold_results.per_class_metrics = pd.DataFrame(per_class)
    # Calibration/ECE
    ece, bin_acc, bin_conf, bin_count = compute_ece(val_true, val_pred_probs, n_bins=10)
    fold_results.ece = ece
    fold_results.calibration_bins = {
        'bin_acc': bin_acc,
        'bin_conf': bin_conf,
        'bin_count': bin_count,
    }
    # Threshold sweep metrics (F1/precision/recall vs threshold) on validation set
    thresholds = np.linspace(0.0, 1.0, 51)
    thr_data = []
    for thr in thresholds:
        preds_thr = (val_pred_probs >= thr).astype(int)
        # Convert predictions to class labels by majority if at least one positive; else argmax
        preds_labels_thr = []
        for row in preds_thr:
            if row.sum() > 0:
                preds_labels_thr.append(np.argmax(row))
            else:
                preds_labels_thr.append(np.argmax(row))
        preds_labels_thr = np.array(preds_labels_thr)
        thr_data.append({
            'threshold': thr,
            'accuracy': accuracy_score(val_true_labels, preds_labels_thr),
            'precision': precision_score(val_true_labels, preds_labels_thr, average='weighted', zero_division=0),
            'recall': recall_score(val_true_labels, preds_labels_thr, average='weighted', zero_division=0),
            'f1': f1_score(val_true_labels, preds_labels_thr, average='weighted', zero_division=0),
        })
    fold_results.threshold_metrics = pd.DataFrame(thr_data)
    # Measure inference time (average time per batch on validation set)
    start = time.time()
    _ = model.predict(val_ds.take(5), verbose=0)
    elapsed = time.time() - start
    num_images = sum(batch[0].shape[0] for batch in val_ds.take(5))
    fold_results.inference_time = float(elapsed / max(num_images, 1))
    # Misclassified samples: capture top examples for inspection
    mis_idx = np.where(val_true_labels != val_pred_labels)[0]
    mis_paths: List[str] = []
    mis_true: List[str] = []
    mis_pred: List[str] = []
    mis_conf: List[float] = []
    for i in mis_idx:
        path = val_files[i] if i < len(val_files) else ''
        mis_paths.append(path)
        mis_true.append(dataset.class_names[val_true_labels[i]])
        mis_pred.append(dataset.class_names[val_pred_labels[i]])
        mis_conf.append(float(val_pred_probs[i][val_pred_labels[i]]))
    mis_df = pd.DataFrame({
        'Image Path': mis_paths,
        'True Label': mis_true,
        'Predicted Label': mis_pred,
        'Prediction Confidence': mis_conf,
    })
    fold_results.misclassified_df = mis_df
    # Save metrics tables
    # Classification report
    save_dataframe(fold_results.classification_report_df, tables_dir / f'{fold_tag}_classification_report.csv', logger)
    # Per class metrics
    save_dataframe(fold_results.per_class_metrics, tables_dir / f'{fold_tag}_per_class_metrics.csv', logger)
    # Confusion matrix numeric table
    cm_df = pd.DataFrame(cm, index=dataset.class_names, columns=dataset.class_names)
    save_dataframe(cm_df.reset_index().rename(columns={'index': 'True\Pred'}), tables_dir / f'{fold_tag}_confusion_matrix.csv', logger)
    # Threshold metrics table
    save_dataframe(fold_results.threshold_metrics, tables_dir / f'{fold_tag}_threshold_metrics.csv', logger)
    # Misclassified samples table
    save_dataframe(mis_df, tables_dir / f'{fold_tag}_misclassified.csv', logger)
    # Inference time table
    inf_df = pd.DataFrame({'Average Inference Time (s)': [fold_results.inference_time]})
    save_dataframe(inf_df, tables_dir / f'{fold_tag}_inference_time.csv', logger)
    # Calibration bin table
    cal_df = pd.DataFrame({
        'Bin': list(range(1, 11)),
        'Bin Accuracy': fold_results.calibration_bins['bin_acc'],
        'Bin Confidence': fold_results.calibration_bins['bin_conf'],
        'Bin Proportion': fold_results.calibration_bins['bin_count'],
    })
    save_dataframe(cal_df, tables_dir / f'{fold_tag}_calibration_bins.csv', logger)
    # Save plots
    # Training history plots
    def plot_history(metric_name: str):
        fig = plt.figure(figsize=(6, 4))
        plt.plot(history.history[metric_name], label=f'Train {metric_name}')
        plt.plot(history.history[f'val_{metric_name}'], label=f'Val {metric_name}')
        plt.xlabel('Epoch')
        plt.ylabel(metric_name.capitalize())
        plt.title(f'{metric_name.capitalize()} vs Epochs – {fold_tag}')
        plt.legend()
        plt.grid(True)
        return fig
    acc_fig = plot_history('accuracy')
    plot_and_save(acc_fig, plots_dir / f'{fold_tag}_accuracy_vs_epoch.png', logger)
    loss_fig = plot_history('loss')
    plot_and_save(loss_fig, plots_dir / f'{fold_tag}_loss_vs_epoch.png', logger)
    # Confusion matrix heatmap
    fig_cm = plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=dataset.class_names, yticklabels=dataset.class_names, cmap='Blues')
    plt.title(f'Confusion Matrix – {fold_tag}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plot_and_save(fig_cm, plots_dir / f'{fold_tag}_confusion_matrix.png', logger)
    # Per class accuracy bar plot
    fig_pc = plt.figure(figsize=(6, 4))
    sns.barplot(x='Class', y='Accuracy', data=fold_results.per_class_metrics)
    plt.title(f'Per‑Class Accuracy – {fold_tag}')
    plt.xticks(rotation=45)
    plot_and_save(fig_pc, plots_dir / f'{fold_tag}_per_class_accuracy.png', logger)
    # ROC and PR curves per class
    fig_roc = plt.figure(figsize=(6, 5))
    for i, cls_name in enumerate(dataset.class_names):
        fpr, tpr, _ = roc_curve(val_true[:, i], val_pred_probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{cls_name} (AUC={roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Chance')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curves – {fold_tag}')
    plt.legend()
    plot_and_save(fig_roc, plots_dir / f'{fold_tag}_roc_curves.png', logger)
    fig_pr = plt.figure(figsize=(6, 5))
    for i, cls_name in enumerate(dataset.class_names):
        precision, recall, _ = precision_recall_curve(val_true[:, i], val_pred_probs[:, i])
        pr_auc = auc(recall, precision)
        plt.plot(recall, precision, label=f'{cls_name} (AUC={pr_auc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision‑Recall Curves – {fold_tag}')
    plt.legend()
    plot_and_save(fig_pr, plots_dir / f'{fold_tag}_pr_curves.png', logger)
    # Calibration / reliability diagram
    fig_cal = plt.figure(figsize=(6, 5))
    plt.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated')
    bin_centers = np.linspace(0.05, 0.95, 10)
    plt.plot(bin_centers, fold_results.calibration_bins['bin_acc'], marker='o', label='Accuracy in Bin')
    plt.plot(bin_centers, fold_results.calibration_bins['bin_conf'], marker='s', label='Average Confidence in Bin')
    plt.xlabel('Predicted Probability Bin Center')
    plt.ylabel('Proportion')
    plt.title(f'Reliability Diagram – {fold_tag} (ECE={fold_results.ece:.3f})')
    plt.legend()
    plot_and_save(fig_cal, plots_dir / f'{fold_tag}_reliability.png', logger)
    # Threshold sweep curves: F1, precision, recall vs threshold
    fig_thr = plt.figure(figsize=(6, 5))
    plt.plot(thr_data := fold_results.threshold_metrics['threshold'], fold_results.threshold_metrics['f1'], label='F1')
    plt.plot(thr_data, fold_results.threshold_metrics['precision'], label='Precision')
    plt.plot(thr_data, fold_results.threshold_metrics['recall'], label='Recall')
    plt.xlabel('Threshold')
    plt.title(f'Threshold Sweep – {fold_tag}')
    plt.legend()
    plot_and_save(fig_thr, plots_dir / f'{fold_tag}_threshold_sweep.png', logger)
    # Predicted label histogram
    fig_hist = plt.figure(figsize=(6, 4))
    sns.countplot(x=val_pred_labels, palette='Set2')
    plt.xlabel('Predicted Label')
    plt.ylabel('Count')
    plt.title(f'Predicted Label Distribution – {fold_tag}')
    plt.xticks(ticks=range(len(dataset.class_names)), labels=dataset.class_names, rotation=45)
    plot_and_save(fig_hist, plots_dir / f'{fold_tag}_predicted_histogram.png', logger)
    # Misclassification bar plot per class
    fig_mis = plt.figure(figsize=(6, 4))
    mis_counts = mis_df.groupby('True Label').size().reindex(dataset.class_names, fill_value=0)
    sns.barplot(x=mis_counts.index, y=mis_counts.values, palette='Reds')
    plt.title(f'Misclassified Samples per True Class – {fold_tag}')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plot_and_save(fig_mis, plots_dir / f'{fold_tag}_misclassified_per_class.png', logger)
    # t‑SNE embedding of penultimate layer outputs (only on a subset for speed)
    feat_model = models.Model(inputs=model.input, outputs=model.layers[-2].output)
    # Use a subset of validation data to compute t‑SNE; limit to 200 samples for efficiency
    subset_size = min(200, len(val_pred_labels))
    subset_indices = np.random.choice(len(val_pred_labels), subset_size, replace=False)
    subset_paths = [val_files[i] for i in subset_indices]
    subset_labels = [val_labels[i] for i in subset_indices]
    subset_images = np.array([load_image(p, img_size) for p in subset_paths])
    subset_features = feat_model.predict(subset_images, verbose=0)
    try:
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, subset_size // 3))
        tsne_emb = tsne.fit_transform(subset_features)
    except Exception:
        # Fallback to LDA if TSNE fails
        lda = LDA(n_components=2)
        tsne_emb = lda.fit_transform(subset_features, subset_labels)

    fig_tsne = plt.figure(figsize=(6, 5))
    scatter = plt.scatter(tsne_emb[:, 0], tsne_emb[:, 1], c=subset_labels, cmap='tab10', alpha=0.7)
    plt.title(f't‑SNE Embedding – {fold_tag}')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    # Add legend for classes
    handles = [plt.Line2D([0], [0], marker='o', color='w', label=dataset.class_names[i],
                          markerfacecolor=plt.cm.tab10(i/len(dataset.class_names)), markersize=6)
               for i in range(len(dataset.class_names))]
    plt.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left')
    plot_and_save(fig_tsne, plots_dir / f'{fold_tag}_tsne.png', logger)
    def compute_gradcam(img_array: np.ndarray, model: tf.keras.Model, pred_index: int) -> Optional[np.ndarray]:
        # Find the last Conv2D layer (Grad-CAM needs spatial feature maps)
        last_conv = None
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                last_conv = layer
                break
        if last_conv is None:
            return None  # Model has no conv layer; skip Grad-CAM

        # Build a model that maps input -> (last conv feature maps, predictions)
        grad_model = tf.keras.models.Model(inputs=model.inputs,
                                        outputs=[last_conv.output, model.output])

        with tf.GradientTape() as tape:
            conv_out, preds = grad_model(tf.expand_dims(img_array, axis=0))  # conv_out: (1,H,W,C)
            class_score = preds[:, pred_index]  # scalar per batch

        # Gradients of the class score w.r.t conv feature maps
        grads = tape.gradient(class_score, conv_out)  # (1,H,W,C)
        conv_out = conv_out[0]   # (H,W,C)
        grads = grads[0]         # (H,W,C)

        # Channel-wise weights = mean gradient over spatial dims
        weights = tf.reduce_mean(grads, axis=(0, 1))  # (C,)

        # Weighted sum over channels -> (H,W)
        cam = tf.tensordot(conv_out, weights, axes=([2], [0]))
        cam = tf.maximum(cam, 0)
        cam = cam / (tf.reduce_max(cam) + 1e-8)
        return cam.numpy()

    # Select up to 3 misclassified images for grad‑CAM
    n_gradcam = min(3, len(mis_idx))
    if n_gradcam > 0:
        fig_cam = plt.figure(figsize=(3 * n_gradcam, 4))
        for i in range(n_gradcam):
            idx = mis_idx[i]
            img_path = val_files[idx]
            img = load_image(img_path, img_size)
            pred_cls = val_pred_labels[idx]
            heatmap = compute_gradcam(img, model, pred_cls)
            if heatmap is None:
                continue  # no conv layer -> skip

            # Rescale heatmap to image size
            heatmap_resized = tf.image.resize(heatmap[..., None], img_size).numpy().squeeze()
            heatmap_resized = (heatmap_resized - heatmap_resized.min()) / (heatmap_resized.max() - heatmap_resized.min() + 1e-8)
            # Overlay on image (same as before)
            overlay = np.uint8(255 * heatmap_resized)
            overlay_rgb = plt.cm.jet(overlay)[:, :, :3]
            blended = 0.4 * overlay_rgb + 0.6 * img
            plt.subplot(2, n_gradcam, i + 1)
            plt.imshow(img)
            plt.title(f'T:{dataset.class_names[val_true_labels[idx]]}\nP:{dataset.class_names[pred_cls]}')
            plt.axis('off')
            plt.subplot(2, n_gradcam, n_gradcam + i + 1)
            plt.imshow(blended)
            plt.title('Grad‑CAM')
            plt.axis('off')
        plt.suptitle(f'Grad‑CAM Examples – {fold_tag}')
        plot_and_save(fig_cam, plots_dir / f'{fold_tag}_gradcam_examples.png', logger)
    # Data augmentation effects: show original vs augmented (flip/rotate) for a few training samples
    n_aug = min(3, len(train_files))
    if n_aug > 0:
        fig_aug = plt.figure(figsize=(4 * n_aug, 4))
        for i in range(n_aug):
            img_path = train_files[i]
            img = load_image(img_path, img_size)
            plt.subplot(2, n_aug, i + 1)
            plt.imshow(img)
            plt.title('Original')
            plt.axis('off')
            # Apply a simple augmentation: horizontal flip and small rotation
            aug_img = tf.image.flip_left_right(img)
            aug_img = tf.keras.preprocessing.image.random_rotation(aug_img.numpy(), rg=20, row_axis=0, col_axis=1, channel_axis=2)
            plt.subplot(2, n_aug, n_aug + i + 1)
            plt.imshow(aug_img)
            plt.title('Augmented')
            plt.axis('off')
        plt.suptitle(f'Data Augmentation Examples – {fold_tag}')
        plot_and_save(fig_aug, plots_dir / f'{fold_tag}_augmentation_examples.png', logger)
    logger.info('--- Completed fold %d ---', fold_idx + 1)
    return fold_results

def hyperparameter_search(dataset: DatasetInfo, img_size: Tuple[int, int], batch_size: int, folds: int,
                          hyperparams: Dict[str, List[Any]], epochs: int, logger: logging.Logger,
                          output_dirs: Tuple[Path, Path, Path], quick: bool) -> pd.DataFrame:

    lr_list = hyperparams.get('learning_rate', [1e-4])
    dr_list = hyperparams.get('dropout_rate', [0.5])
    results: List[Dict[str, Any]] = []
    # Use a single train/val split derived from the first fold of a StratifiedKFold
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    indices = np.arange(len(dataset.files))
    label_arr = np.array(dataset.labels)
    for train_idx, val_idx in skf.split(indices, label_arr):
        # Use only the first split
        break
    for lr in lr_list:
        for dr in dr_list:
            logger.info('Hyperparam search: lr=%.1e, dropout=%.2f', lr, dr)
            fold_result = train_and_evaluate_one_fold(
                fold_idx=0,
                train_idx=train_idx,
                val_idx=val_idx,
                test_indices=None,
                dataset=dataset,
                img_size=img_size,
                batch_size=batch_size,
                epochs=epochs if not quick else max(5, epochs // 5),
                logger=logger,
                output_dirs=output_dirs,
                quick=quick,
                learning_rate=lr,
                dropout_rate=dr,
            )
            results.append({
                'learning_rate': lr,
                'dropout_rate': dr,
                **fold_result.metrics
            })
    df_results = pd.DataFrame(results)
    # Save table and plot heatmap
    _, plots_dir, tables_dir = output_dirs
    hp_table_path = tables_dir / 'hyperparameter_search_results.csv'
    save_dataframe(df_results, hp_table_path, logger)
    # Heatmap for accuracy per hyperparam
    pivot = df_results.pivot_table(index='learning_rate', columns='dropout_rate', values='accuracy')
    fig_heat = plt.figure(figsize=(6, 5))
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='viridis')
    plt.title('Validation Accuracy Heatmap – Hyperparameter Search')
    plot_and_save(fig_heat, plots_dir / 'hyperparameter_search_heatmap.png', logger)
    return df_results

def generate_report(output_dirs: Tuple[Path, Path, Path], fold_results: List[FoldResults], aggregated_metrics: pd.DataFrame,
                    logger: logging.Logger) -> None:
    
    output_root, plots_dir, tables_dir = output_dirs
    report_path = output_root / 'report.pdf'
    logger.info('Generating PDF report at %s', report_path)
    with PdfPages(report_path) as pdf:
        # Title page
        fig_title = plt.figure(figsize=(8.27, 11.69))  # A4 size in inches
        plt.axis('off')
        plt.text(0.5, 0.8, 'RR17 Brain Tumor Classifier Report', ha='center', va='center', fontsize=24, weight='bold')
        plt.text(0.5, 0.7, f'Generated on {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
                 ha='center', fontsize=12)
        # Insert environment summary
        info_file = output_root / 'runtime_info.json'
        try:
            with open(info_file) as f:
                info = json.load(f)
            env_text = '\n'.join([f'{k}: {v}' for k, v in info.items()])
        except Exception:
            env_text = 'Environment information unavailable.'
        plt.text(0.5, 0.55, env_text, ha='center', va='top', fontsize=10)
        pdf.savefig(fig_title)
        plt.close(fig_title)
        # Table of contents (figures and tables)
        fig_toc = plt.figure(figsize=(8.27, 11.69))
        plt.axis('off')
        plt.text(0.5, 0.95, 'Table of Contents', ha='center', fontsize=18, weight='bold')
        # Gather list of plots and tables
        plot_files = sorted([f for f in plots_dir.glob('*.png')])
        table_files = sorted([f for f in tables_dir.glob('*.csv')])
        y_pos = 0.9
        idx = 1
        for f in plot_files:
            plt.text(0.1, y_pos, f'Fig {idx}: {f.name}', fontsize=8)
            y_pos -= 0.02
            idx += 1
            if y_pos < 0.1:
                break  # Only list as many as fit
        for f in table_files:
            if y_pos < 0.1:
                break
            plt.text(0.1, y_pos, f'Table {idx}: {f.name}', fontsize=8)
            y_pos -= 0.02
            idx += 1
        pdf.savefig(fig_toc)
        plt.close(fig_toc)
        # Add each plot as a thumbnail page
        for i, plot_path in enumerate(plot_files, start=1):
            fig = plt.figure(figsize=(8.27, 11.69))
            plt.axis('off')
            img = plt.imread(plot_path)
            # Maintain aspect ratio of the thumbnail; embed at top
            h, w = img.shape[:2]
            # Compute scaling to fit width with margin
            margin = 0.1
            width = 1.0 - 2 * margin
            height = width * (h / w)
            if height > 0.8:
                height = 0.8
                width = height * (w / h)
            x0 = (1 - width) / 2
            y0 = 0.6
            ax_img = fig.add_axes([x0, y0, width, height])
            ax_img.imshow(img)
            ax_img.axis('off')
            # Caption
            plt.text(0.5, 0.55, f'Figure {i}: {plot_path.name}', ha='center', fontsize=10)
            pdf.savefig(fig)
            plt.close(fig)
        # Add selected tables as rendered images (first 5 rows) if small enough
        for j, table_path in enumerate(table_files, start=1):
            try:
                df = pd.read_csv(table_path)
            except Exception:
                continue
            fig = plt.figure(figsize=(8.27, 11.69))
            plt.axis('off')
            plt.text(0.5, 0.95, f'Table {j}: {table_path.name}', ha='center', fontsize=12, weight='bold')
            # Use a smaller subset of the table if very large
            display_df = df.head(15)
            ax_table = fig.add_axes([0.1, 0.4, 0.8, 0.5])
            ax_table.axis('off')
            # Render table
            tbl = ax_table.table(cellText=display_df.values,
                                 colLabels=display_df.columns,
                                 rowLabels=[str(i + 1) for i in range(len(display_df))],
                                 cellLoc='center', loc='upper center')
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(8)
            tbl.auto_set_column_width(col=list(range(len(display_df.columns))))
            pdf.savefig(fig)
            plt.close(fig)
    logger.info('PDF report generated at %s', report_path)

def aggregate_fold_results(fold_results: List[FoldResults], class_names: List[str], logger: logging.Logger,
                          output_dirs: Tuple[Path, Path, Path]) -> pd.DataFrame:
    
    _, plots_dir, tables_dir = output_dirs
    # Aggregate simple metrics
    metric_names = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc_macro', 'pr_auc_macro']
    agg_data = {}
    for m in metric_names:
        values = [fr.metrics[m] for fr in fold_results]
        agg_data[f'{m}_mean'] = np.mean(values)
        agg_data[f'{m}_std'] = np.std(values)
    agg_df = pd.DataFrame([agg_data])
    save_dataframe(agg_df, tables_dir / 'aggregated_metrics.csv', logger)
    # Aggregate confusion matrices
    cms = [fr.confusion_matrix for fr in fold_results]
    agg_cm = np.sum(cms, axis=0)
    cm_df = pd.DataFrame(agg_cm, index=class_names, columns=class_names)
    save_dataframe(cm_df.reset_index().rename(columns={'index': 'True\Pred'}), tables_dir / 'aggregated_confusion_matrix.csv', logger)
    # Plot aggregated confusion matrix
    fig_cm = plt.figure(figsize=(6, 5))
    sns.heatmap(agg_cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Purples')
    plt.title('Aggregated Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plot_and_save(fig_cm, plots_dir / 'aggregated_confusion_matrix.png', logger)
    # Aggregate per class accuracies
    pcs = [fr.per_class_metrics.set_index('Class')['Accuracy'] for fr in fold_results]
    pcs_df = pd.concat(pcs, axis=1)
    pcs_df.columns = [f'fold{i+1}' for i in range(len(fold_results))]
    pcs_df['mean'] = pcs_df.mean(axis=1)
    pcs_df['std'] = pcs_df.std(axis=1)
    save_dataframe(pcs_df.reset_index(), tables_dir / 'aggregated_per_class_accuracy.csv', logger)
    # Plot aggregated per class accuracy
    fig_pc = plt.figure(figsize=(6, 4))
    sns.barplot(x=pcs_df.index, y='mean', data=pcs_df.reset_index())
    plt.title('Aggregated Per‑Class Accuracy (mean)')
    plt.xticks(rotation=45)
    plt.ylabel('Accuracy')
    plot_and_save(fig_pc, plots_dir / 'aggregated_per_class_accuracy.png', logger)
    # ROC/PR across folds (macro average)
    # Compute mean ROC and PR curves by concatenating probabilities
    all_true = np.concatenate([fr.y_true for fr in fold_results], axis=0)
    all_pred = np.concatenate([fr.y_pred_probs for fr in fold_results], axis=0)
    fig_roc = plt.figure(figsize=(6, 5))
    for i, cls_name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(all_true[:, i], all_pred[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{cls_name} (AUC={roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title('Aggregate ROC Curves (All Folds)')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plot_and_save(fig_roc, plots_dir / 'aggregated_roc_curves.png', logger)
    fig_pr = plt.figure(figsize=(6, 5))
    for i, cls_name in enumerate(class_names):
        precision, recall, _ = precision_recall_curve(all_true[:, i], all_pred[:, i])
        pr_auc = auc(recall, precision)
        plt.plot(recall, precision, label=f'{cls_name} (AUC={pr_auc:.2f})')
    plt.title('Aggregate Precision‑Recall Curves (All Folds)')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plot_and_save(fig_pr, plots_dir / 'aggregated_pr_curves.png', logger)
    # Aggregate calibration
    eces = [fr.ece for fr in fold_results]
    mean_ece = np.mean(eces)
    std_ece = np.std(eces)
    ece_df = pd.DataFrame({'ECE Mean': [mean_ece], 'ECE Std': [std_ece]})
    save_dataframe(ece_df, tables_dir / 'aggregated_ece.csv', logger)
    # Plot aggregated reliability diagram using concatenated predictions
    ece_agg, bin_acc, bin_conf, bin_count = compute_ece(all_true, all_pred, n_bins=10)
    fig_cal = plt.figure(figsize=(6, 5))
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
    bin_centers = np.linspace(0.05, 0.95, 10)
    plt.plot(bin_centers, bin_acc, marker='o', label='Accuracy')
    plt.plot(bin_centers, bin_conf, marker='s', label='Confidence')
    plt.title(f'Aggregate Reliability Diagram (ECE={ece_agg:.3f})')
    plt.xlabel('Predicted Probability Bin Center')
    plt.ylabel('Proportion')
    plt.legend()
    plot_and_save(fig_cal, plots_dir / 'aggregated_reliability.png', logger)
    # Aggregate threshold metrics across folds (mean and std for each threshold)
    thresholds = fold_results[0].threshold_metrics['threshold']
    thr_metrics = ['accuracy', 'precision', 'recall', 'f1']
    agg_thr_data = {}
    for m in thr_metrics:
        vals = np.stack([fr.threshold_metrics[m].values for fr in fold_results], axis=0)
        agg_thr_data[f'{m}_mean'] = vals.mean(axis=0)
        agg_thr_data[f'{m}_std'] = vals.std(axis=0)
    agg_thr_df = pd.DataFrame({'threshold': thresholds, **agg_thr_data})
    save_dataframe(agg_thr_df, tables_dir / 'aggregated_threshold_metrics.csv', logger)
    # Plot aggregated threshold curves
    fig_thr = plt.figure(figsize=(6, 5))
    for m in thr_metrics:
        plt.plot(thresholds, agg_thr_df[f'{m}_mean'], label=f'{m.capitalize()} (mean)')
    plt.xlabel('Threshold')
    plt.title('Aggregated Threshold Metrics')
    plt.legend()
    plot_and_save(fig_thr, plots_dir / 'aggregated_threshold_curves.png', logger)
    # Misclassification analysis across folds
    mis_counts_total = {cls: 0 for cls in class_names}
    for fr in fold_results:
        mis_df = fr.misclassified_df
        counts = mis_df.groupby('True Label').size().to_dict()
        for cls, c in counts.items():
            mis_counts_total[cls] += c
    mis_counts_df = pd.DataFrame({'Class': list(mis_counts_total.keys()), 'Misclassified Count': list(mis_counts_total.values())})
    save_dataframe(mis_counts_df, tables_dir / 'aggregated_misclassification_counts.csv', logger)
    fig_mis = plt.figure(figsize=(6, 4))
    sns.barplot(x='Class', y='Misclassified Count', data=mis_counts_df, palette='OrRd')
    plt.title('Misclassified Samples per Class (All Folds)')
    plt.xticks(rotation=45)
    plot_and_save(fig_mis, plots_dir / 'aggregated_misclassified_per_class.png', logger)
    # Return aggregated metrics DataFrame
    return agg_df

def self_check(dataset: DatasetInfo, img_size: Tuple[int, int], batch_size: int, output_dirs: Tuple[Path, Path, Path],
               logger: logging.Logger) -> None:
    
    output_root, plots_dir, tables_dir = output_dirs
    # Find a model checkpoint in the output directory
    ckpts = list(output_root.glob('fold*_best_model.h5'))
    if not ckpts:
        logger.warning('No model checkpoint found for self‑check.')
        return
    ckpt_path = ckpts[0]
    logger.info('Self‑check: loading model from %s', ckpt_path)
    model = tf.keras.models.load_model(ckpt_path)
    # Use a small random subset of the dataset for self check
    idxs = np.random.choice(len(dataset.files), size=min(20, len(dataset.files)), replace=False)
    subset_files = [dataset.files[i] for i in idxs]
    subset_labels = [dataset.labels[i] for i in idxs]
    ds = create_tf_dataset(subset_files, subset_labels, img_size, batch_size, shuffle=False)
    true_labels = np.concatenate([y.numpy() for _, y in ds], axis=0)
    true_label_idx = np.argmax(true_labels, axis=1)
    pred_probs = model.predict(ds, verbose=0)
    pred_label_idx = np.argmax(pred_probs, axis=1)
    acc = accuracy_score(true_label_idx, pred_label_idx)
    # Compare to aggregated accuracy if available
    agg_metrics_path = tables_dir / 'aggregated_metrics.csv'
    expected_acc_mean = None
    try:
        agg_df = pd.read_csv(agg_metrics_path)
        expected_acc_mean = agg_df['accuracy_mean'].iloc[0]
    except Exception:
        pass
    if expected_acc_mean is not None and abs(acc - expected_acc_mean) < 0.1:
        msg = f'SELF‑CHECK PASSED: accuracy={acc:.3f} matches expected mean={expected_acc_mean:.3f}'
        logger.info(msg)
        print(msg)
    else:
        msg = f'SELF‑CHECK FAILED: accuracy={acc:.3f} deviates from expected mean={expected_acc_mean}'
        logger.error(msg)
        print(msg)

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='RR17 Brain Tumor Classifier Extended Pipeline')
    parser.add_argument('--data-dir', type=str, default='./data/brain-tumor-mri-dataset',
                        help='Root directory of the brain tumour dataset (class subdirs inside).')
    parser.add_argument('--download', action='store_true', help='Attempt to download the dataset from Kaggle if missing.')
    parser.add_argument('--kaggle-slug', type=str, default='masoudnickparvar/brain-tumor-mri-dataset',
                        help='Kaggle dataset slug for download.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs per fold.')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size for training.')
    parser.add_argument('--img-size', type=int, nargs=2, default=[224, 224], help='Image size (H W).')
    parser.add_argument('--folds', type=int, default=5, help='Number of folds for cross‑validation (≥5 recommended).')
    parser.add_argument('--quick', action='store_true', help='Run a quick demo with reduced dataset and epochs.')
    parser.add_argument('--hyperparam-search', action='store_true', help='Perform a simple hyperparameter search.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    # Set seeds early
    set_seeds(args.seed)
    # Prepare outputs
    output_root = Path('outputs')
    output_root, plots_dir, tables_dir = create_output_dirs(output_root)
    logger = setup_logging(output_root)
    # Snapshot environment
    snapshot_environment(output_root, logger)
    # Locate or download dataset
    data_dir = Path(args.data_dir)
    if not data_dir.exists() or not any(data_dir.iterdir()):
        if args.download:
            try:
                dataset_dir = download_kaggle_dataset(args.kaggle_slug, Path('./data'), logger)
                data_dir = dataset_dir
            except Exception as e:
                logger.error('Failed to download dataset: %s', e)
                print('❌ Dataset missing and could not be downloaded. Please supply the dataset manually.')
                sys.exit(1)
        else:
            logger.error('Dataset directory %s not found. Use --download to fetch it from Kaggle.', data_dir)
            print(f'❌ Dataset directory {data_dir} not found. Please provide the dataset via --data-dir or enable --download.')
            sys.exit(1)
    # Determine where the class folders reside
    # Many Kaggle datasets contain a Training/Testing split; we include all images for cross‑validation
    if (data_dir / 'Training').exists():
        base_dirs = [data_dir / 'Training', data_dir / 'Testing'] if (data_dir / 'Testing').exists() else [data_dir / 'Training']
        files: List[str] = []
        labels: List[int] = []
        class_names: List[str] = []
        for base in base_dirs:
            f, l, cls = find_image_files(base, logger)
            # Ensure class names are consistent across splits
            if not class_names:
                class_names = cls
            else:
                if cls != class_names:
                    logger.warning('Class names differ between splits. Using first split class order.')
            files.extend(f)
            labels.extend(l)
        dataset = DatasetInfo(files=files, labels=labels, class_names=class_names)
    else:
        # Assume data_dir itself contains class subdirs
        f, l, cls = find_image_files(data_dir, logger)
        dataset = DatasetInfo(files=f, labels=l, class_names=cls)
    # Provide dataset summary and distribution plot (counts per class across entire dataset)
    total_counts = pd.Series(dataset.labels).value_counts().sort_index()
    df_summary = pd.DataFrame({
        'Class': dataset.class_names,
        'Count': total_counts.values,
    })
    save_dataframe(df_summary, tables_dir / 'dataset_summary.csv', logger)
    fig_dist = plt.figure(figsize=(6, 4))
    sns.barplot(x='Class', y='Count', data=df_summary)
    plt.title('Dataset Class Distribution')
    plt.xticks(rotation=45)
    plot_and_save(fig_dist, plots_dir / 'dataset_distribution.png', logger)
    # Create cross‑validation splits
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    indices = np.arange(len(dataset.files))
    fold_results: List[FoldResults] = []
    # Optionally hold out a portion of data as a test set (10% of all samples)
    # Use this set to evaluate each fold’s model after training
    test_indices = None
    if len(dataset.files) > 1000:
        train_indices_all, test_indices_all, _, test_labels_all = train_test_split(indices, dataset.labels, test_size=0.25,
                                                                                   stratify=dataset.labels,
                                                                                   random_state=args.seed)
        # We'll feed test_indices into each fold evaluation
        test_indices = test_indices_all
        indices_for_cv = train_indices_all
        labels_for_cv = np.array(dataset.labels)[indices_for_cv]
    else:
        indices_for_cv = indices
        labels_for_cv = np.array(dataset.labels)
    # Perform cross‑validation
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(indices_for_cv, labels_for_cv)):
        # Map back to original indices if using a reduced training set due to hold‑out
        if test_indices is not None:
            train_idx = indices_for_cv[train_idx]
            val_idx = indices_for_cv[val_idx]
        fr = train_and_evaluate_one_fold(
            fold_idx=fold_idx,
            train_idx=train_idx,
            val_idx=val_idx,
            test_indices=test_indices,
            dataset=dataset,
            img_size=tuple(args.img_size),
            batch_size=args.batch_size,
            epochs=args.epochs if not args.quick else max(5, args.epochs // 5),
            logger=logger,
            output_dirs=(output_root, plots_dir, tables_dir),
            quick=args.quick,
        )
        fold_results.append(fr)
        # Stop early in quick mode after one fold
        if args.quick:
            break
    # Hyperparameter search (optional)
    if args.hyperparam_search and not args.quick:
        hyperparams = {
            'learning_rate': [1e-4, 5e-4, 1e-3],
            'dropout_rate': [0.3, 0.5, 0.7],
        }
        hp_results_df = hyperparameter_search(dataset, tuple(args.img_size), args.batch_size, args.folds,
                                              hyperparams, args.epochs // 2, logger, (output_root, plots_dir, tables_dir), args.quick)
    # Aggregate results and generate summary
    aggregated_df = aggregate_fold_results(fold_results, dataset.class_names, logger, (output_root, plots_dir, tables_dir))
    # Generate report PDF
    try:
        generate_report((output_root, plots_dir, tables_dir), fold_results, aggregated_df, logger)
    except Exception as e:
        logger.error('Failed to generate PDF report: %s', e)
    # Self‑check
    try:
        self_check(dataset, tuple(args.img_size), args.batch_size, (output_root, plots_dir, tables_dir), logger)
    except Exception as e:
        logger.error('Self‑check failed: %s', e)
    try:
        launch_gradio(output_root, dataset.class_names, tuple(args.img_size))
    except Exception as e:
        logger.warning("Gradio UI not launched: %s", e)


if __name__ == '__main__':
    main()