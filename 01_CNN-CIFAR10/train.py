#!/usr/bin/env python3
from pathlib import Path
import pickle
import json
import sys
from datetime import datetime

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class RandomCutout(layers.Layer):
    def __init__(self, mask_size=(8, 8), probability=0.3, **kwargs):
        super().__init__(**kwargs)
        self.mask_size = mask_size
        self.probability = probability

    def call(self, inputs, training=None):
        if not training:
            return inputs

        shape = tf.shape(inputs)
        batch_size, height, width = shape[0], shape[1], shape[2]
        mask_h, mask_w = self.mask_size

        apply_mask = tf.random.uniform([batch_size]) < self.probability
        y0 = tf.random.uniform([batch_size], 0, height - mask_h + 1, dtype=tf.int32)
        x0 = tf.random.uniform([batch_size], 0, width - mask_w + 1, dtype=tf.int32)

        grid_y, grid_x = tf.meshgrid(tf.range(height), tf.range(width), indexing="ij")
        grid_y = tf.expand_dims(grid_y, axis=0)
        grid_x = tf.expand_dims(grid_x, axis=0)
        y0 = tf.reshape(y0, [-1, 1, 1])
        x0 = tf.reshape(x0, [-1, 1, 1])

        mask = ~(
            (grid_y >= y0)
            & (grid_y < y0 + mask_h)
            & (grid_x >= x0)
            & (grid_x < x0 + mask_w)
        )
        apply_mask = tf.reshape(apply_mask, [-1, 1, 1])
        mask = tf.where(apply_mask, mask, tf.ones_like(mask))
        mask = tf.cast(mask, inputs.dtype)
        mask = tf.expand_dims(mask, axis=-1)
        return inputs * mask


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


# -----------------------------------------------------------------------------
# 可调整参数
# -----------------------------------------------------------------------------

SEED = 42
EPOCHS = 120
BATCH_SIZE = 64
VAL_RATIO = 0.1

LEARNING_RATE = 5e-4
MIN_LEARNING_RATE = 1e-5
CLIPNORM = 1.0
L2_WEIGHT = 1e-4

USE_AUGMENTATION = True
CUTOUT_MASK_SIZE = (4, 4)
CUTOUT_PROBABILITY = 0.0
DENSE_DROPOUT = 0.3

LR_PATIENCE = 4
EARLY_STOP_PATIENCE = 15
EARLY_STOP_MIN_DELTA = 1e-4
TRAIN_SAMPLES_PER_CLASS = None

DATA_DIR = Path(__file__).resolve().parent / "cifar-10-python" / "cifar-10-batches-py"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"

# -----------------------------------------------------------------------------
# 基础准备
# -----------------------------------------------------------------------------

tf.random.set_seed(SEED)
np.random.seed(SEED)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")
run_dir = OUTPUT_DIR / run_name
run_dir.mkdir(parents=True, exist_ok=True)

console_log_path = run_dir / "console.log"
console_log_file = console_log_path.open("w", encoding="utf-8")
original_stdout = sys.stdout
original_stderr = sys.stderr
sys.stdout = Tee(sys.stdout, console_log_file)
sys.stderr = Tee(sys.stderr, console_log_file)

gpus = tf.config.list_physical_devices("GPU")
cpus = tf.config.list_physical_devices("CPU")
print(f"run dir: {run_dir}")
print(f"TensorFlow version: {tf.__version__}")
print(f"CPU devices: {len(cpus)}")
print(f"GPU devices: {len(gpus)}")
if gpus:
    print(f"using GPU: {gpus[0].name}")

if not DATA_DIR.exists():
    raise SystemExit(f"数据目录不存在: {DATA_DIR}")


# -----------------------------------------------------------------------------
# 读取 CIFAR-10
# -----------------------------------------------------------------------------

with (DATA_DIR / "batches.meta").open("rb") as file:
    meta = pickle.load(file, encoding="bytes")
class_names = [name.decode("utf-8") for name in meta[b"label_names"]]

train_batches = []
train_labels = []
for batch_index in range(1, 6):
    with (DATA_DIR / f"data_batch_{batch_index}").open("rb") as file:
        batch = pickle.load(file, encoding="bytes")
    train_batches.append(batch[b"data"])
    train_labels.append(np.asarray(batch[b"labels"], dtype=np.int64))

with (DATA_DIR / "test_batch").open("rb") as file:
    test_batch = pickle.load(file, encoding="bytes")

train_x = np.concatenate(train_batches, axis=0)
train_y = np.concatenate(train_labels, axis=0)
test_x = np.asarray(test_batch[b"data"])
test_y = np.asarray(test_batch[b"labels"], dtype=np.int64)

train_x = train_x.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).astype("float32") / 255.0
test_x = test_x.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).astype("float32") / 255.0


# -----------------------------------------------------------------------------
# 可选: 每类只保留部分样本
# -----------------------------------------------------------------------------

if TRAIN_SAMPLES_PER_CLASS is not None:
    rng = np.random.default_rng(SEED)
    keep_indices = []
    for class_id in range(10):
        class_indices = np.where(train_y == class_id)[0]
        rng.shuffle(class_indices)
        keep_indices.extend(class_indices[:TRAIN_SAMPLES_PER_CLASS].tolist())
    keep_indices = np.asarray(keep_indices, dtype=np.int64)
    rng.shuffle(keep_indices)
    train_x = train_x[keep_indices]
    train_y = train_y[keep_indices]


# -----------------------------------------------------------------------------
# 划分训练集 / 验证集
# -----------------------------------------------------------------------------

rng = np.random.default_rng(SEED)
train_indices = []
val_indices = []

for class_id in range(10):
    class_indices = np.where(train_y == class_id)[0]
    rng.shuffle(class_indices)
    val_count = max(1, int(round(len(class_indices) * VAL_RATIO)))
    val_indices.extend(class_indices[:val_count].tolist())
    train_indices.extend(class_indices[val_count:].tolist())

train_indices = np.asarray(train_indices, dtype=np.int64)
val_indices = np.asarray(val_indices, dtype=np.int64)
rng.shuffle(train_indices)
rng.shuffle(val_indices)

val_x = train_x[val_indices]
val_y = train_y[val_indices]
train_x = train_x[train_indices]
train_y = train_y[train_indices]


# 标准化
# -----------------------------------------------------------------------------

mean = train_x.mean(axis=(0, 1, 2), keepdims=True).astype("float32")
std = train_x.std(axis=(0, 1, 2), keepdims=True).astype("float32") + 1e-7

train_x = ((train_x - mean) / std).astype("float32")
val_x = ((val_x - mean) / std).astype("float32")
test_x = ((test_x - mean) / std).astype("float32")

print("class names:", class_names)
print("train:", train_x.shape, train_y.shape)
print("val:  ", val_x.shape, val_y.shape)
print("test: ", test_x.shape, test_y.shape)
print(
    "channel mean/std:",
    mean.reshape(-1).round(4).tolist(),
    std.reshape(-1).round(4).tolist(),
)

config = {
    "seed": SEED,
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "val_ratio": VAL_RATIO,
    "learning_rate": LEARNING_RATE,
    "min_learning_rate": MIN_LEARNING_RATE,
    "clipnorm": CLIPNORM,
    "l2_weight": L2_WEIGHT,
    "use_augmentation": USE_AUGMENTATION,
    "cutout_mask_size": list(CUTOUT_MASK_SIZE),
    "cutout_probability": CUTOUT_PROBABILITY,
    "dense_dropout": DENSE_DROPOUT,
    "lr_patience": LR_PATIENCE,
    "early_stop_patience": EARLY_STOP_PATIENCE,
    "early_stop_min_delta": EARLY_STOP_MIN_DELTA,
    "train_samples_per_class": TRAIN_SAMPLES_PER_CLASS,
    "train_size": int(len(train_x)),
    "val_size": int(len(val_x)),
    "test_size": int(len(test_x)),
    "class_names": class_names,
    "run_dir": str(run_dir),
}
config_path = run_dir / "config.json"
config_path.write_text(
    json.dumps(config, ensure_ascii=False, indent=2),
    encoding="utf-8",
)


# -----------------------------------------------------------------------------
# tf.data.Dataset
# -----------------------------------------------------------------------------

train_ds = tf.data.Dataset.from_tensor_slices((train_x, train_y))
train_ds = train_ds.shuffle(len(train_x), seed=SEED).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

val_ds = tf.data.Dataset.from_tensor_slices((val_x, val_y))
val_ds = val_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

test_ds = tf.data.Dataset.from_tensor_slices((test_x, test_y))
test_ds = test_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


# -----------------------------------------------------------------------------
# 模型
# -----------------------------------------------------------------------------

regularizer = keras.regularizers.l2(L2_WEIGHT)
model_layers = []

if USE_AUGMENTATION:
    model_layers.extend(
        [
            layers.Input(shape=(32, 32, 3)),
            layers.RandomFlip("horizontal"),
            layers.RandomTranslation(0.1, 0.1),
            layers.RandomRotation(0.05),
            layers.RandomZoom(0.1),
            RandomCutout(mask_size=CUTOUT_MASK_SIZE, probability=CUTOUT_PROBABILITY),
        ]
    )
else:
    model_layers.append(layers.Input(shape=(32, 32, 3)))

model_layers.extend(
    [
        layers.Conv2D(
            filters=32,
            kernel_size=5,
            activation="relu",
            padding="same",
            kernel_regularizer=regularizer,
        ),
        layers.MaxPool2D(),
        layers.BatchNormalization(),
        layers.Dropout(0.10),
        layers.Conv2D(
            filters=64,
            kernel_size=3,
            activation="relu",
            padding="same",
            kernel_regularizer=regularizer,
        ),
        layers.MaxPool2D(),
        layers.BatchNormalization(),
        layers.Dropout(0.15),
        layers.Conv2D(
            filters=64,
            kernel_size=3,
            activation="relu",
            padding="same",
            kernel_regularizer=regularizer,
        ),
        layers.MaxPool2D(),
        layers.BatchNormalization(),
        layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(128, activation="relu", kernel_regularizer=regularizer),
        layers.Dropout(DENSE_DROPOUT),
        layers.Dense(10, activation="softmax"),
    ]
)

model = keras.Sequential(model_layers, name="cifar10_sequential")
model.compile(
    optimizer=keras.optimizers.Adam(
        learning_rate=LEARNING_RATE,
        clipnorm=CLIPNORM,
    ),
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"],
)
model.summary()

model_summary_path = run_dir / "model_summary.txt"
with model_summary_path.open("w", encoding="utf-8") as summary_file:
    model.summary(print_fn=lambda line: summary_file.write(line + "\n"))


# -----------------------------------------------------------------------------
# 回调
# -----------------------------------------------------------------------------

best_model_path = run_dir / "cifar10_best.keras"
callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath=best_model_path,
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
        verbose=1,
    ),
    keras.callbacks.CSVLogger(run_dir / "epoch_metrics.csv"),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=LR_PATIENCE,
        min_lr=MIN_LEARNING_RATE,
        verbose=1,
    ),
    keras.callbacks.EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=EARLY_STOP_PATIENCE,
        min_delta=EARLY_STOP_MIN_DELTA,
        restore_best_weights=True,
        verbose=1,
    ),
    keras.callbacks.TerminateOnNaN(),
]


# -----------------------------------------------------------------------------
# 训练
# -----------------------------------------------------------------------------

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1,
)


# -----------------------------------------------------------------------------
# 测试与保存
# -----------------------------------------------------------------------------

train_loss, train_acc = model.evaluate(train_ds, verbose=0)
val_loss, val_acc = model.evaluate(val_ds, verbose=0)
test_loss, test_acc = model.evaluate(test_ds, verbose=0)
print(f"train loss: {train_loss:.4f}")
print(f"train accuracy: {train_acc:.4f}")
print(f"val loss: {val_loss:.4f}")
print(f"val accuracy: {val_acc:.4f}")
print(f"test loss: {test_loss:.4f}")
print(f"test accuracy: {test_acc:.4f}")

last_model_path = run_dir / "cifar10_last.keras"
model.save(last_model_path)
print(f"saved best model: {best_model_path}")
print(f"saved last model: {last_model_path}")

# 额外保存一份 latest，方便直接访问最新模型
model.save(OUTPUT_DIR / "cifar10_best.keras")
model.save(OUTPUT_DIR / "cifar10_last.keras")
print(f"saved latest best model: {OUTPUT_DIR / 'cifar10_best.keras'}")
print(f"saved latest last model: {OUTPUT_DIR / 'cifar10_last.keras'}")

try:
    import matplotlib.pyplot as plt

    epochs = range(1, len(history.history["loss"]) + 1)
    curve_path = run_dir / "training_curves.png"

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, history.history["accuracy"], label="train_acc")
    axes[0].plot(epochs, history.history["val_accuracy"], label="val_acc")
    axes[0].set_title("Accuracy")
    axes[0].legend()

    axes[1].plot(epochs, history.history["loss"], label="train_loss")
    axes[1].plot(epochs, history.history["val_loss"], label="val_loss")
    axes[1].set_title("Loss")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(curve_path, dpi=150)
    plt.close(fig)
    fig.savefig(OUTPUT_DIR / "training_curves.png", dpi=150)
    print(f"saved curves: {curve_path}")
except Exception:
    pass

history_path = run_dir / "history.json"
history_path.write_text(
    json.dumps(history.history, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

best_epoch = int(np.argmax(history.history["val_accuracy"]) + 1)
final_metrics = {
    "train_loss": float(train_loss),
    "train_accuracy": float(train_acc),
    "val_loss": float(val_loss),
    "val_accuracy": float(val_acc),
    "test_loss": float(test_loss),
    "test_accuracy": float(test_acc),
    "best_val_accuracy": float(np.max(history.history["val_accuracy"])),
    "best_val_loss": float(np.min(history.history["val_loss"])),
    "best_epoch_by_val_accuracy": best_epoch,
    "epochs_ran": int(len(history.history["loss"])),
    "run_dir": str(run_dir),
    "best_model_path": str(best_model_path),
    "last_model_path": str(last_model_path),
}
final_metrics_path = run_dir / "final_metrics.json"
final_metrics_path.write_text(
    json.dumps(final_metrics, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

latest_summary_path = OUTPUT_DIR / "latest_run.txt"
latest_summary_path.write_text(str(run_dir) + "\n", encoding="utf-8")
latest_metrics_path = OUTPUT_DIR / "final_metrics.json"
latest_metrics_path.write_text(
    json.dumps(final_metrics, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

sys.stdout = original_stdout
sys.stderr = original_stderr
console_log_file.close()
