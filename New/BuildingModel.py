import tensorflow as tf
import numpy as np
from pathlib import Path
from PIL import Image
import os

# =========================
# CONFIG
# =========================
DATASET_DIR = "C:/Cambridge/Societies/HABS/dataset"
IMG_SIZE = 48
BATCH_SIZE = 32
EPOCHS = 15
SEED = 123

# =========================
# MODEL
# =========================
model = tf.keras.Sequential([
    tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),

    # IMPORTANT: explicit normalization
    tf.keras.layers.Rescaling(1.0 / 255.0),

    tf.keras.layers.Conv2D(8, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(16, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(32, 3, activation='relu'),

    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(2, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# =========================
# DATASETS
# =========================
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE
)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(500).prefetch(AUTOTUNE)
val_ds = val_ds.cache().prefetch(AUTOTUNE)

# =========================
# TRAIN
# =========================
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)

print("Final val accuracy:", history.history["val_accuracy"][-1])

def representative_dataset():
    base = Path(DATASET_DIR)
    images = list(base.glob("*/*.png"))[:100]

    for p in images:
        img = Image.open(p).resize((IMG_SIZE, IMG_SIZE))
        img = np.array(img, dtype=np.float32) / 255.0
        img = np.expand_dims(img, axis=0)
        yield [img]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset

converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS_INT8
]

converter.inference_input_type = tf.int8
converter.inference_output_type = tf.float32

tflite_model = converter.convert()

with open("cloud_vs_clear_48x48_int8.tflite", "wb") as f:
    f.write(tflite_model)

print("Model size:", os.path.getsize("cloud_vs_clear_48x48_int8.tflite"))
