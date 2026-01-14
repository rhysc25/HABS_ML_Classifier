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
EPOCHS = 10
SEED = 123

# =========================
# MODEL (TinyML friendly)
# =========================
model = tf.keras.Sequential([
    tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),

    tf.keras.layers.Conv2D(8, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(16, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),

    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(2, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
)

model.summary()

# =========================
# DATASETS (48x48 RGB)
# =========================
train_dataset = tf.keras.utils.image_dataset_from_directory(
    directory=DATASET_DIR,
    labels="inferred",
    label_mode="int",
    color_mode="rgb",
    batch_size=BATCH_SIZE,
    image_size=(IMG_SIZE, IMG_SIZE),
    shuffle=True,
    seed=SEED,
    validation_split=0.2,
    subset="training"
)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    directory=DATASET_DIR,
    labels="inferred",
    label_mode="int",
    color_mode="rgb",
    batch_size=BATCH_SIZE,
    image_size=(IMG_SIZE, IMG_SIZE),
    shuffle=True,
    seed=SEED,
    validation_split=0.2,
    subset="validation"
)

AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.cache().prefetch(buffer_size=AUTOTUNE)

# =========================
# TRAIN
# =========================
history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHS
)

print("\nLatest validation accuracy:", history.history['val_accuracy'][-1])

# =========================
# REPRESENTATIVE DATASET
# (NO NORMALIZATION!)
# =========================
def representative_data_gen():
    base_dir = Path(DATASET_DIR)

    cloudy_imgs = list((base_dir / "cloudy").glob("*.png"))[:50]
    clear_imgs  = list((base_dir / "clear").glob("*.png"))[:50]

    imgs = cloudy_imgs + clear_imgs

    for img_path in imgs:
        img = Image.open(img_path).resize((IMG_SIZE, IMG_SIZE))
        img = np.array(img, dtype=np.float32)  # 0–255 RGB
        img = np.expand_dims(img, axis=0)
        yield [img]

# =========================
# TFLITE INT8 CONVERSION
# (INT8 input, FLOAT output)
# =========================
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen

converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS_INT8
]

converter.inference_input_type = tf.int8
converter.inference_output_type = tf.float32

tflite_model = converter.convert()

# =========================
# SAVE MODEL
# =========================
MODEL_FILE = "cloud_vs_clear_48x48_int8.tflite"
with open(MODEL_FILE, "wb") as f:
    f.write(tflite_model)

print("\nSaved:", MODEL_FILE)
print("Model size:", os.path.getsize(MODEL_FILE), "bytes")
