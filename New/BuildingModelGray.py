import tensorflow as tf
import numpy as np
from pathlib import Path
from PIL import Image
import os

DATASET_DIR = "C:/Cambridge/Societies/HABS/dataset"
IMG_SIZE = 48
BATCH_SIZE = 32
EPOCHS = 15
SEED = 123

# =========================
# MODEL
# =========================
model = tf.keras.Sequential([
    tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 1)),

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

# =========================
# DATASETS
# =========================
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    color_mode="grayscale"
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    color_mode="grayscale"
)

train_ds = train_ds.cache().shuffle(500).prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.cache().prefetch(tf.data.AUTOTUNE)

history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
print("latest accuracy on unseen images:")
print(val_acc[-1])

import numpy as np
# Create a single 48x48 grayscale image full of ones
x = np.ones((1, 48, 48, 1), dtype=np.float32)
# Run inference
pred = model.predict(x)
print(pred)
print("Predicted class:", np.argmax(pred))

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open("cloud_vs_clear_48x48_int8.tflite", "wb") as f:
    f.write(tflite_model)

print("Model size:", os.path.getsize("cloud_vs_clear_48x48_int8.tflite"))
