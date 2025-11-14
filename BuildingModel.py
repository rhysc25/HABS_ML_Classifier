"""
Develop CNN classification model using 150x150 pngs (RGB888)
Convert to TFlite, upload onto Arduino nano 33 BLE sense 
Using OV767x camera
Capturing QCIF: 176x144 X 2 bytes per pixel (RGB565)
Crop to 144x144 on board
Scale to 150x150
Feed to model, serial output 
"""

import tensorflow as tf
import numpy as np
from pathlib import Path
from PIL import Image


model = tf.keras.Sequential([
    tf.keras.Input(shape=(150,150,3)), # Defined the input shape for out image
    tf.keras.layers.Rescaling(scale=1./255), # Ensures all RGB values are between 0 and 1
    tf.keras.layers.Conv2D(filters=8, kernel_size=(3,3), activation="relu"), # Uses 8 3x3 filters for edge detection
    # ReLU removes negative values. Using for basic edge detection
    tf.keras.layers.MaxPooling2D(), #This takes the feature map and performs either min, max or average
    # on all the cells with the ones around it.
    tf.keras.layers.Conv2D(filters=16, kernel_size=(3,3), activation="relu"), # Deeper combing for features
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), activation="relu"), # Even deeper combing for features
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(), 
    tf.keras.layers.Dense(32, activation='relu'), 
    tf.keras.layers.Dense(2, activation='softmax')
]
)

train_dataset = tf.keras.utils.image_dataset_from_directory(
    directory = "C:\Cambridge\Societies\HABS\dataset",
    labels = "inferred",
    label_mode='int',
    color_mode='rgb',
    batch_size=32,
    image_size=(150,150),
    shuffle=True,
    seed = 123,
    validation_split=0.2,
    subset = "training",
    verbose=True
)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    directory = "C:\Cambridge\Societies\HABS\dataset",
    labels = "inferred",
    label_mode='int',
    color_mode='rgb',
    batch_size=32,
    image_size=(150,150),
    shuffle=True,
    seed = 123,
    validation_split=0.2,
    subset = "validation",
    verbose=True
)

# Configure datasets for performance
AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
# Keeps the images in memory after they're loaded off disk during the first epoch.
validation_dataset = validation_dataset.cache().prefetch(buffer_size=AUTOTUNE)
# Overlaps data preprocessing and model execution while training.

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])
# Adam is an optimization algorithm used during training to adjust the model’s weights.
# SpareCategoricalCrossentropy is Correct if you output raw logits with 2 output neurons.

model.summary()

epochs=10
history = model.fit(
  train_dataset,
  validation_data=validation_dataset,
  epochs=epochs
)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

print("latest accuracy on unseen images:")
print(val_acc[-1])

# In order to quantise model to fit within memory constraints of Arduino Nano 33 BLE
def representative_data_gen():
    base_dir = Path("C:\Cambridge\Societies\HABS\dataset")
    cloudy_imgs = list((base_dir / "cloudy").glob("*.png"))
    clear_imgs = list((base_dir / "clear").glob("*.png"))

    # Combine & take 100 mixed images
    imgs = cloudy_imgs[:50] + clear_imgs[:50]

    for img_path in imgs:
        img = Image.open(img_path).resize((150, 150))
        img = np.array(img, dtype=np.float32) / 255.0
        img = np.expand_dims(img, axis=0)
        yield [img]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
# Enable integer quantization
converter.optimizations = [tf.lite.Optimize.DEFAULT]
# Provide representative data
converter.representative_dataset = representative_data_gen
# Required for microcontrollers
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
# Force input and output to int8
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_model = converter.convert()
# Save the model to disk
open("cloud_classification_model_int8.tflite", "wb").write(tflite_model)
import os
basic_model_size = os.path.getsize("cloud_classification_model_int8.tflite")
print("Model is %d bytes" % basic_model_size)

