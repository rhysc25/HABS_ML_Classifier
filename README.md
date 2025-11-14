# Cloud Classification on Arduino Nano 33 BLE Sense

CNN + TFLite Micro + OV767x Camera

## Overview

This project trains a lightweight convolutional neural network (CNN) to
classify images of clouds ("clear" vs "cloudy") on the **Arduino Nano 33
BLE Sense** using the **OV767x camera**. Because the board has only **1
MB flash** and **256 KB RAM**, the model must be aggressively
optimized---primarily through **full integer quantization**---to run
successfully on-device.

This repository documents the full pipeline:

-   Training a TensorFlow CNN using 150×150 RGB images.
-   Converting the model to a fully--int8 quantized TensorFlow Lite
    model.
-   Exporting the `.tflite` file as a C header for Arduino.
-   Running the model on the Nano 33 BLE Sense with live inference.

------------------------------------------------------------------------

## Hardware Constraints and Challenges

### 1. Flash Memory Limitations

The Arduino Nano 33 BLE Sense provides: - **1 MB of flash**\
- **\~800--900 KB usable once the bootloader and application overhead
are accounted for**

Early versions of the model produced `.tflite` files between **1.2--2.0
MB**, which would not fit on the device.

#### Solution: Full Integer Quantization

Applying: - `optimizations = [tf.lite.Optimize.DEFAULT]` -
Representative dataset -
`target_spec.supported_ops = [TFLITE_BUILTINS_INT8]` - Forcing **int8
input & output**

reduced the model size dramatically---down to a range that fits on the
microcontroller.

------------------------------------------------------------------------

## 2. RAM Limitations (Tensor Arena)

Microcontrollers must allocate a fixed "tensor arena" in RAM.\
Unquantized models required **hundreds of kilobytes**, exceeding
available memory.

### Solution: Reduce Activation Size

Using int8 quantization also shrinks activation tensors by 4×.\
This makes the inference RAM footprint small enough for TFLite Micro.

------------------------------------------------------------------------

## 3. Camera Input Format Mismatch

The OV767x outputs: - **QCIF 176×144** - **RGB565 (2 bytes/pixel)**

The model expects: - **150×150** - **RGB888 (3 bytes/pixel)**

### Solution on Device

-   Capture **176×144**
-   Crop to **144×144**
-   Resize to **150×150**
-   Convert RGB565 → RGB888
-   Normalize/scale appropriately for the int8 model

No retraining is needed as long as preprocessing on microcontroller
matches preprocessing during model export.

------------------------------------------------------------------------

## Training Pipeline

The TensorFlow model is a lightweight CNN:

-   Input: **150×150×3**
-   Three convolution + pooling blocks
-   One dense hidden layer
-   Softmax output for two classes

Data is loaded using:

``` python
tf.keras.utils.image_dataset_from_directory(...)
```

After training, an integer-quantized TFLite model is generated using a
representative dataset of 100 images.

------------------------------------------------------------------------

## Running on Arduino

The workflow on the microcontroller:

1.  Include the generated `model.h`.
2.  Set up TFLite Micro interpreter.
3.  Allocate tensor arena.
4.  Capture + preprocess camera frame.
5.  Run inference.
6.  Print classification result via Serial.

A typical output:

    clear: 0.87
    cloudy: 0.13
    Prediction: clear

------------------------------------------------------------------------

## Files in This Repository

-   `BuildingModel.py` --- Complete training + conversion script.

-   `cloud_classification_model_int8.tflite` --- Quantized model.

-   `model.h` --- C array for embedding on Arduino.

-   `arduino_inference/` --- Code for Nano 33 BLE Sense.

-   `dataset/` --- Training images in folder structure:

        dataset/clear/*.png
        dataset/cloudy/*.png

------------------------------------------------------------------------

## Key Learnings

-   Quantization is essential for microcontroller deployment.
-   The biggest challenges are memory constraints---not accuracy.
-   Preprocessing strategy must be consistent between Python/TFLite and
    Arduino.
