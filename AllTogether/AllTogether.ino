#include <Arduino_OV767X.h>
#include "model.h"

#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/tflite_bridge/micro_error_reporter.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include "version.h"

// ===============================
// TensorFlow Lite globals
// ===============================
tflite::MicroErrorReporter errorReporter;
tflite::AllOpsResolver resolver;

const tflite::Model* modelTf = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;

TfLiteTensor* inputTensor = nullptr;
TfLiteTensor* outputTensor = nullptr;

// ===============================
// Tensor arena
// ===============================
constexpr int kTensorArenaSize = 40 * 1024;
uint8_t tensorArena[kTensorArenaSize] __attribute__((aligned(16)));

// ===============================
// Camera buffer (QCIF RGB565)
// ===============================
unsigned short pixels[176 * 144];

// ===============================
// Labels
// ===============================
const char* LABELS[] = {
  "clear",
  "cloudy"
};

// ===============================
// Image preprocessing
// ===============================
void preprocess(const uint16_t* src, TfLiteTensor* input) {
/*
  const int SRC_W = 176;
  const int SRC_H = 144;
  const int DST_W = 48;
  const int DST_H = 48;

  float scale = input->params.scale;
  int zp = input->params.zero_point;

  int idx = 0;

  for (int y = 0; y < DST_H; y++) {
    int sy = y * SRC_H / DST_H;

    for (int x = 0; x < DST_W; x++) {
      int sx = x * SRC_W / DST_W;
      uint16_t p = src[sy * SRC_W + sx];

      // RGB565 → RGB888
      uint8_t r5 = (p >> 11) & 0x1F;
      uint8_t g6 = (p >> 5)  & 0x3F;
      uint8_t b5 =  p        & 0x1F;

      uint8_t R = (r5 << 3) | (r5 >> 2);
      uint8_t G = (g6 << 2) | (g6 >> 4);
      uint8_t B = (b5 << 3) | (b5 >> 2);

      // Luminance (ITU-R BT.601)
      uint8_t Y = (77 * R + 150 * G + 29 * B) >> 8;

      float yf = Y / 255.0f;

      int32_t q = (int32_t)round(yf / scale) + zp;
      q = constrain(q, -128, 127);

      input->data.int8[idx++] = (int8_t)q;
    }
  }
  */

  for (int i = 0; i < inputTensor->bytes; i++) {
    inputTensor->data.int8[i] = random(-128, 127);
  }

}


// ===============================
// Setup
// ===============================
void setup() {
  Serial.begin(115200);
  while (!Serial);

  Serial.println("\nCloud vs Clear — TinyML");

  // Load model
  modelTf = tflite::GetModel(model);
  if (modelTf->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model schema mismatch!");
    while (1);
  }

  // Create interpreter
  interpreter = new tflite::MicroInterpreter(
    modelTf,
    resolver,
    tensorArena,
    kTensorArenaSize
  );

  // Allocate tensors
  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("AllocateTensors failed");
    while (1);
  }

  inputTensor = interpreter->input(0);
  outputTensor = interpreter->output(0);

  // Debug info
  Serial.print("Arena used: ");
  Serial.println(interpreter->arena_used_bytes());

  Serial.print("Input scale: ");
  Serial.println(inputTensor->params.scale, 6);
  Serial.print("Input zero point: ");
  Serial.println(inputTensor->params.zero_point);

  Serial.print("Input shape: ");
  for (int i = 0; i < inputTensor->dims->size; i++) {
    Serial.print(inputTensor->dims->data[i]);
    Serial.print(" ");
  }
  Serial.println();

  // Camera init
  if (!Camera.begin(QCIF, RGB565, 1)) {
    Serial.println("Camera init failed!");
    while (1);
  }

  Camera.setBrightness(2);
  Camera.setContrast(2);
  Camera.setSaturation(2);
  Camera.setGain(7);

  Serial.println("Ready — send 'c' to capture");
}

// ===============================
// Loop
// ===============================
void loop() {
  if (Serial.read() == 'c') {
    Camera.readFrame(pixels);

    preprocess(pixels, inputTensor);

    if (interpreter->Invoke() != kTfLiteOk) {
      Serial.println("Invoke failed");
      return;
    }

    for (int i = 0; i < 2; i++) {
      Serial.print(LABELS[i]);
      Serial.print(": ");
      Serial.println(outputTensor->data.f[i], 5);
    }
    Serial.println();
  }
}
