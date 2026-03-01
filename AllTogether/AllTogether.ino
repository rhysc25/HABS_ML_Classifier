//#include <Arduino_OV767X.h>
#include <TinyMLShield.h>
#include "model.h"

#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
//#include <tensorflow/lite/micro/tflite_bridge/micro_error_reporter.h>
#include <tensorflow/lite/micro/micro_error_reporter.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include "version.h"

tflite::MicroErrorReporter errorReporter;
tflite::AllOpsResolver resolver;

const tflite::Model* modelTf = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;

TfLiteTensor* inputTensor = nullptr;
TfLiteTensor* outputTensor = nullptr;

constexpr int kTensorArenaSize = 120 * 1024;
uint8_t tensorArena[kTensorArenaSize] __attribute__((aligned(16)));
uint8_t pixels[160 * 120];
uint8_t newpixels[48*48];

const char* LABELS[] = {
  "clear",
  "cloudy"
};

void resizeQQVGAto48(const uint8_t* input, uint8_t* output) {

  const int inW = 160;
  const int inH = 120;

  const int scaledH = 36;
  const float scale = 160.0f / 48.0f;  // 3.3333

  for (int y = 0; y < scaledH; y++) {
    for (int x = 0; x < 48; x++) {

      float gx = x * scale;
      float gy = y * scale;

      int x0 = (int)gx;
      int y0 = (int)gy;
      int x1 = min(x0 + 1, inW - 1);
      int y1 = min(y0 + 1, inH - 1);

      float dx = gx - x0;
      float dy = gy - y0;

      uint8_t p00 = input[y0 * inW + x0];
      uint8_t p10 = input[y0 * inW + x1];
      uint8_t p01 = input[y1 * inW + x0];
      uint8_t p11 = input[y1 * inW + x1];

      float val =
        p00 * (1 - dx) * (1 - dy) +
        p10 * dx * (1 - dy) +
        p01 * (1 - dx) * dy +
        p11 * dx * dy;

      int outY = y + 6; // padding top
      output[outY * 48 + x] = (uint8_t)val;
    }
  }

  // Zero top and bottom padding
  for (int i = 0; i < 6 * 48; i++) {
    output[i] = 0;
    output[(42 * 48) + i] = 0;
  }
}

void inputFill(const uint8_t* newpixels, TfLiteTensor* input){
  float* dst = input->data.f;
  
  for (int i=0; i<48*48; i++) {
    dst[i] = newpixels[i];
  }
}

void setup() {
  Serial.begin(115200);
  while (!Serial);

  Serial.println("\nCloud vs Clear — TinyML");

  if (!Camera.begin(QQVGA, GRAYSCALE, 1, OV7675)) {
    Serial.println("Failed to initialize camera!");
    while (1);
  }

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
    kTensorArenaSize,
    &errorReporter,
    nullptr
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

  Serial.print("Input type: ");
  Serial.println(inputTensor->type);
  Serial.print("Output type: ");
  Serial.println(outputTensor->type);

  Serial.println("Ready — send 'c' to capture");
}

void loop() {
  if (Serial.read() == 'c') {

    Camera.readFrame(pixels);

    resizeQQVGAto48(pixels, newpixels);
    inputFill(newpixels, inputTensor);

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
