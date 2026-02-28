#include <Arduino_OV767X.h>
#include "model.h"

#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/tflite_bridge/micro_error_reporter.h>
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
unsigned short pixels[176 * 144];

const char* LABELS[] = {
  "clear",
  "cloudy"
};


void preprocess(const uint16_t* src, TfLiteTensor* input) {
  float* dst = input->data.f;

  int numPixels = input->dims->data[1] *
                  input->dims->data[2] *
                  input->dims->data[3];

  for (int i = 0; i < numPixels; i++) {
    // TEMP TEST: random noise in [0,1]
    dst[i] = 1.0;

  }
}

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

  Serial.print("Input type: ");
  Serial.println(inputTensor->type);
  Serial.print("Output type: ");
  Serial.println(outputTensor->type);

  Serial.println("Ready — send 'c' to capture");
}

void loop() {
  if (Serial.read() == 'c') {

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
