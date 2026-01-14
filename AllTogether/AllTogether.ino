#include <Arduino_OV767X.h>
#include "model.h"

#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/tflite_bridge/micro_error_reporter.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include "version.h"


// global variables used for TensorFlow Lite (Micro)
tflite::MicroErrorReporter tflErrorReporter;
// pull in all the TFLM ops, can remove line and only pull in the TFLM ops you need, if need to reduce compiled size.
tflite::AllOpsResolver tflOpsResolver;

const tflite::Model* tflModel = nullptr;
tflite::MicroInterpreter* tflInterpreter = nullptr;
TfLiteTensor* tflInputTensor = nullptr;
TfLiteTensor* tflOutputTensor = nullptr;

// Create a static memory buffer for TFLM, the size may need to
// be adjusted based on the model you are using
constexpr int tensorArenaSize = 40 * 1024;
byte tensorArena[tensorArenaSize] __attribute__((aligned(16)));

// array to map weather conditions index to a name
const char* WEATHERS[] = {
  "clear",
  "cloudy"
};

unsigned short pixels[176 * 144]; // QCIF: 176x144 X 2 bytes per pixel (RGB565)

void preprocess(const uint16_t* src, TfLiteTensor* input) {
    const int SRC_W = 176;
    const int SRC_H = 144;
    const int DST_W = 48;
    const int DST_H = 48;

    float scale = input->params.scale;
    int zp = input->params.zero_point;

    int i = 0;

    for (int y = 0; y < DST_H; y++) {
        int sy = y * SRC_H / DST_H;

        for (int x = 0; x < DST_W; x++) {
            int sx = x * SRC_W / DST_W;

            uint16_t p = src[sy * SRC_W + sx];

            uint8_t r5 = (p >> 11) & 0x1F;
            uint8_t g6 = (p >> 5)  & 0x3F;
            uint8_t b5 =  p        & 0x1F;

            uint8_t R = (r5 << 3) | (r5 >> 2);
            uint8_t G = (g6 << 2) | (g6 >> 4);
            uint8_t B = (b5 << 3) | (b5 >> 2);

            input->data.int8[i++] = (int8_t)((R - zp) / scale);
            input->data.int8[i++] = (int8_t)((G - zp) / scale);
            input->data.int8[i++] = (int8_t)((B - zp) / scale);
        }
    }
}

void setup() {
  Serial.begin(9600);
  while (!Serial);

  Serial.println("Test");

  // get the TFL representation of the model byte array
  tflModel = tflite::GetModel(model);
  if (tflModel->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model schema mismatch!");
    while (1);
  }

  // Create an interpreter to run the model
  //tflInterpreter = new tflite::MicroInterpreter(tflModel, tflOpsResolver, tensorArena, tensorArenaSize, &tflErrorReporter);
  tflInterpreter = new tflite::MicroInterpreter(
    tflModel,
    tflOpsResolver,
    tensorArena,
    tensorArenaSize
  );

  // Allocate memory for the model's input and output tensors
  if (tflInterpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("AllocateTensors() failed");
    while (1);
  }

  // Get pointers for the model's input and output tensors
  tflInputTensor = tflInterpreter->input(0);
  tflOutputTensor = tflInterpreter->output(0);

  // For testing
  Serial.print("Arena used: ");
  Serial.println(tflInterpreter->arena_used_bytes());

  Serial.print("Input tensor bytes: ");
  Serial.println(tflInputTensor->bytes);

  Serial.print("Input dims: ");
  for (int i=0; i<tflInputTensor->dims->size; i++) {
    Serial.print(tflInputTensor->dims->data[i]);
    Serial.print(" ");
  }
  Serial.println();

  if (!Camera.begin(QCIF, RGB565, 1)) {
    Serial.println("Failed to initialize camera!");
    while (1);
  }
  // Increase brightness & contrast aggressively
  Camera.setBrightness(2);   // -2 to +2
  Camera.setContrast(2);     // -2 to +2
  Camera.setSaturation(2);   // -2 to +2

  // Boost analog gain
  Camera.setGain(7);         // 0–7 (7 = strongest gain)
}

void loop() {
  if (Serial.read() == 'c') {
    Camera.readFrame(pixels);

    int numPixels = Camera.width() * Camera.height();

    preprocess(pixels, tflInputTensor);

    // Run inferencing
    TfLiteStatus invokeStatus = tflInterpreter->Invoke();
    if (invokeStatus != kTfLiteOk) {
      Serial.println("Invoke failed!");
      while (1);
      return;
    }

    // Loop through the output tensor values from the model
    for (int i = 0; i < 2; i++) {
      float score = tflOutputTensor->data.f[i]; 

      Serial.print(WEATHERS[i]);
      Serial.print(": ");
      Serial.println(score, 6);
    }
    Serial.println();

  }
}
