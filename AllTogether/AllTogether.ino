#include <Arduino_OV767X.h>
#include "model.h"

#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/tflite_bridge/micro_error_reporter.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include "version.h"


// NOT COMPATIBLE WITH QUANTISATION YET


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
constexpr int tensorArenaSize = 120 * 1024;
byte tensorArena[tensorArenaSize] __attribute__((aligned(16)));

// array to map weather conditions index to a name
const char* WEATHERS[] = {
  "clear",
  "cloudy"
};

unsigned short pixels[176 * 144]; // QCIF: 176x144 X 2 bytes per pixel (RGB565)

void crop_pad_convert_image(unsigned short *image, TfLiteTensor* input, int image_size) {
    int added = 0;
    int i;
    for (i=0;i<1350;i++) {
        input->data.int8[added] = 0x00;
        added++;
    }
    
    for (i=0;i<image_size;i++) {
        int pos = (i % 176);
        if (pos >= 13 && pos < 163) {
          unsigned char first_byte = image[i] & 0xFF;
          unsigned char second_byte = (image[i] & 0xFF00) >> 8;

          unsigned char R5 = (first_byte & 0xF8) >> 3;
          unsigned char G6 = ((first_byte & 0x07) << 3) | ((second_byte & 0xE0) >> 5);
          unsigned char B5 = (second_byte & 0x1F);

          // Convert to RGB888
          unsigned char new_first_byte  = (R5 << 3) | (R5 >> 2);     // R 0–255
          unsigned char new_second_byte = (G6 << 2) | (G6 >> 4);     // G 0–255
          unsigned char new_third_byte  = (B5 << 3) | (B5 >> 2);     // B 0–255
          
          input->data.int8[added] = new_first_byte - 128;
          added++;
          input->data.int8[added] = new_second_byte - 128;
          added++;
          input->data.int8[added] = new_third_byte - 128;
          added++;
        }
    }
    for (i=0;i<1350;i++) {
        input->data.int8[added] = 0x00;
        added++;
    }
    Serial.println(added);
}

void setup() {
  Serial.begin(9600);
  while (!Serial);

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
  tflInterpreter->AllocateTensors();

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

  // Get pointers for the model's input and output tensors
  tflInputTensor = tflInterpreter->input(0);
  tflOutputTensor = tflInterpreter->output(0);


  if (!Camera.begin(QCIF, RGB565, 1)) {
    Serial.println("Failed to initialize camera!");
    while (1);
  }
}

void loop() {
  if (Serial.read() == 'c') {
    Camera.readFrame(pixels);

    int numPixels = Camera.width() * Camera.height();

    crop_pad_convert_image(pixels, tflInputTensor, numPixels);

    // Run inferencing
    TfLiteStatus invokeStatus = tflInterpreter->Invoke();
    if (invokeStatus != kTfLiteOk) {
      Serial.println("Invoke failed!");
      while (1);
      return;
    }

    // Loop through the output tensor values from the model
    for (int i = 0; i < 2; i++) {
      int8_t value = tflOutputTensor->data.int8[i];
      float score = (value - tflOutputTensor->params.zero_point) *
                    tflOutputTensor->params.scale;

      Serial.print(WEATHERS[i]);
      Serial.print(": ");
      Serial.println(score, 6);
    }
    Serial.println();

  }
}
