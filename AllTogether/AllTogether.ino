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

unsigned short pixels[176 * 144]; // QCIF: 176x144 X 2 bytes per pixel (RGB565)
unsigned char new_image[67500];

void crop_pad_convert_image(unsigned short *image, unsigned char *new_image, int image_size) {
    int added = 0;
    int i;
    for (i=0;i<1350;i++) {
        new_image[added] = 0x00;
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
          
          new_image[added] = new_first_byte;
          added++;
          new_image[added] = new_second_byte;
          added++;
          new_image[added] = new_third_byte;
          added++;
        }
    }
    for (i=0;i<1350;i++) {
        new_image[added] = 0x00;
        added++;
    }
    Serial.println(added);
}

void setup() {
  Serial.begin(9600);
  while (!Serial);

  if (!Camera.begin(QCIF, RGB565, 1)) {
    Serial.println("Failed to initialize camera!");
    while (1);
  }
}

void loop() {
  if (Serial.read() == 'c') {
    Camera.readFrame(pixels);

    int numPixels = Camera.width() * Camera.height();

    crop_pad_convert_image(pixels, new_image, numPixels);
  }
}
