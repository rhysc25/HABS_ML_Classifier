// Weather Classifier Set Up ///////////////////////////////////////////////////////////////////////////////////

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
uint8_t newpixels[48*48];

int classifierOKFlag = true;

// Persistent Storage Set Up ///////////////////////////////////////////////////////////////////////////////////

#include <Arduino.h>
#include "kvstore_global_api.h"

struct DataRecord {
  uint32_t timestamp;   // milliseconds since boot
  float pressure;   // in kPa
  float temperature;  // in °C
  float ax;
  float ay;
  float az;
  float weather;         // A measurement of how likely it is to be cloudy
};

uint32_t starttime;
int recordCount;
int status;

// For Gyroscope, Barometer and Thermometer

#include <Arduino_LSM9DS1.h>
#include <Arduino_LPS22HB.h>

float ax, ay, az;
int sensorsOKFlag = true;

// Classifier Preprocess Functions ///////////////////////////////////////////////////////////////////////////////////

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

// Everything Set Up ///////////////////////////////////////////////////////////////////////////////////

void setup() {

  if (!IMU.begin()) {
    sensorsOKFlag = false;
  }

  if (!BARO.begin()) {
    sensorsOKFlag = false;
  }

  if (!Camera.begin(QQVGA, GRAYSCALE, 1, OV7675)) {
    classifierOKFlag = false;
  }

  // Load model
  modelTf = tflite::GetModel(model);
  if (modelTf->version() != TFLITE_SCHEMA_VERSION) {
    classifierOKFlag = false;
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
    classifierOKFlag = false;
  }

  inputTensor = interpreter->input(0);
  outputTensor = interpreter->output(0);


  recordCount = 0;
  starttime = millis();
  status = kv_get("record_count", &recordCount, sizeof(recordCount), 0);
  if (status != 0) {
    recordCount = 0;
  }

  Serial.begin(115200);

  unsigned long t = millis();
  bool serialConnected = false;

  while (millis() - t < 10000) {  // wait up to 10 seconds
    if (Serial) {
      serialConnected = true;
      break;
    }
  }

  delay(30000);

  if (serialConnected) {
    Serial.print("Total records: ");
    Serial.println(recordCount);

    for (int i = 0; i < recordCount; i++) {

      char key[16];
      sprintf(key, "log_%03d", i);

      DataRecord record;
      size_t actualSize;

      if (kv_get(key, &record, sizeof(record), &actualSize) == 0) {
        Serial.print(record.timestamp);
        Serial.print(",");
        Serial.print(record.pressure);
        Serial.print(",");
        Serial.print(record.temperature);
        Serial.print(",");
        Serial.print(record.ax);
        Serial.print(",");
        Serial.print(record.ay);
        Serial.print(",");
        Serial.print(record.az);
        Serial.print(",");
        Serial.println(record.weather);
      } else {
        Serial.print("Initial stored data failed to be retrieved");
      }
    }
  } 

}


void loop() {

  delay(10000);
  
  DataRecord record;

  Camera.readFrame(pixels);

  resizeQQVGAto48(pixels, newpixels);
  inputFill(newpixels, inputTensor);

  if (interpreter->Invoke() != kTfLiteOk) {
    return;
  }

  // Accelerometer
  if (IMU.accelerationAvailable()) {
    IMU.readAcceleration(ax, ay, az);
  }

  // Pressure
  float pressure = BARO.readPressure();       // in kPa
  float temperature = BARO.readTemperature(); // in °C

  record.timestamp = millis() - starttime;
  record.pressure = pressure;
  record.temperature = temperature;
  record.ax = ax;
  record.ay = ay;
  record.az = az;
  record.weather = outputTensor->data.f[1];;

  char key[16];
  sprintf(key, "log_%03d", recordCount);

  status = kv_set(key, &record, sizeof(record), 0);

  if (status == 0) {
    recordCount++;
    status = kv_set("record_count", &recordCount, sizeof(recordCount), 0);

    if (status != 0) {
      while (1);
    } 
  }
}
