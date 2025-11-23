#include "image.h"

unsigned char new_image[67500];

void crop_pad_convert_image(const unsigned char *image, unsigned char *new_image, int image_size) {
    int added = 0;
    int i;
    for (i=0;i<1350;i++) {
        new_image[added] = 0x00;
        added++;
    }
    
    for (i=0;i<image_size;i++) {
        int pos = (i % 352);
        if (pos >= 26 && pos < 326 && pos%2==0) {
          unsigned char first_byte = image[i];
          unsigned char second_byte = image[i+1];

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
  // put your setup code here, to run once:
    Serial.begin(9600);
    while (!Serial);

    // print the header
    Serial.println("Welcome");

    crop_pad_convert_image(image, new_image, sizeof(image));
    Serial.println(sizeof(new_image));
}

void loop() {
  
}
