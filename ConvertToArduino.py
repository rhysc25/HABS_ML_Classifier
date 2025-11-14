import os

tflite_path = "cloud_classification_model_int8.tflite"
header_path = "model.h"

# Check file exists
if not os.path.exists(tflite_path):
    raise FileNotFoundError(f"{tflite_path} not found in: {os.getcwd()}")

# Read model bytes
with open(tflite_path, "rb") as f:
    data = f.read()

# Write header file
with open(header_path, "w") as f:
    f.write("const unsigned char model[] = {")

    for i, b in enumerate(data):
        if i % 12 == 0:
            f.write("\n  ")
        f.write(f"0x{b:02x}, ")

    f.write("\n};\n")
    f.write(f"const unsigned int model_len = {len(data)};\n")

print(f"Created model.h ({os.path.getsize(header_path)} bytes)")
