import serial
import numpy as np
import cv2

# Change this to your Arduino's serial port
# Example: "COM5" on Windows, or "/dev/ttyACM0" on Linux/Mac
port = "COM9"

ser = serial.Serial(port, 115200, timeout=5)
print("Waiting for image...")

width, height = None, None
image_data = bytearray()
started = False

while True:
    line = ser.readline().decode(errors='ignore').strip()
    if line.startswith("WIDTH:"):
        width = int(line.split(":")[1])
    elif line.startswith("HEIGHT:"):
        height = int(line.split(":")[1])
    elif line == "BEGIN_IMAGE":
        started = True
        image_data = bytearray()
    elif line == "END_IMAGE":
        break
    elif started:
        image_data.extend(line.encode())

print("Image received!")

# Convert to NumPy array
img = np.frombuffer(image_data, dtype=np.uint8)
img = img.reshape((height, width, 2))

# Convert from RGB565 to BGR888 (for OpenCV display)
bgr = cv2.cvtColor(img, cv2.COLOR_BGR5652BGR)

cv2.imshow("Arduino Camera", bgr)
cv2.waitKey(0)
cv2.destroyAllWindows()
