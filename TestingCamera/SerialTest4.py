import serial
import numpy as np
from PIL import Image

# --- CONFIGURATION ---
PORT = "COM9"          # change if needed
BAUD = 1000000          # must match Serial.begin()
WIDTH, HEIGHT = 320, 240
# ----------------------

BYTES_PER_PIXEL = 2
FRAME_SIZE = WIDTH * HEIGHT * BYTES_PER_PIXEL

# Binary start/end markers
START_MARKER = b'\xFF\xD8'
END_MARKER = b'\xFF\xD9'

def rgb565_to_rgb888(data):
    """Convert RGB565 bytes to RGB888 NumPy array."""
    arr = np.frombuffer(data, dtype=np.uint16).reshape((HEIGHT, WIDTH))
    r = ((arr >> 11) & 0x1F) << 3
    g = ((arr >> 5) & 0x3F) << 2
    b = (arr & 0x1F) << 3
    rgb = np.dstack((r, g, b)).astype(np.uint8)
    return rgb

with serial.Serial(PORT, BAUD, timeout=2) as ser:
    print(f"Listening on {PORT} at {BAUD} baud...")
    buffer = bytearray()
    recording = False

    while True:
        byte = ser.read(1)
        if not byte:
            continue
        buffer += byte

        # Detect start marker
        if not recording and START_MARKER in buffer:
            # Trim everything before the marker
            start_index = buffer.find(START_MARKER)
            buffer = buffer[start_index + len(START_MARKER):]
            recording = True
            print("📸 Frame started")

        # Detect end marker
        if recording and END_MARKER in buffer:
            end_index = buffer.find(END_MARKER)
            frame_data = buffer[:end_index]
            print(f"✅ Got frame ({len(frame_data)} bytes)")

            # Save the raw data
            with open("frame.raw", "wb") as f:
                f.write(frame_data)

            # Convert and display if correct size
            #if len(frame_data) == FRAME_SIZE:
                img = rgb565_to_rgb888(frame_data)
                Image.fromarray(img).show()
            #else:
            #    print(f"⚠️ Expected {FRAME_SIZE} bytes, got {len(frame_data)}")

            # Reset buffer for next frame
            buffer = buffer[end_index + len(END_MARKER):]
            recording = False

"""
Capturing at QVGA = 320 x 240

Using RGB565 format → 2 bytes per pixel

So each frame = 320 x 240 x 2 = 153,600 bytes
"""