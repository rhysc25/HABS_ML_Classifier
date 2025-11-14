import serial
import numpy as np
from PIL import Image

# --- CONFIGURATION ---
PORT = "COM9"          # change if needed
BAUD = 115200          # must match Serial.begin()
WIDTH, HEIGHT = 320, 240
# ----------------------

BYTES_PER_PIXEL = 2
FRAME_SIZE = WIDTH * HEIGHT * BYTES_PER_PIXEL

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

        # detect start marker
        if not recording and b"<START>" in buffer:
            buffer = bytearray()  # clear everything before start
            recording = True
            print("📸 Frame started")

        # detect end marker
        if recording and b"<END>" in buffer:
            frame_end = buffer.find(b"<END>")
            frame_data = buffer[:frame_end]
            print(f"✅ Got frame ({len(frame_data)} bytes)")

            # save raw data
            with open("frame.raw", "wb") as f:
                f.write(frame_data)

            # convert + show image
            if len(frame_data) == FRAME_SIZE:
                img = rgb565_to_rgb888(frame_data)
                Image.fromarray(img).show()
            else:
                print(f"⚠️ Expected {FRAME_SIZE} bytes, got {len(frame_data)}")

            # reset for next frame
            recording = False
            buffer = bytearray()
