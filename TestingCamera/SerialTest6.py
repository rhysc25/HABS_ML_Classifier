import numpy as np
import matplotlib.pyplot as plt

# Image dimensions
WIDTH, HEIGHT = 176, 144

def rgb565_to_rgb888(data):
    """Convert RGB565 bytes to RGB888 NumPy array."""
    # Make sure the data length is correct
    expected_bytes = WIDTH * HEIGHT * 2
    if len(data) < expected_bytes:
        data += bytes(expected_bytes - len(data))  # pad with zeros if needed
    
    arr = np.frombuffer(data, dtype=np.uint16).reshape((HEIGHT, WIDTH))
    
    # OV7670 is Little Endian
    arr = arr.byteswap()  # Swap bytes if needed (RGB565 little endian)
    
    # Extract R, G, B components
    r = ((arr >> 11) & 0x1F) << 3
    g = ((arr >> 5) & 0x3F) << 2
    b = (arr & 0x1F) << 3
    
    rgb = np.dstack((r, g, b)).astype(np.uint8)
    return rgb

# Load the raw file
with open("frame2.raw", "rb") as f:
    raw_data = f.read()

# Convert and display
image = rgb565_to_rgb888(raw_data)
plt.imshow(image)
plt.axis('off')
plt.show()
