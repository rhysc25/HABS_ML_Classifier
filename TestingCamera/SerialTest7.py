import numpy as np
import matplotlib.pyplot as plt

WIDTH, HEIGHT = 176, 144

def hex_text_to_rgb888(hex_text):
    """Convert hex text from Arduino RGB565 output to RGB888 array."""
    # Remove whitespace/newlines
    hex_text = ''.join(hex_text.split())

    num_pixels = WIDTH * HEIGHT
    pixels = []
    
    for i in range(num_pixels):
        hex_pixel = hex_text[i*4:i*4+4]  # 4 chars per pixel
        if len(hex_pixel) < 4:
            hex_pixel = '0000'  # pad missing pixels
        val = int(hex_pixel, 16)
        pixels.append(val)
    
    arr = np.array(pixels, dtype=np.uint16).reshape((HEIGHT, WIDTH))
    
    # OV7670 is little endian
    arr = arr.byteswap()
    
    # Convert RGB565 → RGB888
    r = ((arr >> 11) & 0x1F) << 3
    g = ((arr >> 5) & 0x3F) << 2
    b = (arr & 0x1F) << 3

    rgb = np.dstack((r, g, b)).astype(np.uint8)
    return rgb

# Load the hex file
with open("camera_output.raw", "r") as f:
    hex_data = f.read()

image = hex_text_to_rgb888(hex_data)
plt.imshow(image)
plt.axis('off')
plt.show()
