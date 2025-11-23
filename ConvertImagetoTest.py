from PIL import Image
import struct

def rgb888_to_rgb565(r, g, b):
    """Convert 24-bit RGB888 to 16-bit RGB565."""
    r5 = (r >> 3) & 0x1F
    g6 = (g >> 2) & 0x3F
    b5 = (b >> 3) & 0x1F
    return (r5 << 11) | (g6 << 5) | b5


def convert_image(input_path, output_path):
    # Load image
    img = Image.open(input_path).convert("RGB")

    # Resize to 176×144
    img_resized = img.resize((176, 144), Image.BILINEAR)

    # Convert to RGB565 and store the bytes
    rgb565 = bytearray()
    for y in range(img_resized.height):
        for x in range(img_resized.width):
            r, g, b = img_resized.getpixel((x, y))
            value = rgb888_to_rgb565(r, g, b)
            rgb565 += struct.pack("<H", value)

    # Write as a .h file
    with open(output_path, "w") as h:
        h.write("const unsigned char image[] PROGMEM = {\n")

        for i, b in enumerate(rgb565):
            h.write(f"0x{b:02X}, ")
            if (i + 1) % 16 == 0:
                h.write("\n")

        h.write("};\n\n")
        h.write(f"const unsigned int image_len = {len(rgb565)};\n")

    print(f"Saved header file: {output_path}")


if __name__ == "__main__":
    convert_image(
        r"C:\\Cambridge\\Societies\\HABS\\dataset\\cloudy\\B_img13.png",
        "image.h"
    )
