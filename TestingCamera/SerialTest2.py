import serial

PORT = "COM9"
BAUD = 115200
FILENAME = "frame.jpg"

with serial.Serial(PORT, BAUD, timeout=1) as ser:
    print("Listening...")
    buffer = bytearray()
    recording = False

    while True:
        byte = ser.read(1)
        if not byte:
            continue
        buffer += byte

        if b"<START>" in buffer:
            buffer = bytearray()  # clear everything before start
            recording = True
            print("Frame started.")

        if recording and b"<END>" in buffer:
            frame_end = buffer.find(b"<END>")
            image_data = buffer[:frame_end]
            with open(FILENAME, "wb") as f:
                f.write(image_data)
            print(f"Frame saved as {FILENAME}")
            recording = False
            buffer = bytearray()