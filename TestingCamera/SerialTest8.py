import serial
import time

# =======================
# CONFIGURATION
# =======================
SERIAL_PORT = "COM9"        # Replace with your Arduino COM port
BAUD_RATE = 115200
OUTPUT_FILE = "camera_output.raw"
READ_TIMEOUT = 10           # seconds

# =======================
# MAIN
# =======================
try:
    # Open serial connection
    with serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1) as ser:
        print(f"Connected to {SERIAL_PORT} at {BAUD_RATE} baud.")

        # Wait a short moment for Arduino reset
        time.sleep(2)

        # Flush any existing data
        ser.reset_input_buffer()

        # Send the 'c' character to Arduino
        ser.write(b'c')
        print("Sent 'c' to Arduino, waiting for frame...")

        hex_data = ""
        start_time = time.time()

        # Read until timeout
        while True:
            if time.time() - start_time > READ_TIMEOUT:
                print("Timeout reached, stopping read.")
                break

            line = ser.readline().decode(errors='ignore')
            if line:
                hex_data += line

        # Save to file
        with open(OUTPUT_FILE, "w") as f:
            f.write(hex_data)

        print(f"Saved output to {OUTPUT_FILE}")

except serial.SerialException as e:
    print(f"Serial error: {e}")
