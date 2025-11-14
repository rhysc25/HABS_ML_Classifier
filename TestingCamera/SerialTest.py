import serial

# --- CONFIGURATION ---
PORT = "COM9"      # Change this to your Arduino's port
BAUD = 115200        # Match your sketch's Serial.begin() baud rate
# ----------------------

try:
    with serial.Serial(PORT, BAUD, timeout=1) as ser:
        print(f"Connected to {PORT} at {BAUD} baud.\nPress Ctrl+C to stop.\n")
        while True:
            line = ser.readline().decode(errors='ignore').strip()
            if line:
                print(line)
except serial.SerialException as e:
    print(f"Serial error: {e}")
except KeyboardInterrupt:
    print("\nStopped by user.")
