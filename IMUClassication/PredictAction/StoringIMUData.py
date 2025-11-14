import serial, csv
ser = serial.Serial("COM9", 9600)
with open("flex.csv","w",newline="") as f:
    w = csv.writer(f)
    w.writerow(["aX","aY","aZ","gX","gY","gZ"])
    while True:
        try:
            line = ser.readline().decode().strip()
            if line:
                w.writerow(line.split(","))
                print(line)
        except KeyboardInterrupt:
            break