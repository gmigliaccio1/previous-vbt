import sys
import serial

ser = serial.Serial()
ser.port = sys.argv[1]
ser.baudrate = 115200
ser.timeout = 5
ser.dtr = False
ser.rts = False
ser.open()

ser.reset_input_buffer()
ser.write(b"rmall\n")

while True:
    line = ser.readline().decode(errors="ignore").strip()
    if line == "END" or line == "":
        break
    print(line)

print("done -- device storage cleared")
ser.close()