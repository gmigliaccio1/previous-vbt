import sys
import os
from datetime import datetime
import serial

ser = serial.Serial()
ser.port = sys.argv[1]
ser.baudrate = 115200
ser.timeout = 5
ser.dtr = False
ser.rts = False
ser.open()

def read_until_end():
    lines = []
    while True:
        line = ser.readline().decode(errors="ignore").strip()
        if line in ("END", ""):
            return lines
        lines.append(line)

out_dir = os.path.join("data", "raw", datetime.now().strftime("%Y-%m-%d_%H%M%S"))
os.makedirs(out_dir, exist_ok=False)

ser.reset_input_buffer()
ser.write(b"ls\n")
files = [l.split()[1] for l in read_until_end() if l.startswith("FILE")]

for name in files:
    ser.write(f"dump {name}\n".encode())
    lines = read_until_end()
    if lines and lines[0] == "BEGIN":
        out = os.path.join(out_dir, name.replace(".bin", ".csv"))
        if os.path.exists(out):
            raise SystemExit(f"refusing to overwrite {out}")
        with open(out, "w") as f:
            f.write("\n".join(lines[1:]) + "\n")
        print(out, len(lines) - 2, "samples")