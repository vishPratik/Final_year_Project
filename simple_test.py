# simple_test.py
import socket
import numpy as np
from collections import deque

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(('0.0.0.0', 9002))
sock.settimeout(1.0)

buffer = deque(maxlen=250)  # 1 second of data at 250Hz

print("Waiting for ECG data...")
try:
    while True:
        try:
            data, addr = sock.recvfrom(1024)
            values = data.decode().strip().split(',')
            if len(values) == 3:
                ecg, lo_plus, lo_minus = map(int, values)
                buffer.append(ecg)
                print(f"ECG: {ecg}, LO+: {lo_plus}, LO-: {lo_minus}, Buffer: {len(buffer)}")
                
                if len(buffer) == 250:
                    print("Got 1 second of data! Streamlit should work now.")
                    break
                    
        except socket.timeout:
            print("Waiting for data...")
except KeyboardInterrupt:
    pass
finally:
    sock.close()