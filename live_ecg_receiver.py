"""
Live ECG receiver for ESP8266 AD8232 -> Streamlit
- Starts a TCP server on 0.0.0.0:<PORT>
- Receives CSV lines: t_us,adc,lo_plus,lo_minus
- Maintains a ring buffer of raw ADC samples and derived volts
- Provides helpers to preprocess, filter, and return a fixed-length window
"""
import socket
import threading
import time
from collections import deque
import numpy as np
from typing import Optional, Deque

# Optional heavy deps (only used if available)
try:
    import scipy.signal as sps
except Exception:
    sps = None
try:
    import pywt
except Exception:
    pywt = None

DEFAULT_PORT = 9002
FS_HZ = 250               # must match firmware
ADC_RES = 1023.0          # 10-bit
VREF_NODEMCU = 3.3        # typical NodeMCU/D1 mini A0 full-scale
VREF_BARE = 1.0           # bare ESP8266

class ECGStreamServer:
    def __init__(self, host: str = "0.0.0.0", port: int = DEFAULT_PORT, 
                 fs_hz: int = FS_HZ, buffer_seconds: int = 20,
                 a0_fullscale_volts: float = VREF_NODEMCU):
        self.addr = (host, port)
        self.fs = fs_hz
        self.buffer_len = int(buffer_seconds * fs_hz)
        self.a0_fullscale_volts = float(a0_fullscale_volts)

        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind(self.addr)
        self._sock.listen(1)

        self.samples_adc: Deque[int] = deque(maxlen=self.buffer_len)
        self.samples_volts: Deque[float] = deque(maxlen=self.buffer_len)
        self.leadoff_plus: Deque[int] = deque(maxlen=self.buffer_len)
        self.leadoff_minus: Deque[int] = deque(maxlen=self.buffer_len)
        self.timestamps_us: Deque[int] = deque(maxlen=self.buffer_len)

        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._accept_loop, daemon=True)
        self._thread.start()
        
        print(f"ECG Server started on {host}:{port}")

    def _accept_loop(self):
        while not self._stop.is_set():
            try:
                conn, peer = self._sock.accept()
                conn.settimeout(2.0)
                print(f"Client connected: {peer}")
                self._client_loop(conn, peer)
            except Exception as e:
                if not self._stop.is_set():
                    print(f"Accept error: {e}")
                continue

    def _client_loop(self, conn: socket.socket, peer):
        print(f"Starting client loop for {peer}")
        try:
            while not self._stop.is_set():
                try:
                    # Read line from TCP connection
                    data = conn.recv(1024)
                    if not data:
                        print("Client disconnected")
                        break
                    
                    lines = data.decode().split('\n')
                    for line in lines:
                        line = line.strip()
                        if not line or line.startswith("#"):
                            continue
                            
                        # Parse: timestamp_us,adc,lo_plus,lo_minus
                        parts = line.split(",")
                        if len(parts) == 4:
                            try:
                                t_us = int(parts[0])
                                adc = int(parts[1])
                                lo_p = int(parts[2])
                                lo_m = int(parts[3])
                                
                                volts = (adc / ADC_RES) * self.a0_fullscale_volts
                                self.timestamps_us.append(t_us)
                                self.samples_adc.append(adc)
                                self.samples_volts.append(volts)
                                self.leadoff_plus.append(lo_p)
                                self.leadoff_minus.append(lo_m)
                                
                                # Debug: print first few samples
                                if len(self.samples_adc) < 5:
                                    print(f"Received: {t_us},{adc},{lo_p},{lo_m} -> {volts:.2f}V")
                                    
                            except ValueError as e:
                                print(f"Parse error: {line} -> {e}")
                                continue
                        else:
                            print(f"Invalid format: {line}")
                            
                except socket.timeout:
                    continue
                except Exception as e:
                    print(f"Receive error: {e}")
                    break
                    
        except Exception as e:
            print(f"Client loop error: {e}")
        finally:
            try:
                conn.close()
                print("Connection closed")
            except:
                pass

    def get_latest_window(self, n_samples: int) -> Optional[np.ndarray]:
        """Return the latest n_samples (volts) as numpy array or None if not enough."""
        if len(self.samples_volts) < n_samples:
            return None
        return np.array(list(self.samples_volts)[-n_samples:], dtype=np.float32)

    def stop(self):
        self._stop.set()
        try:
            self._sock.close()
        except:
            pass
        print("ECG Server stopped")

# --------- Signal processing helpers ---------
def bandpass_notch(x: np.ndarray, fs: int = FS_HZ) -> np.ndarray:
    """Bandpass 0.5-40 Hz + notch 50 Hz if SciPy is available; otherwise returns x."""
    if sps is None:
        return x
    
    y = x.copy()
    
    # Notch at 50 Hz; skip if fs<120
    if fs >= 120:
        try:
            w0 = 50.0 / (fs / 2.0)
            b, a = sps.iirnotch(w0, Q=30.0)
            y = sps.filtfilt(b, a, y)
        except:
            pass
    
    # Bandpass filter
    try:
        low = 0.5 / (fs / 2.0)
        high = 40.0 / (fs / 2.0)
        b, a = sps.butter(4, [low, high], btype="band")
        y = sps.filtfilt(b, a, y)
    except:
        pass
    
    return y

def normalize(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    x = x - np.mean(x)
    std = np.std(x) + 1e-8
    return x / std

def to_model_vector(window_volts: np.ndarray, expected_len: int, fs: int = FS_HZ) -> np.ndarray:
    """
    Convert the latest window to the exact length the scaler expects.
    - Filter + normalize
    - If longer: center-crop; if shorter: pad with zeros
    """
    y = bandpass_notch(window_volts, fs)
    y = normalize(y)
    n = len(y)
    if n > expected_len:
        start = (n - expected_len) // 2
        y = y[start:start+expected_len]
    elif n < expected_len:
        y = np.pad(y, (0, expected_len - n), mode="constant")
    return y.astype(np.float32)

def estimate_hr_bpm(window: np.ndarray, fs: int = FS_HZ) -> Optional[float]:
    if sps is None or len(window) < fs:  # Need at least 1 second of data
        return None
    
    try:
        y = bandpass_notch(window, fs)
        y = normalize(y)
        
        # Peak detection
        peaks, _ = sps.find_peaks(y, distance=int(0.4*fs), prominence=0.5)
        if len(peaks) < 2:
            return None
        
        rr_intervals = np.diff(peaks) / fs
        mean_rr = np.mean(rr_intervals)
        
        if mean_rr <= 0:
            return None
        
        hr = 60.0 / mean_rr
        return float(hr) if 40 <= hr <= 200 else None
        
    except:
        return None