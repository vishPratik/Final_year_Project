# test_udp.py
import socket
import sys

def test_udp_server():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(('0.0.0.0', 9002))
    sock.settimeout(5.0)
    
    print("Listening on UDP port 9002...")
    print("Press Ctrl+C to stop")
    
    try:
        while True:
            try:
                data, addr = sock.recvfrom(1024)
                print(f"Received from {addr}: {data.decode()}")
            except socket.timeout:
                print("No data received...")
    except KeyboardInterrupt:
        print("\nStopping server...")
    finally:
        sock.close()

if __name__ == "__main__":
    test_udp_server()