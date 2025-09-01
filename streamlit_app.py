import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from collections import deque
import time

# Import your ECG receiver
try:
    from live_ecg_receiver import ECGStreamServer, to_model_vector, estimate_hr_bpm, FS_HZ
except ImportError as e:
    st.error(f"Could not import live_ecg_receiver: {e}")
    # Create dummy functions to prevent errors
    class ECGStreamServer:
        def __init__(self, *args, **kwargs):
            self.leadoff_plus = deque()
            self.leadoff_minus = deque()
            self.samples_volts = deque()
            self.samples_adc = deque()
            pass
        def stop(self):
            pass
        def get_latest_window(self, n):
            return None
    
    def to_model_vector(*args, **kwargs):
        return np.random.randn(100)
    
    def estimate_hr_bpm(*args, **kwargs):
        return 72.0
    
    FS_HZ = 250

# App title
st.title("ECG AF Detection App")

# Model loading function
@st.cache_resource
def load_model():
    try:
        model = joblib.load('models/afib_ensemble_model.pkl')
        scaler = joblib.load('models/ensemble_scaler.pkl')
        return model, scaler
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        return None, None

# Prediction function
def predict_afib(model, scaler, ecg_data):
    try:
        expected_features = scaler.n_features_in_
        
        if len(ecg_data) > expected_features:
            ecg_data = ecg_data[:expected_features]
        elif len(ecg_data) < expected_features:
            ecg_data = np.pad(ecg_data, (0, expected_features - len(ecg_data)), 
                            mode='constant')
        
        processed_data = scaler.transform([ecg_data])
        prediction = model.predict(processed_data)
        prediction_proba = model.predict_proba(processed_data)
        
        return prediction[0], prediction_proba[0]
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None, None

# Main app
def main():
    st.sidebar.header("Input ECG Data")
    
    model, scaler = load_model()
    
    if model is None or scaler is None:
        st.error("Models could not be loaded. Please check model files.")
        return
    
    if scaler is not None:
        st.sidebar.info(f"📊 Model requires: {scaler.n_features_in_} ECG features")
    
    option = st.sidebar.selectbox(
        "Input method",
        ["File Upload", "Manual Input", "Live Stream (ESP8266)"]
    )
    
    # --- File Upload branch ---
    if option == "File Upload":
        st.header("Upload ECG Data File")
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file is not None:
            try:
                data = pd.read_csv(uploaded_file)
                st.write("Uploaded Data Preview:")
                st.write(data.head())
                
                if st.button("Analyze"):
                    if len(data) == 0:
                        st.error("Uploaded file is empty!")
                    else:
                        ecg_values = data.values.flatten()
                        st.write(f"📈 Your data has: {len(ecg_values)} features")
                        
                        result, probabilities = predict_afib(model, scaler, ecg_values)
                        
                        if result is not None:
                            st.success(f"🔍 Prediction: {'AFib Detected' if result == 1 else 'Normal Rhythm'}")
                            st.write(f"🎯 Confidence: {max(probabilities)*100:.2f}%")
                            
            except Exception as e:
                st.error(f"Error reading file: {e}")

    # --- Manual Input branch ---
    elif option == "Manual Input":
        st.header("Manual Input ECG Features")
        st.write("Enter ECG feature values (comma-separated):")
        user_input = st.text_area("ECG values", "")
        
        if st.button("Analyze"):
            try:
                ecg_values = np.array([float(x.strip()) for x in user_input.split(",") if x.strip() != ""])
                st.write(f"📈 You entered: {len(ecg_values)} features")
                
                result, probabilities = predict_afib(model, scaler, ecg_values)
                
                if result is not None:
                    st.success(f"🔍 Prediction: {'AFib Detected' if result == 1 else 'Normal Rhythm'}")
                    st.write(f"🎯 Confidence: {max(probabilities)*100:.2f}%")
            except Exception as e:
                st.error(f"Input error: {e}")

    # --- Live Stream branch ---
    elif option == "Live Stream (ESP8266)":
        st.header("Live Stream from ESP8266 + AD8232")
        st.write("1) Flash the ESP8266 with **esp8266_ad8232_streamer.ino**")
        st.write("2) Ensure this PC and the ESP8266 are on the same Wi-Fi network.")
        st.write("3) Press **Start Server** below, then power the ESP8266.")

        port = st.number_input("TCP Server Port", min_value=1024, max_value=65535, value=9002, step=1)
        a0_vref = st.selectbox("A0 full-scale (board)", ["NodeMCU/D1 mini ~3.3V", "Bare ESP8266 1.0V"])
        vref = 3.3 if a0_vref.startswith("NodeMCU") else 1.0

        # Initialize session state variables
        if "server" not in st.session_state:
            st.session_state.server = None
        if "running" not in st.session_state:
            st.session_state.running = False
        if "ecg_buffer" not in st.session_state:
            st.session_state.ecg_buffer = deque(maxlen=1000)

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Start Server", disabled=st.session_state.running):
                try:
                    st.session_state.server = ECGStreamServer(port=int(port), a0_fullscale_volts=float(vref))
                    st.session_state.running = True
                    st.session_state.ecg_buffer.clear()
                    st.success("Server started successfully!")
                except Exception as e:
                    st.error(f"Failed to start server: {e}")
        
        with col2:
            if st.button("Stop Server", disabled=not st.session_state.running):
                try:
                    if st.session_state.server is not None:
                        st.session_state.server.stop()
                    st.session_state.running = False
                    st.info("Server stopped")
                except Exception as e:
                    st.error(f"Error stopping server: {e}")

        # Display status and data
        if st.session_state.running:
            if st.session_state.server is None:
                st.error("Server is not initialized properly")
            else:
                expected = scaler.n_features_in_ if hasattr(scaler, 'n_features_in_') else 100
                
                # DEBUG: Show what's being received
                st.subheader("📊 Debug Information")
                if hasattr(st.session_state.server, 'samples_volts'):
                    sample_count = len(st.session_state.server.samples_volts)
                    st.write(f"**Samples received:** {sample_count}")
                    
                    if sample_count > 0:
                        st.write(f"**Latest ECG value:** {st.session_state.server.samples_volts[-1]:.3f} V")
                        st.write(f"**Latest ADC value:** {st.session_state.server.samples_adc[-1] if hasattr(st.session_state.server, 'samples_adc') and len(st.session_state.server.samples_adc) > 0 else 'N/A'}")
                        
                        if sample_count > 1:
                            st.write(f"**Sample rate:** ~{sample_count / (time.time() - st.session_state.start_time if hasattr(st.session_state, 'start_time') else 1):.1f} Hz")
                
                if hasattr(st.session_state.server, 'leadoff_plus') and len(st.session_state.server.leadoff_plus) > 0:
                    lo_p = st.session_state.server.leadoff_plus[-1]
                    lo_m = st.session_state.server.leadoff_minus[-1] if hasattr(st.session_state.server, 'leadoff_minus') and len(st.session_state.server.leadoff_minus) > 0 else 0
                    
                    status_col1, status_col2 = st.columns(2)
                    with status_col1:
                        st.success("LO+: ✅ GOOD" if lo_p == 0 else "LO+: ❌ DETACHED")
                    with status_col2:
                        st.success("LO-: ✅ GOOD" if lo_m == 0 else "LO-: ❌ DETACHED")
                
                # Get data window for processing
                win = st.session_state.server.get_latest_window(expected)
                
                # Update ECG buffer for plotting
                if win is not None:
                    st.session_state.ecg_buffer.extend(win)
                
                if win is None:
                    st.info("Waiting for data… Connect the ESP8266. You should see this update within a few seconds.")
                    st.write("Check if your ESP8266 is connected and sending data.")
                    
                    # Initialize start time for sample rate calculation
                    if not hasattr(st.session_state, 'start_time'):
                        st.session_state.start_time = time.time()
                else:
                    # Preprocess & predict
                    try:
                        vec = to_model_vector(win, expected_len=expected, fs=FS_HZ)
                        X = scaler.transform([vec])
                        y = model.predict(X)[0]
                        proba = model.predict_proba(X)[0] if hasattr(model, "predict_proba") else None

                        # Display results
                        st.subheader("🧠 Prediction Results")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Prediction", "AFib Detected" if int(y)==1 else "Normal Rhythm")
                            if proba is not None:
                                st.write(f"🎯 Confidence: {float(np.max(proba))*100:.2f}%")
                        
                        with col2:
                            hr = estimate_hr_bpm(win, fs=FS_HZ)
                            if hr:
                                st.metric("Estimated HR (bpm)", f"{hr:.1f}")

                        # Real-time ECG Plot
                        st.subheader("📈 Real-time ECG Signal")
                        if len(st.session_state.ecg_buffer) > 0:
                            fig, ax = plt.subplots(figsize=(10, 4))
                            ax.plot(list(st.session_state.ecg_buffer), 'b-', linewidth=1)
                            ax.set_title("Live ECG Signal")
                            ax.set_xlabel("Samples")
                            ax.set_ylabel("Volts")
                            ax.grid(True, alpha=0.3)
                            st.pyplot(fig)
                            
                            st.write(f"**Displaying:** {len(st.session_state.ecg_buffer)} samples ({len(st.session_state.ecg_buffer)/FS_HZ:.1f} seconds)")

                    except Exception as e:
                        st.error(f"Processing error: {e}")

if __name__ == "__main__":
    main()