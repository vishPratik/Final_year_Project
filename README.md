# Project documentation
# ECG AFib Detection Project

## Overview
This project detects Atrial Fibrillation (AFib) from ECG data using machine learning. It processes raw ECG signals, extracts features, trains models, and provides real-time and batch prediction interfaces, including a Streamlit web app.

## Features
- Feature engineering and selection from ECG signals
- Multiple model training (Random Forest, XGBoost, LightGBM, Ensemble)
- Hyperparameter tuning and threshold optimization
- Real-time ECG streaming and prediction
- User-friendly Streamlit web app for predictions

## Project Structure

```
ecg_af_detection/
├── main.py                # Main pipeline: feature extraction, model training, evaluation
├── feature_selection.py   # Feature selection using Random Forest
├── binary_afib.py         # Binary AFib detection model
├── ensemble_model.py      # Ensemble model (RF, XGB, LGBM)
├── tune_binary_model.py   # Hyperparameter tuning for binary model
├── improve_afib_recall.py # Optimize recall by threshold adjustment
├── check_model.py         # Check model files for integrity
├── live_ecg_receiver.py   # Live ECG data receiver for real-time prediction
├── streamlit_app.py       # Streamlit web app for user interaction
├── models/                # Saved models, scalers, selectors, encoders
├── data/                  # Raw ECG data and reference labels
├── requirements.txt       # Python dependencies
├── config/config.yaml     # Configuration settings
```

## Feature Engineering & Selection
Extracted features include:
- Statistical: mean, std, min, max, range, skewness, kurtosis, percentiles, iqr, rms, zero_crossing
- Energy: energy, abs_energy
- Frequency: spectral_centroid, spectral_energy, dominant_frequency, spectral_bandwidth, spectral_flatness, spectral_rolloff
- Heart Rate Variability: mean_rr, std_rr, hrv, rmssd, nn50, pnn50, max_rr, min_rr, rr_range
- Complexity: max_slope, mean_slope, slope_std, signal_entropy
- Engineered: pnn50_slope_interaction, spectral_features_combined, hrv_complexity, rr_variability_ratio, amplitude_asymmetry

## Setup
1. Clone the repository and install dependencies:
	```
	pip install -r requirements.txt
	```
2. Place raw ECG data in `data/raw/` and reference labels in `data/REFERENCE-v3.csv`.

## Usage
1. **Feature Extraction & Model Training:**
	```
	python main.py
	```
2. **Feature Selection:**
	```
	python feature_selection.py
	```
3. **Train Binary/Ensemble Models:**
	```
	python binary_afib.py
	python ensemble_model.py
	```
4. **Hyperparameter Tuning & Optimization:**
	```
	python tune_binary_model.py
	python improve_afib_recall.py
	```
5. **Check Model Files:**
	```
	python check_model.py
	```
6. **Run Streamlit App:**
	```
	streamlit run streamlit_app.py
	```

## Live ECG Streaming
Use `live_ecg_receiver.py` to receive and process live ECG data for real-time AFib detection.

## Models & Outputs
- Trained models, scalers, feature selectors, and results are saved in the `models/` directory.

## Contributing
Feel free to open issues or pull requests for improvements.

## License
MIT License (or specify your license here)
