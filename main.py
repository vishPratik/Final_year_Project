import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score, balanced_accuracy_score, matthews_corrcoef
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import wfdb
import joblib
from scipy import signal
from scipy.stats import kurtosis, skew
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek, SMOTEENN
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import warnings
warnings.filterwarnings('ignore')

def extract_features(ecg_signal, fs=300):
    """Extract features from ECG signal with enhanced feature set"""
    features = {}
    
    # Normalize signal first
    ecg_normalized = (ecg_signal - np.mean(ecg_signal)) / (np.std(ecg_signal) + 1e-8)
    
    # Basic statistical features
    features['mean'] = np.mean(ecg_normalized)
    features['std'] = np.std(ecg_normalized)
    features['min'] = np.min(ecg_normalized)
    features['max'] = np.max(ecg_normalized)
    features['range'] = features['max'] - features['min']
    features['skewness'] = skew(ecg_normalized)
    features['kurtosis'] = kurtosis(ecg_normalized)
    
    # Percentiles
    features['p10'] = np.percentile(ecg_normalized, 10)
    features['p25'] = np.percentile(ecg_normalized, 25)
    features['p50'] = np.percentile(ecg_normalized, 50)
    features['p75'] = np.percentile(ecg_normalized, 75)
    features['p90'] = np.percentile(ecg_normalized, 90)
    features['iqr'] = features['p75'] - features['p25']
    
    # Additional features
    features['rms'] = np.sqrt(np.mean(ecg_normalized**2))
    zero_crossings = np.where(np.diff(np.sign(ecg_normalized)))[0]
    features['zero_crossing'] = len(zero_crossings) / len(ecg_normalized)
    
    # Signal energy features
    features['energy'] = np.sum(ecg_normalized**2)
    features['abs_energy'] = np.sum(np.abs(ecg_normalized))
    
    # Frequency domain features
    f, Pxx = signal.welch(ecg_normalized, fs=fs, nperseg=min(512, len(ecg_normalized)))
    
    if len(Pxx) > 0:
        features['spectral_centroid'] = np.sum(f * Pxx) / (np.sum(Pxx) + 1e-8)
        features['spectral_energy'] = np.sum(Pxx)
        features['dominant_frequency'] = f[np.argmax(Pxx)]
        
        # Additional spectral features
        features['spectral_bandwidth'] = np.sqrt(np.sum((f - features['spectral_centroid'])**2 * Pxx) / (np.sum(Pxx) + 1e-8))
        features['spectral_flatness'] = np.exp(np.mean(np.log(Pxx + 1e-12))) / (np.mean(Pxx) + 1e-8)
        
        # Spectral rolloff (85%)
        cumulative_energy = np.cumsum(Pxx)
        total_energy = np.sum(Pxx)
        rolloff_index = np.where(cumulative_energy >= 0.85 * total_energy)[0]
        features['spectral_rolloff'] = f[rolloff_index[0]] if len(rolloff_index) > 0 else 0
    else:
        features['spectral_centroid'] = 0
        features['spectral_energy'] = 0
        features['dominant_frequency'] = 0
        features['spectral_bandwidth'] = 0
        features['spectral_flatness'] = 0
        features['spectral_rolloff'] = 0
    
    # Heart rate variability features with improved peak detection
    try:
        # Improved peak detection
        peaks, _ = signal.find_peaks(
            ecg_normalized, 
            height=np.percentile(ecg_normalized, 75),
            distance=fs//3,
            prominence=0.5
        )
        
        if len(peaks) > 3:
            rr_intervals = np.diff(peaks) / fs
            features['mean_rr'] = np.mean(rr_intervals)
            features['std_rr'] = np.std(rr_intervals)
            features['hrv'] = features['std_rr'] / (features['mean_rr'] + 1e-8)
            features['rmssd'] = np.sqrt(np.mean(np.diff(rr_intervals)**2))
            features['nn50'] = np.sum(np.abs(np.diff(rr_intervals)) > 0.05)
            features['pnn50'] = features['nn50'] / (len(rr_intervals) + 1e-8)
            features['max_rr'] = np.max(rr_intervals)
            features['min_rr'] = np.min(rr_intervals)
            features['rr_range'] = features['max_rr'] - features['min_rr']
        else:
            features.update({
                'mean_rr': 0, 'std_rr': 0, 'hrv': 0, 'rmssd': 0,
                'nn50': 0, 'pnn50': 0, 'max_rr': 0, 'min_rr': 0, 'rr_range': 0
            })
    except:
        features.update({
            'mean_rr': 0, 'std_rr': 0, 'hrv': 0, 'rmssd': 0,
            'nn50': 0, 'pnn50': 0, 'max_rr': 0, 'min_rr': 0, 'rr_range': 0
        })
    
    # Signal complexity features
    diff_signal = np.diff(ecg_normalized)
    features['max_slope'] = np.max(np.abs(diff_signal))
    features['mean_slope'] = np.mean(np.abs(diff_signal))
    features['slope_std'] = np.std(np.abs(diff_signal))
    
    # Approximate entropy
    features['signal_entropy'] = -np.sum(ecg_normalized**2 * np.log(ecg_normalized**2 + 1e-12))
    
    return features

def create_directory(directory_path):
    """Create directory if it doesn't exist"""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

def plot_confusion_matrix(y_true, y_pred, class_names, title="Confusion Matrix"):
    """Plot confusion matrix with better visualization"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                annot_kws={"size": 12})
    plt.ylabel('True Label', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.title(title, fontsize=16)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

def apply_sampling_strategy(X, y, strategy='smote'):
    """Apply sampling strategy to handle class imbalance"""
    if strategy == 'smote':
        class_counts = np.bincount(y)
        min_samples = np.min(class_counts)
        k_neighbors = min(5, min_samples - 1)
        sampler = SMOTE(random_state=42, k_neighbors=k_neighbors)
    elif strategy == 'adasyn':
        sampler = ADASYN(random_state=42)
    elif strategy == 'smote_tomek':
        class_counts = np.bincount(y)
        min_samples = np.min(class_counts)
        k_neighbors = min(5, min_samples - 1)
        sampler = SMOTETomek(random_state=42, smote=SMOTE(k_neighbors=k_neighbors))
    elif strategy == 'smote_enn':
        class_counts = np.bincount(y)
        min_samples = np.min(class_counts)
        k_neighbors = min(5, min_samples - 1)
        sampler = SMOTEENN(random_state=42, smote=SMOTE(k_neighbors=k_neighbors))
    elif strategy == 'undersample':
        sampler = RandomUnderSampler(random_state=42)
    else:
        return X, y
    
    X_resampled, y_resampled = sampler.fit_resample(X, y)
    return X_resampled, y_resampled

def engineer_features(features_df):
    """Create additional features based on feature importance"""
    # Create interaction features from top important features
    if 'pnn50' in features_df.columns and 'slope_std' in features_df.columns:
        features_df['pnn50_slope_interaction'] = features_df['pnn50'] * features_df['slope_std']
    
    if 'spectral_rolloff' in features_df.columns and 'spectral_centroid' in features_df.columns:
        features_df['spectral_features_combined'] = features_df['spectral_rolloff'] * features_df['spectral_centroid']
    
    if 'pnn50' in features_df.columns and 'nn50' in features_df.columns and 'hrv' in features_df.columns:
        features_df['hrv_complexity'] = features_df['pnn50'] * features_df['nn50'] * features_df['hrv']
    
    # Create ratio features
    if 'mean_rr' in features_df.columns and 'std_rr' in features_df.columns:
        features_df['rr_variability_ratio'] = features_df['std_rr'] / (features_df['mean_rr'] + 1e-8)
    
    if 'max' in features_df.columns and 'min' in features_df.columns:
        features_df['amplitude_asymmetry'] = (features_df['max'] - np.abs(features_df['min'])) / (features_df['max'] + np.abs(features_df['min']) + 1e-8)
    
    return features_df

def train_models(X_train, y_train, model_types=['rf']):
    """Train multiple models and return them"""
    models = {}
    
    for model_type in model_types:
        if model_type == 'rf':
            model = RandomForestClassifier(
                n_estimators=200, 
                random_state=42, 
                class_weight='balanced_subsample',
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                n_jobs=-1
            )
        elif model_type == 'xgb':
            model = XGBClassifier(
                n_estimators=200,
                random_state=42,
                max_depth=6,
                learning_rate=0.1,
                scale_pos_weight='balanced',
                use_label_encoder=False,
                eval_metric='logloss'
            )
        elif model_type == 'lgbm':
            model = LGBMClassifier(
                n_estimators=200,
                random_state=42,
                max_depth=6,
                learning_rate=0.05,
                num_leaves=31,
                min_child_samples=20,
                class_weight='balanced',
                verbosity=-1
            )
        else:
            continue
            
        model.fit(X_train, y_train)
        models[model_type] = model
        print(f"Trained {model_type.upper()} model")
    
    return models

def evaluate_models(models, X_test, y_test, label_encoder):
    """Evaluate multiple models and return results"""
    results = {}
    
    for model_name, model in models.items():
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        report = classification_report(y_test, y_pred, target_names=label_encoder.classes_, zero_division=0, output_dict=True)
        f1 = f1_score(y_test, y_pred, average='macro')
        balanced_acc = balanced_accuracy_score(y_test, y_pred)
        mcc = matthews_corrcoef(y_test, y_pred)
        
        results[model_name] = {
            'model': model,
            'report': report,
            'f1_score': f1,
            'balanced_accuracy': balanced_acc,
            'mcc': mcc,
            'y_pred': y_pred
        }
        
        print(f"\n{model_name.upper()} Results:")
        print(f"Macro F1 Score: {f1:.4f}")
        print(f"Balanced Accuracy: {balanced_acc:.4f}")
        print(f"Matthews Correlation Coefficient: {mcc:.4f}")
    
    return results

def save_features_and_labels(features_df, labels, label_encoder, filename='./models/features_labels.pkl'):
    """Save features and labels for later use"""
    import joblib
    data_to_save = {
        'features': features_df,
        'labels': labels,
        'label_encoder': label_encoder
    }
    joblib.dump(data_to_save, filename)
    print(f"Features and labels saved to {filename}")

def main():
    # Configuration
    data_dir = "./data/raw"
    reference_file = "./data/REFERENCE-v3.csv"
    max_records = 1000
    test_size = 0.2
    random_state = 42
    sampling_strategy = 'smote_tomek'
    model_types = ['rf', 'xgb', 'lgbm']
    
    # Create directories
    create_directory(data_dir)
    create_directory("./models")
    
    # Load reference data
    print("Loading reference data...")
    try:
        reference_df = pd.read_csv(reference_file, header=None, names=['filename', 'label'])
        print(f"Found {len(reference_df)} records in reference file")
    except Exception as e:
        print(f"Error loading reference file: {e}")
        return
    
    # Process ECG records
    print("Processing ECG records...")
    features_list = []
    labels_list = []
    failed_records = []
    
    for i, (idx, row) in tqdm(enumerate(reference_df.iterrows()), total=min(max_records, len(reference_df))):
        if i >= max_records:
            break
            
        record_name = row['filename']
        label = row['label']
        
        try:
            # Load ECG record
            record_path = os.path.join(data_dir, record_name)
            record = wfdb.rdrecord(record_path)
            
            if record.p_signal is not None and record.p_signal.shape[1] > 0:
                ecg_signal = record.p_signal[:, 0]
                
                # Extract features
                features = extract_features(ecg_signal, record.fs)
                features_list.append(features)
                labels_list.append(label)
            else:
                failed_records.append(record_name)
                
        except Exception as e:
            failed_records.append(record_name)
            if len(failed_records) < 5:
                print(f"Failed to process {record_name}: {e}")
    
    print(f"Successfully processed {len(features_list)} records")
    print(f"Failed to process {len(failed_records)} records")
    
    if len(features_list) == 0:
        print("No records were processed successfully. Exiting.")
        return
    
    # Convert to DataFrame
    features_df = pd.DataFrame(features_list)
    labels_series = pd.Series(labels_list)
    
    # Apply feature engineering
    print("Engineering additional features...")
    features_df = engineer_features(features_df)
    
    # Encode labels
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels_series)
    
    print("Class distribution:")
    for i, class_name in enumerate(label_encoder.classes_):
        count = np.sum(encoded_labels == i)
        print(f"  {class_name}: {count} records")
    
    # Save features and labels for later use
    save_features_and_labels(features_df, encoded_labels, label_encoder)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        features_df, encoded_labels, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=encoded_labels
    )
    
    # Apply sampling to training data
    print(f"Applying {sampling_strategy} sampling strategy...")
    X_train_resampled, y_train_resampled = apply_sampling_strategy(X_train, y_train, sampling_strategy)
    
    print("Resampled class distribution:")
    unique, counts = np.unique(y_train_resampled, return_counts=True)
    for cls, count in zip(unique, counts):
        print(f"  {label_encoder.inverse_transform([cls])[0]}: {count} records")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_resampled)
    X_test_scaled = scaler.transform(X_test)
    
    # Train multiple models
    print("Training models...")
    models = train_models(X_train_scaled, y_train_resampled, model_types)
    
    # Evaluate models
    print("Evaluating models...")
    results = evaluate_models(models, X_test_scaled, y_test, label_encoder)
    
    # Find best model
    best_model_name = max(results.keys(), key=lambda x: results[x]['f1_score'])
    best_model = results[best_model_name]['model']
    best_f1 = results[best_model_name]['f1_score']
    
    print(f"\nBest model: {best_model_name.upper()} with F1 = {best_f1:.4f}")
    
    # Detailed report for best model
    print(f"\nDetailed classification report for {best_model_name.upper()}:")
    print(classification_report(y_test, results[best_model_name]['y_pred'], 
                               target_names=label_encoder.classes_, zero_division=0))
    
    # Class-specific performance for best model
    print("Class-specific performance:")
    y_pred_best = results[best_model_name]['y_pred']
    for i, class_name in enumerate(label_encoder.classes_):
        class_idx = i
        if class_idx in y_test:
            class_mask = y_test == class_idx
            if np.any(class_mask):
                class_precision = np.sum((y_pred_best[class_mask] == class_idx) & (y_test[class_mask] == class_idx)) / np.sum(y_pred_best[class_mask] == class_idx) if np.sum(y_pred_best[class_mask] == class_idx) > 0 else 0
                class_recall = np.sum((y_pred_best[class_mask] == class_idx) & (y_test[class_mask] == class_idx)) / np.sum(y_test[class_mask] == class_idx)
                class_f1 = 2 * (class_precision * class_recall) / (class_precision + class_recall + 1e-8)
                
                print(f"  {class_name}: Precision={class_precision:.3f}, Recall={class_recall:.3f}, F1={class_f1:.3f}")
    
    # Plot confusion matrix for best model
    plot_confusion_matrix(y_test, y_pred_best, label_encoder.classes_, 
                         f"Confusion Matrix - {best_model_name.upper()}")
    
    # Feature importance for tree-based models
    if hasattr(best_model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'feature': features_df.columns,
            'importance': best_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        plt.figure(figsize=(12, 8))
        sns.barplot(x='importance', y='feature', data=feature_importance.head(20))
        plt.title(f'Top 20 Feature Importance - {best_model_name.upper()}')
        plt.tight_layout()
        plt.show()
        
        # Save feature importance
        feature_importance.to_csv(f'./models/feature_importance_{best_model_name}.csv', index=False)
    
    # Save models and artifacts
    joblib.dump(best_model, f'./models/best_model_{best_model_name}.pkl')
    joblib.dump(scaler, './models/scaler.pkl')
    joblib.dump(label_encoder, './models/label_encoder.pkl')
    
    # Save all results
    results_summary = {
        'best_model': best_model_name,
        'best_f1': float(best_f1),
        'all_results': {k: {'f1_score': float(v['f1_score']), 
                          'balanced_accuracy': float(v['balanced_accuracy']),
                          'mcc': float(v['mcc'])} for k, v in results.items()}
    }
    
    import json
    with open('./models/results_summary.json', 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print("Models and results saved to ./models/ directory")
    print("\nTraining completed successfully!")

if __name__ == "__main__":
    main()