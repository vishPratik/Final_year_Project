import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, precision_recall_curve, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from imblearn.over_sampling import SMOTE

def optimize_afib_detection():
    """Optimize AFib detection by adjusting prediction threshold"""
    
    # Load saved features and labels
    data = joblib.load('./models/features_labels.pkl')
    features_df = data['features']
    encoded_labels = data['labels']
    label_encoder = data['label_encoder']
    
    # Create binary labels
    afib_idx = list(label_encoder.classes_).index('A')
    y_binary = np.where(encoded_labels == afib_idx, 1, 0)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        features_df, y_binary, test_size=0.2, random_state=42, stratify=y_binary
    )
    
    # Apply SMOTE and scale
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_res)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = RandomForestClassifier(
        n_estimators=200, random_state=42,
        class_weight='balanced_subsample', n_jobs=-1
    )
    model.fit(X_train_scaled, y_train_res)
    
    # Get prediction probabilities
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Find optimal threshold for AFib detection
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    
    # Find threshold that gives at least 80% recall for AFib
    optimal_threshold = 0.3  # Start with a lower threshold
    optimal_idx = 0
    for i, threshold in enumerate(thresholds):
        if tpr[i] >= 0.8:  # Target 80% recall
            optimal_threshold = threshold
            optimal_idx = i
            break
    
    print(f"Optimal threshold for 80% recall: {optimal_threshold:.3f}")
    
    # Make predictions with optimal threshold
    y_pred_optimized = (y_pred_proba >= optimal_threshold).astype(int)
    
    # Calculate metrics
    print("\nOptimized Classification Report:")
    print(classification_report(y_test, y_pred_optimized, target_names=['Non-AFib', 'AFib']))
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_test, y_pred_optimized)
    print("Optimized Confusion Matrix:")
    print(f"True Non-AFib correctly predicted: {cm[0, 0]}")
    print(f"True Non-AFib incorrectly predicted as AFib: {cm[0, 1]}")
    print(f"True AFib incorrectly predicted as Non-AFib: {cm[1, 0]}")
    print(f"True AFib correctly predicted: {cm[1, 1]}")
    
    # Plot ROC curve with optimal threshold
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc_score(y_test, y_pred_proba):.3f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.scatter(fpr[optimal_idx], tpr[optimal_idx], 
                color='red', label=f'Optimal Threshold ({optimal_threshold:.3f})')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate (Recall)')
    plt.title('ROC Curve with Optimal Threshold')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Plot precision-recall curve
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    plt.figure(figsize=(10, 6))
    plt.plot(recall, precision, marker='.')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.grid(True)
    plt.show()
    
    # Save optimized model
    joblib.dump(model, './models/afib_optimized_model.pkl')
    joblib.dump({'threshold': optimal_threshold}, './models/optimal_threshold.pkl')
    
    print(f"Optimized model saved with threshold {optimal_threshold:.3f}")
    
    return optimal_threshold

if __name__ == "__main__":
    optimize_afib_detection()