import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from imblearn.over_sampling import SMOTE

def run_binary_afib_classification():
    """Run binary classification focusing on AFib detection"""
    
    # Load saved features and labels
    try:
        data = joblib.load('./models/features_labels.pkl')
        features_df = data['features']
        encoded_labels = data['labels']
        label_encoder = data['label_encoder']
        print("Successfully loaded features and labels")
    except Exception as e:
        print(f"Error loading features: {e}")
        print("Please run main.py first to generate features")
        return
    
    print("Running binary classification for AFib detection...")
    
    # Find the index of AFib class
    afib_idx = list(label_encoder.classes_).index('A')
    
    # Create binary labels: AFib (1) vs Non-AFib (0)
    y_binary = np.where(encoded_labels == afib_idx, 1, 0)
    
    print(f"Binary class distribution:")
    print(f"  AFib (1): {np.sum(y_binary == 1)} samples")
    print(f"  Non-AFib (0): {np.sum(y_binary == 0)} samples")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        features_df, y_binary, 
        test_size=0.2, 
        random_state=42, 
        stratify=y_binary
    )
    
    # Apply SMOTE for binary classification
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    
    print(f"After SMOTE:")
    print(f"  AFib (1): {np.sum(y_train_res == 1)} samples")
    print(f"  Non-AFib (0): {np.sum(y_train_res == 0)} samples")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_res)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest classifier optimized for binary classification
    model = RandomForestClassifier(
        n_estimators=200, 
        random_state=42,
        class_weight='balanced_subsample',
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        n_jobs=-1
    )
    
    model.fit(X_train_scaled, y_train_res)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Calculate metrics
    print("\nBinary Classification Report (AFib vs Non-AFib):")
    print(classification_report(y_test, y_pred, target_names=['Non-AFib', 'AFib']))
    
    # ROC AUC
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    print(f"ROC AUC Score: {roc_auc:.4f}")
    
    # Precision-Recall curve
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    pr_auc = auc(recall, precision)
    print(f"Precision-Recall AUC: {pr_auc:.4f}")
    
    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Non-AFib', 'AFib'],
                yticklabels=['Non-AFib', 'AFib'])
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix - AFib vs Non-AFib')
    plt.show()
    
    # Plot Precision-Recall curve
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, marker='.')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve (AUC = {pr_auc:.3f})')
    plt.grid(True)
    plt.show()
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': features_df.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x='importance', y='feature', data=feature_importance.head(15))
    plt.title('Top 15 Feature Importance - AFib Binary Classification')
    plt.tight_layout()
    plt.show()
    
    # Save binary model
    joblib.dump(model, './models/afib_binary_model.pkl')
    joblib.dump(scaler, './models/afib_scaler.pkl')
    print("Binary AFib model saved!")
    
    return model, roc_auc, pr_auc

if __name__ == "__main__":
    run_binary_afib_classification()