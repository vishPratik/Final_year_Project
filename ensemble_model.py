import numpy as np
import pandas as pd
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import roc_auc_score, classification_report
import joblib

def create_ensemble():
    """Create ensemble of best models"""
    
    # Load data
    data = joblib.load('./models/features_labels.pkl')
    features_df = data['features']
    encoded_labels = data['labels']
    label_encoder = data['label_encoder']
    
    # Create binary labels
    afib_idx = list(label_encoder.classes_).index('A')
    y_binary = np.where(encoded_labels == afib_idx, 1, 0)
    
    # Calculate scale_pos_weight for XGBoost
    n_non_afib = np.sum(y_binary == 0)
    n_afib = np.sum(y_binary == 1)
    scale_pos_weight = n_non_afib / n_afib if n_afib > 0 else 1.0
    print(f"XGBoost scale_pos_weight: {scale_pos_weight:.2f}")
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        features_df, y_binary, test_size=0.2, random_state=42, stratify=y_binary
    )
    
    # Apply SMOTE and scale
    from imblearn.over_sampling import SMOTE
    from sklearn.preprocessing import StandardScaler
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_res)
    X_test_scaled = scaler.transform(X_test)
    
    # Load or train individual models
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier
    
    # Create ensemble of different models with proper parameters
    estimators = [
        ('rf', RandomForestClassifier(
            n_estimators=200, 
            random_state=42, 
            class_weight='balanced_subsample',
            n_jobs=-1
        )),
        ('xgb', XGBClassifier(
            n_estimators=200,
            random_state=42,
            max_depth=6,
            learning_rate=0.1,
            scale_pos_weight=scale_pos_weight,  # Use calculated float value
            use_label_encoder=False,
            eval_metric='logloss'
        ))
    ]
    
    ensemble = VotingClassifier(estimators=estimators, voting='soft', n_jobs=-1)
    print("Training ensemble model...")
    ensemble.fit(X_train_scaled, y_train_res)
    
    # Evaluate
    y_pred_proba = ensemble.predict_proba(X_test_scaled)[:, 1]
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    print(f"Ensemble ROC AUC: {roc_auc:.4f}")
    
    # Make predictions and get classification report
    y_pred = ensemble.predict(X_test_scaled)
    print("\nEnsemble Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Non-AFib', 'AFib']))
    
    # Save ensemble model
    joblib.dump(ensemble, './models/afib_ensemble_model.pkl')
    joblib.dump(scaler, './models/ensemble_scaler.pkl')
    
    print("Ensemble model saved!")
    
    return ensemble, roc_auc

if __name__ == "__main__":
    create_ensemble()