import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, roc_auc_score
import joblib

def tune_binary_model():
    """Hyperparameter tuning for binary AFib detection"""
    
    # Load data
    data = joblib.load('./models/features_labels.pkl')
    features_df = data['features']
    encoded_labels = data['labels']
    label_encoder = data['label_encoder']
    
    # Create binary labels
    afib_idx = list(label_encoder.classes_).index('A')
    y_binary = np.where(encoded_labels == afib_idx, 1, 0)
    
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
    
    # Parameter distribution
    param_dist = {
        'n_estimators': [100, 200, 300, 400],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None],
        'class_weight': ['balanced', 'balanced_subsample', None]
    }
    
    # Use ROC AUC as scoring metric
    scorer = make_scorer(roc_auc_score)
    
    # Randomized search
    random_search = RandomizedSearchCV(
        RandomForestClassifier(random_state=42),
        param_distributions=param_dist,
        n_iter=30,
        scoring=scorer,
        cv=3,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    
    print("Starting hyperparameter tuning for binary AFib detection...")
    random_search.fit(X_train_scaled, y_train_res)
    
    print("Best parameters found:")
    for param, value in random_search.best_params_.items():
        print(f"  {param}: {value}")
    
    print(f"Best ROC AUC score: {random_search.best_score_:.4f}")
    
    # Evaluate on test set
    X_test_scaled = scaler.transform(X_test)
    best_model = random_search.best_estimator_
    y_pred_proba = best_model.predict_proba(X_test_scaled)[:, 1]
    test_roc_auc = roc_auc_score(y_test, y_pred_proba)
    print(f"Test set ROC AUC: {test_roc_auc:.4f}")
    
    # Save tuned model
    joblib.dump(best_model, './models/afib_tuned_model.pkl')
    joblib.dump(scaler, './models/tuned_scaler.pkl')
    
    return best_model, random_search.best_params_

if __name__ == "__main__":
    tune_binary_model()