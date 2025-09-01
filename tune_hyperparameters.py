from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, f1_score
import numpy as np
import joblib

def tune_random_forest():
    """Perform hyperparameter tuning for Random Forest"""
    
    # Load saved features and labels
    try:
        data = joblib.load('./models/features_labels.pkl')
        features_df = data['features']
        encoded_labels = data['labels']
        print("Successfully loaded features and labels")
    except Exception as e:
        print(f"Error loading features: {e}")
        print("Please run main.py first to generate features")
        return
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        features_df, encoded_labels, 
        test_size=0.2, 
        random_state=42, 
        stratify=encoded_labels
    )
    
    # Apply sampling
    from imblearn.over_sampling import SMOTE
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    
    # Scale features
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_res)
    
    # Define parameter distribution
    param_dist = {
        'n_estimators': [100, 200, 300, 400],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None],
        'bootstrap': [True, False],
        'class_weight': ['balanced', 'balanced_subsample', None]
    }
    
    # Create model
    rf = RandomForestClassifier(random_state=42)
    
    # Use macro F1 as scoring metric
    scorer = make_scorer(f1_score, average='macro')
    
    # Randomized search
    random_search = RandomizedSearchCV(
        rf, 
        param_distributions=param_dist,
        n_iter=50,
        scoring=scorer,
        cv=3,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    
    print("Starting hyperparameter tuning...")
    random_search.fit(X_train_scaled, y_train_res)
    
    print("Best parameters found:")
    for param, value in random_search.best_params_.items():
        print(f"  {param}: {value}")
    
    print(f"Best cross-validation score: {random_search.best_score_:.4f}")
    
    # Save best model
    best_model = random_search.best_estimator_
    joblib.dump(best_model, './models/tuned_rf_model.pkl')
    print("Tuned model saved!")
    
    return best_model, random_search.best_params_

if __name__ == "__main__":
    tune_random_forest()