from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import joblib

def select_important_features(threshold='median'):
    """Select most important features using Random Forest"""
    
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
    X_test_scaled = scaler.transform(X_test)
    
    # Train a random forest to get feature importance
    selector = SelectFromModel(
        RandomForestClassifier(n_estimators=100, random_state=42),
        threshold=threshold
    )
    
    selector.fit(X_train_scaled, y_train_res)
    
    X_train_selected = selector.transform(X_train_scaled)
    X_test_selected = selector.transform(X_test_scaled)
    
    print(f"Selected {X_train_selected.shape[1]} out of {X_train_scaled.shape[1]} features")
    
    # Train model on selected features
    model = RandomForestClassifier(
        n_estimators=200, 
        random_state=42,
        class_weight='balanced_subsample',
        n_jobs=-1
    )
    
    model.fit(X_train_selected, y_train_res)
    
    # Evaluate
    from sklearn.metrics import f1_score
    y_pred = model.predict(X_test_selected)
    f1 = f1_score(y_test, y_pred, average='macro')
    print(f"F1 with feature selection ({threshold} threshold): {f1:.4f}")
    
    # Get selected feature names
    selected_features = features_df.columns[selector.get_support()]
    print("Selected features:")
    for i, feature in enumerate(selected_features):
        print(f"  {i+1}. {feature}")
    
    # Save selector and selected features
    joblib.dump(selector, './models/feature_selector.pkl')
    joblib.dump(selected_features, './models/selected_features.pkl')
    
    return selector, selected_features, f1

if __name__ == "__main__":
    # Test different thresholds
    thresholds = ['median', 'mean', 0.01, 0.02, 0.03]
    
    best_f1 = 0
    best_threshold = None
    
    for threshold in thresholds:
        print(f"\nTesting threshold: {threshold}")
        try:
            selector, features, f1 = select_important_features(threshold)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        except Exception as e:
            print(f"Error with threshold {threshold}: {e}")
    
    print(f"\nBest threshold: {best_threshold} with F1: {best_f1:.4f}")