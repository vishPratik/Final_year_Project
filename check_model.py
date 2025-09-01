import pickle
import joblib

def check_model_file(file_path):
    print(f"Checking: {file_path}")
    
    # Pehle file size check karo
    try:
        with open(file_path, 'rb') as f:
            data = f.read()
        print(f"File size: {len(data)} bytes")
    except Exception as e:
        print(f"File read error: {e}")
        return
    
    # Joblib se try karo
    try:
        model = joblib.load(file_path)
        print("✓ Joblib load successful")
        print(f"Model type: {type(model)}")
        return True
    except Exception as e:
        print(f"✗ Joblib failed: {e}")
    
    # Pickle se try karo
    try:
        with open(file_path, 'rb') as f:
            model = pickle.load(f)
        print("✓ Pickle load successful")
        print(f"Model type: {type(model)}")
        return True
    except Exception as e:
        print(f"✗ Pickle failed: {e}")
    
    return False

# Dono files check karo
print("="*50)
check_model_file('models/afib_ensemble_model.pkl')
print("="*50)
check_model_file('models/ensemble_scaler.pkl')