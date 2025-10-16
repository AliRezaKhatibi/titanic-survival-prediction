"""
Titanic Model Loading and Prediction Script
Generated: 20251016_134039
"""
import joblib
import pandas as pd
import numpy as np

# Load best model
print("Loading best model...")
best_model = joblib.load("saved_models/best_model_Random Forest_20251016_134039.pkl")

# Load preprocessing pipeline
print("Loading preprocessing pipeline...")
preprocessing = joblib.load("saved_models/preprocessing_pipeline_20251016_134039.pkl")

# Example prediction function
def predict_survival(passenger_data):
    """
    Predict survival probability for new passenger data
    
    Args:
        passenger_data (dict): Passenger features matching training data
    
    Returns:
        dict: Prediction and probability
    """
    # Convert to DataFrame
    df = pd.DataFrame([passenger_data])
    
    # Apply preprocessing (implement based on your pipeline)
    # processed_data = preprocess_data(df, preprocessing)
    
    # Predict
    prediction = best_model.predict(df)[0]
    probability = best_model.predict_proba(df)[:, 1][0] if hasattr(best_model, 'predict_proba') else None
    
    return {
        'survived': int(prediction),
        'survival_probability': float(probability) if probability else None,
        'model': "Random Forest"
    }

# Example usage:
# result = predict_survival({
#     'Pclass': 1, 'Sex': 0, 'Age': 25, 'Fare': 100, ...
# })
# print(result)

print("✅ Models loaded successfully!")
print(f"Best model: {best_model}")
