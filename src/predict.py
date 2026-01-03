import torch
import pandas as pd
import numpy as np
import joblib
from src.model1 import ChurnModel

def load_inference_tools():
    # 1. Load the saved Scaler
    scaler = joblib.load('models/scaler.pkl')
    
    # 2. Automatically extract the number of features from the scaler
    # scaler.n_features_in_ tells us how many columns were in the original training data
    input_size = scaler.n_features_in_ 
    print(f"Detected Input Size: {input_size}") 

    # 3. Reconstruct the model with the same input size
    model = ChurnModel(input_size=input_size)
    model.load_state_dict(torch.load('models/best_model.pt'))
    model.eval() # Set model to evaluation mode
    
    return model, scaler

def predict_churn(raw_data):
    """
    raw_data: dictionary or array containing customer data before processing
    """
    model, scaler = load_inference_tools()
    
    # Convert data to numpy array and reshape for a single sample prediction
    # Note: In a production environment, this data would go through the same Encoding steps
    # For now, we assume the input is numerical and ready for scaling
    
    features = np.array(raw_data).reshape(1, -1)
    
    # 1. Apply scaling using the same scaler from the training phase
    features_scaled = scaler.transform(features)
    
    # 2. Convert to PyTorch Tensor
    features_tensor = torch.FloatTensor(features_scaled)
    
    # 3. Perform Inference
    with torch.no_grad():
        prob = model(features_tensor).item()
        # Apply the optimized decision threshold: 0.35
        prediction = 1 if prob >= 0.35 else 0
        
    return prob, prediction

if __name__ == "__main__":
    from src.data_loader import DataLoader as CustomDataLoader
    
    # 1. Load and prepare data exactly as done during training
    data_path = r'data/WA_Fn-UseC_-Telco-Customer-Churn.csv'
    loader = CustomDataLoader(data_path, target_column='Churn')
    X_train, X_test, y_train, y_test = loader.prepare_data()

    # 2. Select the first 5 customers from the Test Set
    num_samples = 5
    sample_features = X_test[:num_samples] # Features for 5 customers
    sample_labels = y_test.values[:num_samples] # Ground Truth (Actual status)

    print(f"\n{'#'*30}")
    print(f"--- Testing on {num_samples} Real Customers ---")
    print(f"{'#'*30}\n")

    for i in range(num_samples):
        # Predict for each customer
        # Note: Since X_test is already scaled by the loader, 
        # we pass the features directly to get the prediction
        
        prob, decision = predict_churn(sample_features[i])
        
        actual = "Churn" if sample_labels[i] == 1 else "Stay"
        predicted = "Churn" if decision == 1 else "Stay"
        
        print(f"Customer {i+1}:")
        print(f"   - Actual Status: {actual}")
        print(f"   - Model Prediction: {predicted} (Confidence: {prob:.2%})")
        
        # Check if the model prediction matches the actual status
        status = "✅ Correct" if actual == predicted else "❌ Wrong"
        print(f"   - Result: {status}")
        print("-" * 20)