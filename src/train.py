import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from src.data_loader import DataLoader as CustomDataLoader # Our custom data loader class
from src.model1 import ChurnModel # The improved ANN model
import joblib # Library for saving the scaler
import os # For directory management

def train():
    # 1. Data Preparation
    data_path = r'data/WA_Fn-UseC_-Telco-Customer-Churn.csv' 
    loader = CustomDataLoader(data_path, target_column='Churn')
    X_train, X_test, y_train, y_test = loader.prepare_data()

    # Convert data to PyTorch Tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train.values).view(-1, 1)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test.values).view(-1, 1)

    # Wrap data in PyTorch DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)

    # 2. Model Definition and Hyperparameters
    input_size = X_train.shape[1]
    model = ChurnModel(input_size)
    
    criterion = nn.BCELoss() # Binary Cross Entropy Loss
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5) 

    # 3. Training Loop
    epochs = 100
    print("Starting Training...")
    
    for epoch in range(epochs):
        model.train() # Set model to training mode
        epoch_loss = 0
        
        for batch_X, batch_y in train_loader:
            # Forward pass
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            # Backward pass and optimization
            optimizer.zero_grad() 
            loss.backward()        
            optimizer.step()       
            
            epoch_loss += loss.item()
        
        # Log loss every 10 epochs
        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss/len(train_loader):.4f}')

    # 4. Save Model and Scaler
    # Ensure 'models' directory exists
    if not os.path.exists('models'):
        os.makedirs('models')

    # Save model state dictionary
    torch.save(model.state_dict(), 'models/best_model.pt')
    
    # Save the scaler for inference use
    joblib.dump(loader.scaler, 'models/scaler.pkl') 
    
    print("--- Success! ---")
    print("Model saved to: models/best_model.pt")
    print("Scaler saved to: models/scaler.pkl")

if __name__ == "__main__":
    train()