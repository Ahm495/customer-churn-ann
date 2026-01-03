import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from src.data_loader import DataLoader as CustomDataLoader
from src.model1 import ChurnModel

def evaluate():
    # 1. Load Data (Same procedure as in training)
    data_path = r'data/WA_Fn-UseC_-Telco-Customer-Churn.csv'
    loader = CustomDataLoader(data_path, target_column='Churn')
    _, X_test, _, y_test = loader.prepare_data()

    # Convert Test Set to PyTorch Tensors
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_true = y_test.values

    # 2. Load the Saved Model Weights
    input_size = X_test.shape[1]
    model = ChurnModel(input_size)
    model.load_state_dict(torch.load('models/best_model.pt'))
    
    # Crucial step: Set model to evaluation mode (disables Dropout and Batch Norm)
    model.eval() 

    # 3. Prediction Phase
    # Disable gradient calculation to save memory and compute power
    with torch.no_grad(): 
        outputs = model(X_test_tensor)
        # Convert probabilities to binary outcomes using the optimized 0.35 threshold
        y_pred = (outputs >= 0.35).float().numpy()

    # 4. Metrics Calculation
    print("         === Classification Report ===")
    print(classification_report(y_test_true, y_pred))
    
    acc = accuracy_score(y_test_true, y_pred)
    print(f"Overall Accuracy: {acc*100:.2f}%")

    # 5. Visualize Confusion Matrix (To analyze False Positives vs False Negatives)
    cm = confusion_matrix(y_test_true, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

if __name__ == "__main__":
    evaluate()