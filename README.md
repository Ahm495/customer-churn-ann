# 🚀 Customer Churn Prediction (81% Recall)
### Predicting Telecom Churn with Deep Learning & Decision Optimization

---

## 📌 Project Overview
Customer churn is a critical challenge for telecom companies. This project implements a **Deep Neural Network (ANN)** built with **PyTorch** to identify high-risk customers.

The primary focus was **maximizing Recall**, ensuring the business can proactively intervene before a customer leaves.

---

## 📊 Business Impact & Results
Unlike standard models that focus only on Accuracy, this model is optimized for the **Recall of the Churn class**, achieving:

* **Recall (Class 1): 81%** (We successfully identify 81% of customers who actually churn).
* **Optimized Threshold:** Adjusted decision boundary to **0.35** (instead of 0.5) to balance business cost vs. customer retention.
* **Overall Accuracy:** 71.22%

---

## 🧠 Model Architecture (ANN)
A custom 4-layer Deep Learning model designed to handle feature complexity:
* **Input Layer:** 23 features (after preprocessing & one-hot encoding).
* **Hidden Layers:** Three dense layers with **ReLU** activation and **Dropout (0.3)** to prevent overfitting.
* **Batch Normalization:** Applied to ensure faster convergence and stability.
* **Output Layer:** **Sigmoid** activation for binary classification.

---

## 🛠️ Tech Stack
* **Deep Learning:** PyTorch
* **Data Processing:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn (MinMaxScaler, SMOTE)
* **Visualization:** Matplotlib, Seaborn
* **Workflow:** Modular Python scripts (`src/`) & Jupyter Notebooks.

---

## 📂 Project Structure
```plaintext
customer_churn_ann/
├── data/               # Dataset (Telco Churn)
├── models/             # Saved model weights (.pt) & Scalers (.pkl)
├── notebook/           # Exploratory Data Analysis (EDA)
├── src/                # Modular Python source code
│   ├── data_loader.py  # Data cleaning and pipeline
│   ├── model1.py       # ANN Architecture
│   ├── train.py        # Training & Validation loop
│   ├── predict.py      # Real-time inference script
│   └── evaluate.py     # Metrics and Confusion Matrix
├── requirements.txt    # Project dependencies
└── README.md           # Project documentation