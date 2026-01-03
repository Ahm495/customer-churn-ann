import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from imblearn.over_sampling import SMOTE

class DataLoader:
    def __init__(self, file_path, target_column):
        self.file_path = file_path
        self.target_column = target_column
        self.scaler = MinMaxScaler()

    def load_data(self):
        return pd.read_csv(self.file_path)

    def _clean_data(self, data):
        """Clean basic data and drop unnecessary identifiers"""
        if 'customerID' in data.columns:
            data = data.drop('customerID', axis=1)
        
        # Convert TotalCharges to numeric and handle NaN values
        data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
        data = data.dropna(subset=['TotalCharges'])
        return data

    def _simplify_and_map(self, data):
        """Simplify categories and map Yes/No values to binary (0/1)"""
        # 1. Simplify service columns (Convert 'No internet/phone service' to 'No')
        service_cols = ['MultipleLines', 'OnlineSecurity', 'OnlineBackup', 
                        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
        for col in service_cols:
            data[col] = data[col].replace({'No internet service': 'No', 'No phone service': 'No'})
        
        # 2. Map Gender column
        if 'gender' in data.columns:
            data['gender'] = data['gender'].map({'Female': 1, 'Male': 0})
            
        # 3. Map Yes/No to Binary for all relevant object columns (including Target)
        binary_cols = data.select_dtypes(include=['object']).columns
        for col in binary_cols:
            # Ensure the column contains Yes/No values before mapping
            if set(data[col].unique()).issubset({'Yes', 'No', 'Female', 'Male'}):
                data[col] = data[col].map({'Yes': 1, 'No': 0})
        return data

    def _encode_data(self, data):
        """Apply One-Hot Encoding to remaining categorical columns (Contract, InternetService, etc.)"""
        # Identify remaining categorical columns
        cat_cols = data.select_dtypes(include=['object']).columns.tolist()
        if self.target_column in cat_cols:
            cat_cols.remove(self.target_column)
            
        return pd.get_dummies(data, columns=cat_cols, drop_first=True)

    def prepare_data(self):
        """Main pipeline: Load, Clean, Map, Encode, Split, Handle Imbalance, and Scale"""
        df = self.load_data()
        df = self._clean_data(df)
        df = self._simplify_and_map(df)
        df = self._encode_data(df)
        
        # Split features and target
        X = df.drop(columns=[self.target_column])
        y = df[self.target_column]
        
        # Train-Test Split with Stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Handle Class Imbalance using SMOTE
        smote = SMOTE(random_state=42)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
        
        # Scale Numerical Features
        X_train_final = self.scaler.fit_transform(X_train_res)
        X_test_final = self.scaler.transform(X_test)
        
        return X_train_final, X_test_final, y_train_res, y_test