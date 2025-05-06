import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np

class DiabetesDataProcessor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.scaler = StandardScaler()

    def load_data(self):
        self.data = pd.read_csv(self.file_path)
        return self.data

    def handle_zeros(self):
        cols_to_check = ['Glucose', 'BloodPressure', 'Insulin', 'BMI','Age']
        for col in cols_to_check:
            self.data[col] = self.data[col].replace(0, np.nan)
            median_val = self.data[col].median()
            self.data[col] = self.data[col].fillna(median_val)

    def preprocess(self):
        """Pisahkan fitur dan label, lalu lakukan scaling."""
        if self.data is None:
            raise ValueError("Data belum dimuat. Panggil load_data() dulu.")
        
        self.handle_zeros()
        X = self.data.drop("Outcome", axis=1)
        y = self.data["Outcome"]

        X_scaled = self.scaler.fit_transform(X)
        return X_scaled, y

    def transform_new_data(self, input_data):
        """Transformasi data input baru (1D array) ke bentuk terstandarisasi."""
        return self.scaler.transform([input_data])
    
if __name__ == "__main__":
    processor = DiabetesDataProcessor("diabetes.csv")
    processor.load_data()
    X_scaled, y = processor.preprocess()
    print("X shape:", X_scaled.shape)
    print("y distribusi:\n", y.value_counts())


