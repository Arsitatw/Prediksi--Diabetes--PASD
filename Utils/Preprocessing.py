import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
import os

# Kelas untuk memproses data diabetes
class DiabetesDataProcessor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.scaler = StandardScaler()

    def load_data(self):
        self.data = pd.read_csv(self.file_path)
        return self.data

    def handle_zeros(self):
        cols_to_check = ['Glucose', 'BloodPressure', 'Insulin', 'BMI', 'Age']
        for col in cols_to_check:
            self.data[col] = self.data[col].replace(0, np.nan)
            median_val = self.data[col].median()
            self.data[col] = self.data[col].fillna(median_val)

    def preprocess(self):
        if self.data is None:
            raise ValueError("Data belum dimuat. Panggil load_data() dulu.")
        
        self.data = self.data.drop(columns=['Pregnancies', 'SkinThickness'])
        self.handle_zeros()
        
        features = self.data.drop("Outcome", axis=1)
        labels = self.data["Outcome"]

        features_scaled = self.scaler.fit_transform(features)
        return features_scaled, labels

    def transform_new_data(self, input_data):
        return self.scaler.transform([input_data])

# Cek jalankan file ini langsung
if __name__ == "__main__":
    current_dir = os.path.dirname(__file__)
    csv_path = os.path.abspath(os.path.join(current_dir, "..", "Dataset", "diabetes.csv"))
    processor = DiabetesDataProcessor(csv_path)
    processor.load_data()
    X_scaled, y = processor.preprocess()
    print("X shape:", X_scaled.shape)
    print("y distribusi:\n", y.value_counts())
