import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np

# Kelas untuk memproses data diabetes
class DiabetesDataProcessor:
    def __init__(self, file_path):
        # Menyimpan path ke file CSV dan menginisialisasi objek scaler
        self.file_path = file_path
        self.data = None
        self.scaler = StandardScaler()

    def load_data(self):
        # Membaca file CSV dan menyimpannya ke dalam atribut data
        self.data = pd.read_csv(self.file_path)
        return self.data

    def handle_zeros(self):
        # Mengganti nilai nol di kolom tertentu dengan nilai median kolom tersebut
        cols_to_check = ['Glucose', 'BloodPressure', 'Insulin', 'BMI', 'Age']
        for col in cols_to_check:
            self.data[col] = self.data[col].replace(0, np.nan)  # Anggap nol sebagai nilai hilang
            median_val = self.data[col].median()                # Hitung median kolom
            self.data[col] = self.data[col].fillna(median_val)  # Isi NaN dengan median

    def preprocess(self):
        """
        Membersihkan dan menyiapkan data:
        - Menghapus kolom yang tidak digunakan
        - Menangani nilai nol
        - Memisahkan fitur dan label
        - Melakukan standarisasi fitur
        """
        if self.data is None:
            raise ValueError("Data belum dimuat. Panggil load_data() dulu.")

        # Drop kolom 'Pregnancies' dan 'SkinThickness' karena dianggap tidak relevan
        self.data = self.data.drop(columns=['Pregnancies', 'SkinThickness'])

        # Tangani nilai nol di kolom penting
        self.handle_zeros()

        # Pisahkan antara fitur (X) dan label (y)
        features = self.data.drop("Outcome", axis=1)
        labels = self.data["Outcome"]

        # Lakukan scaling terhadap fitur menggunakan StandardScaler
        features_scaled = self.scaler.fit_transform(features)

        return features_scaled, labels

    def transform_new_data(self, input_data):
        """
        Fungsi untuk mentransformasi input baru ke dalam bentuk yang sama dengan data pelatihan.
        Input: list atau array 1 dimensi
        Output: array 2 dimensi yang telah diskalakan
        """
        return self.scaler.transform([input_data])

# Bagian utama program untuk uji coba
if __name__ == "__main__":
    processor = DiabetesDataProcessor("diabetes.csv")      # Buat objek untuk preprocessing
    processor.load_data()                                  # Load data dari CSV
    features_scaled, labels = processor.preprocess()       # Preprocessing data
    print("Features shape:", features_scaled.shape)        # Cetak dimensi fitur setelah scaling
    print("Labels distribusi:\n", labels.value_counts())   # Tampilkan distribusi label 0-1