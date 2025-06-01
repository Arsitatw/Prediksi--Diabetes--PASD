# train.py

from Utils.Preprocessing import DiabetesDataProcessor  # Sesuaikan folder dan nama file
from Models.model import DiabetesModel                 # Sesuaikan folder dan nama file

# Inisialisasi dan preprocessing data
processor = DiabetesDataProcessor("Dataset/diabetes.csv")  # Pastikan path benar sesuai struktur folder kamu
processor.load_data()
features, labels = processor.preprocess()

# Inisialisasi dan latih model
model = DiabetesModel()
acc = model.train(features, labels)  # Latih model dan dapatkan akurasi

# Simpan model
model.save_RF("Models/diabetes_model.pkl")  # Simpan model ke dalam folder (jika belum ada, buat dulu)
print(f"Akurasi model: {acc:.2f}")
