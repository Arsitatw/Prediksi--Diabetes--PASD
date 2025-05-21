from Preprocessing import DiabetesDataProcessor #mengimport data yg sudah di proses dari modul preprocessing
from model import DiabetesModel #sama dari model untuk membuat dan melatih model

# Inisialisasi dan preprocessing data
processor = DiabetesDataProcessor("diabetes.csv")
processor.load_data()
features, labels = processor.preprocess()

# Inisialisasi dan latih model
model = DiabetesModel()
acc = model.train(features, labels) #pembagian fitur dan target untuk output yang ingin di prediksi

# Simpan model
model.save_model("diabetes_model.pkl") #perintah untuk menyimpan model yang sudah di latih
print(f"Akurasi model: {acc:.2f}") #print akurasi model
