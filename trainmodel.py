from Preprocessing import DiabetesDataProcessor
from model import DiabetesModel

# Inisialisasi dan preprocessing data
processor = DiabetesDataProcessor("diabetes.csv")
processor.load_data()
X, y = processor.preprocess()

# Inisialisasi dan latih model
model = DiabetesModel()
acc = model.train(X, y)

# Simpan model
model.save_model("diabetes_model.pkl")
print(f"Akurasi model: {acc:.2f}")
