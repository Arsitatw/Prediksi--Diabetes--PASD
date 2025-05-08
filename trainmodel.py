from Preprocessing import DiabetesDataProcessor
from model import DiabetesModel

#inisialisasi dan preprocessing data
processor = DiabetesDataProcessor("diabetes.csv")
processor.load_data()

#melakukan preprocessing data (mengatasi nilai nol, scaling, dsb.)
X, y = processor.preprocess()

#melatih model dengan data yang sudah diproses, lalu simpan akurasinya 
model = DiabetesModel()
acc = model.train(X, y)

#menyimpan model
model.save_model("model_diabetes.pkl")
print(f"Akurasi model: {acc:.2f}")
