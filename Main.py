import streamlit as st
import numpy as np

from model import DiabetesModel
from Preprocessing import DiabetesDataProcessor

class DiabetesApp:
    def __init__(self):
        self.model = DiabetesModel()
        self.model.load_model("diabetes_model.pkl")  
        self.processor = DiabetesDataProcessor("diabetes.csv")
        self.processor.load_data()
        self.processor.preprocess() 

    def run(self):
        st.title("Prediksi Diabetes ")
        
        # Input pengguna
    
        glucose = st.number_input("Glukosa", min_value=0)
        blood_pressure = st.number_input("Tekanan Darah", min_value=0)
        
        insulin = st.number_input("Insulin", min_value=0)
        bmi = st.number_input("BMI", min_value=0.0)
        dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0)
        age = st.number_input("Umur", min_value=0)

        # Button untuk memulai prediksi
        if st.button("Prediksi"):
            input_data = np.array([ glucose, blood_pressure,
                                    insulin, bmi, dpf, age])

            # Preprocess input
            input_scaled = self.processor.transform_new_data(input_data)

            # Prediksi
            result = self.model.predict(input_scaled[0])

            # Tampilkan hasil
            if result == 1:
                st.error("⚠️ Hasil Prediksi: Berisiko Diabetes")
            else:
                st.success("✅ Hasil Prediksi: Tidak Berisiko Diabetes")

# Jalankan aplikasi
if __name__ == "__main__":
    app = DiabetesApp()
    app.run()
