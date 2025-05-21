import streamlit as st
import numpy as np
from model import DiabetesModel
from Preprocessing import DiabetesDataProcessor

class DiabetesApp:
    def _init_(self):
        self.model = DiabetesModel()
        self.model.load_model("diabetes_model.pkl")  
        self.processor = DiabetesDataProcessor("diabetes.csv")
        self.processor.load_data()
        self.processor.preprocess()

    def run(self):
        st.markdown("""
        <style>
        html, body, [class*="css"]  {
            font-family: 'Arial Black', Gadget, sans-serif;
            background-color: #fefefe;
        }
        .block-container {
            border: 5px solid black;
            padding: 30px;
            background-color: #ffffff;
            box-shadow: 8px 8px 0px black;
        }
        .main-header {
            color: #fff;
            background: linear-gradient(90deg, #00c3ff 0%, #ffff1c 100%);
            border: 4px solid #000;
            box-shadow: 6px 6px 0px #000;
            padding: 20px 0;
            margin-bottom: 30px;
            text-align: center;
            font-size: 2.8em;
            font-family: 'Arial Black', Gadget, sans-serif;
            letter-spacing: 2px;
        }
        h2 {
            color: black;
            background-color: #ffd500;
            padding: 10px;
            border: 3px solid black;
            box-shadow: 4px 4px 0px black;
        }
        .stButton>button {
            color: black;
            background-color: #00ffff;
            border: 3px solid black;
            box-shadow: 4px 4px 0px black;
            font-weight: bold;
        }
        .stNumberInput>div>input {
            border: 2px solid black;
        }
        </style>
        """, unsafe_allow_html=True)

        st.markdown('<div class="main-header">🧪 Prediksi Diabetes</div>', unsafe_allow_html=True)

        st.subheader("Masukkan Data Pengguna")

        glucose = st.number_input("🩸 Glukosa", min_value=0)
        blood_pressure = st.number_input("💓 Tekanan Darah", min_value=0)
        insulin = st.number_input("💉 Insulin", min_value=0)
        bmi = st.number_input("📏 BMI", min_value=0.0)
        dpf = st.number_input("📊 Diabetes Pedigree Function", min_value=0.0)
        age = st.number_input("🎂 Umur", min_value=0)

        if st.button("🔍 Prediksi"):
            input_data = np.array([glucose, blood_pressure, insulin, bmi, dpf, age])
            input_scaled = self.processor.transform_new_data(input_data)
            result = self.model.predict(input_scaled[0])

            if result == 1:
                st.error("⚠ Hasil Prediksi: Berisiko Diabetes")
            else:
                st.success("✅ Hasil Prediksi: Tidak Berisiko Diabetes")


# Jalankan aplikasi
if __name__ == "__main__":
    app = DiabetesApp()
    app.run()
