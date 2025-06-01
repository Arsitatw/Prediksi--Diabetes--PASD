import streamlit as st
import numpy as np
from Models.model import DiabetesModel
from Utils.Preprocessing import DiabetesDataProcessor

class DiabetesApp:
    def __init__(self):  
        # Inisialisasi model dan preprocessing data
        self.model = DiabetesModel()
        self.model.load_model("Models/diabetes_model.pkl")  
        self.processor = DiabetesDataProcessor("Dataset/diabetes.csv") 
        self.processor.load_data()
        self.processor.preprocess()

    def run(self):
        # Custom CSS untuk mempercantik tampilan aplikasi Streamlit
        st.markdown("""
        <style>
        /* Mengatur font utama dan warna latar belakang aplikasi */
        html, body, [class*="css"]  {
            font-family: 'Arial Black', Gadget, sans-serif;
            background-color: #fefefe;
        }
        /* Membuat border tebal dan bayangan pada kontainer utama */
        .block-container {
            border: 5px solid black;
            padding: 30px;
            background-color: #ffffff;
            box-shadow: 8px 8px 0px black;
        }
       
        /* Header utama aplikasi dengan warna biru polos, border, dan bayangan */
        .main-header {
         color: #fff;
        background: #00c3ff; /* Warna biru polos */
        border: 4px solid #000;
        box-shadow: 6px 6px 0px #000;
        padding: 20px 0;
        margin-bottom: 30px;
        text-align: center;
        font-size: 2.8em;
        font-family: 'Arial Black', Gadget, sans-serif;
        letter-spacing: 2px;
        }

        }
        /* Subheader dengan warna kuning dan border hitam */
        h2 {
            color: black;
            background-color: #ffd500;
            padding: 10px;
            border: 3px solid black;
            box-shadow: 4px 4px 0px black;
        }
        /* Tampilan tombol prediksi agar lebih menarik */
        .stButton>button {
            color: black;
            background-color: #00ffff;
            border: 3px solid black;
            box-shadow: 4px 4px 0px black;
            font-weight: bold;
        }
        /* Border pada input angka agar lebih jelas */
        .stNumberInput>div>input {
            border: 2px solid black;
        }
        </style>
        """, unsafe_allow_html=True)

        # Header aplikasi
        st.markdown('<div class="main-header">🧪 Prediksi Diabetes</div>', unsafe_allow_html=True)

        # Subheader untuk input data pengguna
        st.subheader("Masukkan Data Pengguna")

        # Input fitur dari pengguna
        glucose = st.number_input("🩸 Glukosa", min_value=0)
        blood_pressure = st.number_input("💓 Tekanan Darah", min_value=0)
        insulin = st.number_input("💉 Insulin", min_value=0)
        bmi = st.number_input("📏 BMI", min_value=0.0)
        dpf = st.number_input("📊 Diabetes Pedigree Function", min_value=0.0)
        age = st.number_input("🎂 Umur", min_value=0)

        # Tombol prediksi
        if st.button("🔍 Prediksi"):
            # Gabungkan input menjadi array dan lakukan scaling
            input_data = np.array([glucose, blood_pressure, insulin, bmi, dpf, age])
            input_scaled = self.processor.transform_new_data(input_data)
            result = self.model.predict(input_scaled[0])

            # Hitung probabilitas terkena diabetes
            probability = self.model.model.predict_proba(input_scaled)[0][1] * 100  # dalam persen

            # Tampilkan hasil prediksi dan probabilitas saja (tanpa akurasi)
            if result == 1:
                st.error(f"⚠ Hasil Prediksi: Berisiko Diabetes\n\nPeluang terkena diabetes: {probability:.2f}%")
            else:
                st.success(f"✅ Hasil Prediksi: Tidak Berisiko Diabetes\n\nPeluang terkena diabetes: {probability:.2f}%")

# Jalankan aplikasi
if __name__ == "__main__":
    app = DiabetesApp()
    app.run()
