import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Import joblib
try:
    import joblib
except ImportError:
    from sklearn.externals import joblib

# Load model dan komponen
st.write("Memuat model...")
try:
    model = joblib.load('random_forest_model_compressed.pkl')
    st.write("Model dimuat dengan sukses!")
except Exception as e:
    st.write(f"Error memuat model: {e}")

try:
    scaler = pickle.load(open('scaler.pkl', 'rb'))
    label_encoders = pickle.load(open('label_encoders.pkl', 'rb'))
    le_target = pickle.load(open('target_encoder.pkl', 'rb'))
    selected_features = pickle.load(open('selected_features.pkl', 'rb'))
    st.write("Semua komponen model berhasil dimuat!")
except Exception as e:
    st.error(f"Error memuat komponen model: {e}")
    st.stop()

st.title("Aplikasi Prediksi Kategori Obesitas")
st.write("Masukkan data pasien untuk memprediksi kategori obesitas (Normal Weight, Overweight, Obesity).")

st.header("Input Data Pasien")

# Input
age = st.number_input("Usia", min_value=0, max_value=120, value=25)
gender = st.selectbox("Jenis Kelamin", ["Male", "Female"])
height = st.number_input("Tinggi Badan (cm)", min_value=50.0, max_value=250.0, value=165.0)
weight = st.number_input("Berat Badan (kg)", min_value=20.0, max_value=300.0, value=70.0)
favc = st.selectbox("Sering konsumsi makanan berkalori tinggi? (FAVC)", ["Yes", "No"])
fcvc = st.number_input("Frekuensi konsumsi sayuran (FCVC)", min_value=1.0, max_value=3.0, value=2.0)
ncp = st.number_input("Jumlah makan utama per hari (NCP)", min_value=1.0, max_value=4.0, value=3.0)
family_history = st.selectbox("Riwayat keluarga dengan obesitas?", ["Yes", "No"])
caec = st.selectbox("Konsumsi makanan antara waktu makan (CAEC)", ["Sometimes", "Frequently", "Always", "No"])
mtrans = st.selectbox("Metode transportasi utama (MTRANS)", ["Public_Transportation", "Walking", "Automobile", "Motorbike", "Bike"])
ch2o = st.number_input("Konsumsi air per hari (CH2O, liter)", min_value=0.0, max_value=5.0, value=2.0)
faf = st.number_input("Frekuensi aktivitas fisik (FAF, hari/minggu)", min_value=0.0, max_value=7.0, value=3.0)
tue = st.number_input("Waktu penggunaan teknologi (TUE, jam/hari)", min_value=0.0, max_value=24.0, value=2.0)
calc = st.selectbox("Konsumsi alkohol (CALC)", ["No", "Sometimes", "Frequently", "Always"])
scc = st.selectbox("Pemantauan kalori (SCC)", ["Yes", "No"])
smoke = st.selectbox("Merokok? (SMOKE)", ["Yes", "No"])

if st.button("Prediksi"):
    try:
        # Susun data input
        input_data = pd.DataFrame({
            'Age': [age],
            'Gender': [gender.lower()],
            'Height': [height],
            'Weight': [weight],
            'FAVC': [favc.lower()],
            'FCVC': [fcvc],
            'NCP': [ncp],
            'family_history_with_overweight': [family_history.lower()],
            'CAEC': [caec.lower()],
            'MTRANS': [mtrans.lower()],
            'CH2O': [ch2o],
            'FAF': [faf],
            'TUE': [tue],
            'CALC': [calc.lower()],
            'SCC': [scc.lower()],
            'SMOKE': [smoke.lower()]
        })

        # Transformasi label encoding
        categorical_cols = ['Gender', 'CALC', 'FAVC', 'SCC', 'SMOKE', 'family_history_with_overweight', 'CAEC', 'MTRANS']
        for col in categorical_cols:
            if col in label_encoders:
                input_value = input_data[col].iloc[0]
                if input_value not in label_encoders[col].classes_:
                    st.error(f"Nilai '{input_value}' untuk kolom '{col}' tidak dikenali oleh model.")
                    st.write(f"Nilai yang valid: {list(label_encoders[col].classes_)}")
                    st.stop()
                input_data[col] = label_encoders[col].transform(input_data[col])
            else:
                st.error(f"Label encoder untuk '{col}' tidak ditemukan!")
                st.stop()

        # Seleksi fitur dan scaling
        input_data = input_data[selected_features]
        input_data_scaled = scaler.transform(input_data)

        # Prediksi
        prediction = model.predict(input_data_scaled)
        prediction_label = le_target.inverse_transform(prediction)[0]

        st.success(f"Prediksi Kategori Obesitas: **{prediction_label}**")

    except Exception as e:
        st.error(f"Error saat melakukan prediksi: {e}")
        st.write("Pastikan semua file model tersedia dan input valid.")

st.write("---")
st.write("**Performa Model (Setelah Tuning):**")
st.write("- Random Forest: Accuracy 94.5%, Precision 94.3%, Recall 94.3%, F1-Score 0.96")
