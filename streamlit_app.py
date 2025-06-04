import streamlit as st
import pickle
import joblib
import numpy as np
import pandas as pd

# --- Caching untuk load model dan komponen preprocessing ---
@st.cache_resource
def load_model():
    model = joblib.load("random_forest_model_compressed.pkl")
    return model

@st.cache_resource
def load_pickle(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)

# Load model dan komponen preprocessing
model = load_model()
scaler = load_pickle("scaler.pkl")
label_encoders = load_pickle("label_encoders.pkl")
le_target = load_pickle("target_encoder.pkl")
selected_features = load_pickle("selected_features.pkl")

# --- UI Judul ---
st.title("Aplikasi Prediksi Kategori Obesitas")
st.write("Masukkan data pasien untuk memprediksi kategori obesitas berdasarkan model Random Forest.")

# --- Input Form ---
st.header("📋 Input Data Pasien")
age = st.number_input("Usia", min_value=0, max_value=120, value=25)
gender = st.selectbox("Jenis Kelamin", ["Male", "Female"])
height = st.number_input("Tinggi Badan (cm)", min_value=50.0, max_value=250.0, value=165.0)
weight = st.number_input("Berat Badan (kg)", min_value=20.0, max_value=300.0, value=70.0)
favc = st.selectbox("Sering konsumsi makanan berkalori tinggi? (FAVC)", ["yes", "no"])
fcvc = st.number_input("Frekuensi konsumsi sayuran (FCVC)", min_value=1.0, max_value=3.0, value=2.0)
ncp = st.number_input("Jumlah makan utama per hari (NCP)", min_value=1.0, max_value=4.0, value=3.0)
family_history = st.selectbox("Riwayat keluarga dengan obesitas?", ["yes", "no"])
caec = st.selectbox("Konsumsi makanan antara waktu makan (CAEC)", ["Sometimes", "Frequently", "Always", "no"])
mtrans = st.selectbox("Metode transportasi utama (MTRANS)", ["Public_Transportation", "Walking", "Automobile", "Motorbike", "Bike"])
ch2o = st.number_input("Konsumsi air per hari (CH2O, liter)", min_value=0.0, max_value=5.0, value=2.0)
faf = st.number_input("Frekuensi aktivitas fisik (FAF, hari/minggu)", min_value=0.0, max_value=7.0, value=3.0)
tue = st.number_input("Waktu penggunaan teknologi (TUE, jam/hari)", min_value=0.0, max_value=24.0, value=2.0)
calc = st.selectbox("Konsumsi alkohol (CALC)", ["no", "Sometimes", "Frequently", "Always"])
scc = st.selectbox("Pemantauan kalori (SCC)", ["yes", "no"])
smoke = st.selectbox("Merokok? (SMOKE)", ["yes", "no"])

# --- Prediksi ---
if st.button("🔍 Prediksi"):
    # Siapkan DataFrame input
    input_data = pd.DataFrame([{
        'Age': age,
        'Gender': gender,
        'Height': height,
        'Weight': weight,
        'FAVC': favc,
        'FCVC': fcvc,
        'NCP': ncp,
        'family_history_with_overweight': family_history,
        'CAEC': caec,
        'MTRANS': mtrans,
        'CH2O': ch2o,
        'FAF': faf,
        'TUE': tue,
        'CALC': calc,
        'SCC': scc,
        'SMOKE': smoke
    }])

    # Encode kolom kategorikal
    for col in label_encoders:
        if col in input_data.columns:
            input_data[col] = label_encoders[col].transform(input_data[col])

    # Pilih fitur yang digunakan oleh model
    input_data = input_data[selected_features]

    # Skalakan
    input_scaled = scaler.transform(input_data)

    # Prediksi
    prediction = model.predict(input_scaled)
    prediction_label = le_target.inverse_transform(prediction)[0]

    st.success(f"✅ Prediksi Kategori Obesitas: **{prediction_label}**")

    # Tampilkan probabilitas
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(input_scaled)[0]
        proba_df = pd.DataFrame({
            'Kategori': le_target.inverse_transform(np.arange(len(proba))),
            'Probabilitas': proba
        }).sort_values(by='Probabilitas', ascending=False)

        st.subheader("📊 Probabilitas Kategori:")
        st.bar_chart(proba_df.set_index("Kategori"))

# --- Informasi Tambahan ---
st.markdown("---")
st.markdown("📌 **Model ini menggunakan Random Forest Classifier** yang telah dioptimasi dengan RandomizedSearchCV.")
st.markdown("📈 **Performa Model (CV Score):**")
st.write("- Akurasi: 94.5%")
st.write("- Precision: 94.3%")
st.write("- Recall: 94.3%")
st.write("- F1-Score: 0.96")
