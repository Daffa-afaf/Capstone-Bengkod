import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load semua file pickle
with open('random_forest_model_compressed.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('label_encoders.pkl', 'rb') as f:
    label_encoders = pickle.load(f)

with open('target_encoder.pkl', 'rb') as f:
    target_encoder = pickle.load(f)

with open('selected_features.pkl', 'rb') as f:
    selected_features = pickle.load(f)

# Judul aplikasi
st.title("Prediksi Kategori Obesitas")
st.write("Masukkan informasi pribadi untuk memprediksi kondisi obesitas berdasarkan model machine learning.")

# Input dari pengguna
def user_input():
    age = st.slider("Umur", 10, 100, 25)
    gender = st.selectbox("Jenis Kelamin", ["Male", "Female"])
    height = st.number_input("Tinggi (meter)", min_value=1.0, max_value=2.5, value=1.7)
    weight = st.number_input("Berat (kg)", min_value=30.0, max_value=200.0, value=70.0)
    favc = st.selectbox("Apakah Anda sering mengonsumsi makanan tinggi kalori?", ["yes", "no"])
    fcvc = st.slider("Frekuensi konsumsi sayur (1=jarang, 3=sering)", 1.0, 3.0, 2.0)
    ncp = st.slider("Jumlah makanan utama per hari", 1.0, 4.0, 3.0)
    fhwo = st.selectbox("Apakah ada riwayat keluarga dengan obesitas?", ["yes", "no"])
    caec = st.selectbox("Frekuensi konsumsi makanan antara waktu makan utama", ["no", "Sometimes", "Frequently", "Always"])
    mtrans = st.selectbox("Transportasi utama", ["Public_Transportation", "Walking", "Automobile", "Motorbike", "Bike"])

    # Buat dictionary
    user_data = {
        'Age': age,
        'Gender': gender,
        'Height': height,
        'Weight': weight,
        'FAVC': favc,
        'FCVC': fcvc,
        'NCP': ncp,
        'family_history_with_overweight': fhwo,
        'CAEC': caec,
        'MTRANS': mtrans
    }
    return pd.DataFrame([user_data])

# Ambil input
input_df = user_input()

# Encode kolom kategorikal
for col in label_encoders:
    if col in input_df.columns:
        input_df[col] = label_encoders[col].transform(input_df[col])

# Pilih dan susun ulang fitur
X_input = input_df[selected_features]

# Skalakan data
X_scaled = scaler.transform(X_input)

# Prediksi
if st.button("Prediksi"):
    pred = model.predict(X_scaled)
    pred_label = target_encoder.inverse_transform(pred)[0]
    st.success(f"Model memprediksi: **{pred_label}**")

    # Opsional: tambahkan probabilitas jika model mendukung
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_scaled)[0]
        proba_df = pd.DataFrame({
            'Kategori': target_encoder.inverse_transform(np.arange(len(proba))),
            'Probabilitas': proba
        }).sort_values(by='Probabilitas', ascending=False)
        st.subheader("Probabilitas Prediksi:")
        st.bar_chart(proba_df.set_index('Kategori'))

st.markdown("---")
st.markdown("Model ini menggunakan Random Forest yang telah dioptimasi dengan RandomizedSearchCV.")
