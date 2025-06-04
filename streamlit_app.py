import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Import joblib with error handling
try:
    import joblib
except ImportError:
    from sklearn.externals import joblib

st.title("Debug: Aplikasi Prediksi Kategori Obesitas")

# Load model components with detailed debugging
st.write("## 🔍 Debug: Loading Model Components")

@st.cache_data
def load_model_components():
    components = {}
    try:
        # Load model
        components['model'] = joblib.load('random_forest_model_compressed.pkl')
        st.success("✅ Model loaded successfully!")
        
        # Load scaler
        components['scaler'] = pickle.load(open('scaler.pkl', 'rb'))
        st.success("✅ Scaler loaded successfully!")
        
        # Load label encoders
        components['label_encoders'] = pickle.load(open('label_encoders.pkl', 'rb'))
        st.success("✅ Label encoders loaded successfully!")
        
        # Load target encoder
        components['le_target'] = pickle.load(open('target_encoder.pkl', 'rb'))
        st.success("✅ Target encoder loaded successfully!")
        
        # Load selected features
        components['selected_features'] = pickle.load(open('selected_features.pkl', 'rb'))
        st.success("✅ Selected features loaded successfully!")
        
        return components
        
    except Exception as e:
        st.error(f"❌ Error loading components: {e}")
        return None

# Load components
components = load_model_components()

if components is None:
    st.stop()

# Display detailed information about label encoders
st.write("## 📊 Label Encoders Information")

label_encoders = components['label_encoders']
selected_features = components['selected_features']

st.write("### Valid values for each categorical column:")
valid_values = {}
for col, encoder in label_encoders.items():
    valid_values[col] = list(encoder.classes_)
    st.write(f"**{col}**: {valid_values[col]}")

st.write("### Selected Features:")
st.write(selected_features)

st.write("---")
st.write("## 📝 Input Data Pasien")

# Create input fields using the actual valid values from label encoders
age = st.number_input("Usia", min_value=0, max_value=120, value=25)

# Gender
gender_options = valid_values.get('Gender', ['Male', 'Female'])
gender = st.selectbox("Jenis Kelamin", gender_options)

height = st.number_input("Tinggi Badan (cm)", min_value=50.0, max_value=250.0, value=165.0)
weight = st.number_input("Berat Badan (kg)", min_value=20.0, max_value=300.0, value=70.0)

# FAVC
favc_options = valid_values.get('FAVC', ['Yes', 'No'])
favc = st.selectbox("Sering konsumsi makanan berkalori tinggi? (FAVC)", favc_options)

fcvc = st.number_input("Frekuensi konsumsi sayuran (FCVC)", min_value=1.0, max_value=3.0, value=2.0)
ncp = st.number_input("Jumlah makan utama per hari (NCP)", min_value=1.0, max_value=4.0, value=3.0)

# Family history
family_options = valid_values.get('family_history_with_overweight', ['Yes', 'No'])
family_history = st.selectbox("Riwayat keluarga dengan obesitas?", family_options)

# CAEC
caec_options = valid_values.get('CAEC', ['Sometimes', 'Frequently', 'Always', 'No'])
caec = st.selectbox("Konsumsi makanan antara waktu makan (CAEC)", caec_options)

# MTRANS
mtrans_options = valid_values.get('MTRANS', ['Public_Transportation', 'Walking', 'Automobile', 'Motorbike', 'Bike'])
mtrans = st.selectbox("Metode transportasi utama (MTRANS)", mtrans_options)

ch2o = st.number_input("Konsumsi air per hari (CH2O, liter)", min_value=0.0, max_value=5.0, value=2.0)
faf = st.number_input("Frekuensi aktivitas fisik (FAF, hari/minggu)", min_value=0.0, max_value=7.0, value=3.0)
tue = st.number_input("Waktu penggunaan teknologi (TUE, jam/hari)", min_value=0.0, max_value=24.0, value=2.0)

# CALC
calc_options = valid_values.get('CALC', ['No', 'Sometimes', 'Frequently', 'Always'])
calc = st.selectbox("Konsumsi alkohol (CALC)", calc_options)

# SCC
scc_options = valid_values.get('SCC', ['Yes', 'No'])
scc = st.selectbox("Pemantauan kalori (SCC)", scc_options)

# SMOKE
smoke_options = valid_values.get('SMOKE', ['Yes', 'No'])
smoke = st.selectbox("Merokok? (SMOKE)", smoke_options)

if st.button("🔮 Prediksi"):
    try:
        # Prepare input data
        input_data = pd.DataFrame({
            'Age': [age],
            'Gender': [gender],
            'Height': [height],
            'Weight': [weight],
            'FAVC': [favc],
            'FCVC': [fcvc],
            'NCP': [ncp],
            'family_history_with_overweight': [family_history],
            'CAEC': [caec],
            'MTRANS': [mtrans],
            'CH2O': [ch2o],
            'FAF': [faf],
            'TUE': [tue],
            'CALC': [calc],
            'SCC': [scc],
            'SMOKE': [smoke]
        })
        
        st.write("### 🔍 Debug: Input Data Before Encoding")
        st.dataframe(input_data)
        
        # Encode categorical variables
        categorical_cols = ['Gender', 'CALC', 'FAVC', 'SCC', 'SMOKE', 'family_history_with_overweight', 'CAEC', 'MTRANS']
        
        for col in categorical_cols:
            if col in label_encoders:
                input_value = input_data[col].iloc[0]
                st.write(f"Encoding {col}: '{input_value}' -> ", end="")
                input_data[col] = label_encoders[col].transform(input_data[col])
                st.write(f"{input_data[col].iloc[0]}")
            else:
                st.error(f"Label encoder untuk kolom {col} tidak ditemukan!")
                st.stop()
        
        st.write("### 🔍 Debug: Input Data After Encoding")
        st.dataframe(input_data)
        
        # Select features
        if not all(feature in input_data.columns for feature in selected_features):
            missing_features = [f for f in selected_features if f not in input_data.columns]
            st.error(f"Missing features: {missing_features}")
            st.stop()
            
        input_data_selected = input_data[selected_features]
        st.write("### 🔍 Debug: Selected Features Data")
        st.dataframe(input_data_selected)
        
        # Scale data
        input_data_scaled = components['scaler'].transform(input_data_selected)
        st.write("### 🔍 Debug: Scaled Data")
        st.write(input_data_scaled)
        
        # Make prediction
        prediction = components['model'].predict(input_data_scaled)
        prediction_proba = components['model'].predict_proba(input_data_scaled)
        
        st.write("### 🔍 Debug: Raw Prediction")
        st.write(f"Prediction: {prediction}")
        st.write(f"Prediction Probabilities: {prediction_proba}")
        
        # Convert prediction to label
        prediction_label = components['le_target'].inverse_transform(prediction)[0]
        
        st.success(f"## 🎯 Hasil Prediksi: **{prediction_label}**")
        
        # Show probabilities
        class_labels = components['le_target'].classes_
        prob_dict = {label: prob for label, prob in zip(class_labels, prediction_proba[0])}
        
        st.write("### 📊 Probabilitas untuk setiap kategori:")
        for label, prob in prob_dict.items():
            st.write(f"- **{label}**: {prob:.4f} ({prob*100:.2f}%)")
        
    except Exception as e:
        st.error(f"❌ Error saat melakukan prediksi: {e}")
        st.write("**Debug info:**")
        st.write(f"Error type: {type(e).__name__}")
        import traceback
        st.code(traceback.format_exc())

st.write("---")
st.write("**Performa Model (Setelah Tuning):**")
st.write("- Random Forest: Accuracy 94.5%, Precision 94.3%, Recall 94.3%, F1-Score 0.96")
