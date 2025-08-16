import streamlit as st
import pandas as pd
from pycaret.classification import load_model, predict_model

# Set page title and icon
st.set_page_config(page_title="BMW Car Sales Prediction App", page_icon=":car:")

# Load the trained PyCaret model
# Make sure 'tuned_svm_model.pkl' is in the same directory as your app.py file,
# or provide the full path to the model file.
try:
    model = load_model('tuned_svm_model')
    
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop() # Stop the app if model loading fails

st.title("Aplikasi Prediksi Penjualan Mobil BMW")
st.write("Masukkan detail mobil untuk memprediksi Klasifikasi Penjualannya (High/Low). Anda bisa memasukkan data untuk beberapa mobil sekaligus.")

# --- Input Section ---
st.header("Masukkan Data Mobil")

# Option to upload a CSV file or input data manually
input_method = st.radio("Pilih metode input data:", ("Input Manual", "Upload File CSV"))

data_input = None

if input_method == "Input Manual":
    st.subheader("Input Data Secara Manual")
    # Create input fields for each feature
    # You can add more features based on your model's requirements
    model_name = st.selectbox("Model", ['5 Series', 'i8', 'X3', '7 Series', 'X5', 'M3', 'i3', 'X1', '4 Series', '6 Series']) # Replace with actual models from your data
    year = st.number_input("Tahun", min_value=2010, max_value=2024, value=2020)
    region = st.selectbox("Wilayah", ['Asia', 'North America', 'Middle East', 'South America', 'Europe', 'Africa']) # Replace with actual regions
    color = st.selectbox("Warna", ['Red', 'Blue', 'Black', 'Silver', 'White', 'Grey']) # Replace with actual colors
    fuel_type = st.selectbox("Jenis Bahan Bakar", ['Petrol', 'Hybrid', 'Diesel', 'Electric']) # Replace with actual fuel types
    transmission = st.selectbox("Transmisi", ['Manual', 'Automatic'])
    engine_size = st.number_input("Ukuran Mesin (L)", min_value=1.0, max_value=6.0, value=3.0, step=0.1)
    mileage = st.number_input("Jarak Tempuh (KM)", min_value=0, value=50000)
    price_usd = st.number_input("Harga (USD)", min_value=10000, value=50000)
    sales_volume = st.number_input("Volume Penjualan", min_value=100, value=5000)

    # Create a dictionary for the input data
    input_data_dict = {
        'Model': [model_name],
        'Year': [year],
        'Region': [region],
        'Color': [color],
        'Fuel_Type': [fuel_type],
        'Transmission': [transmission],
        'Engine_Size_L': [engine_size],
        'Mileage_KM': [mileage],
        'Price_USD': [price_usd],
        'Sales_Volume': [sales_volume]
    }
    data_input = pd.DataFrame(input_data_dict)

elif input_method == "Upload File CSV":
    st.subheader("Upload File CSV")
    uploaded_file = st.file_uploader("Pilih file CSV", type="csv")
    if uploaded_file is not None:
        try:
            data_input = pd.read_csv(uploaded_file)
            st.write("Data dari file CSV:")
            st.dataframe(data_input.head())
        except Exception as e:
            st.error(f"Error membaca file CSV: {e}")


# --- Prediction Section ---
if data_input is not None:
    if st.button("Prediksi"):
        try:
            # Make predictions using the loaded model
            predictions = predict_model(model, data=data_input)

            # Display the predictions
            st.subheader("Hasil Prediksi")
            # Rename the prediction column for clarity
            predictions = predictions.rename(columns={'prediction_label': 'Predicted_Sales_Classification'})
            st.dataframe(predictions[['Model', 'Year', 'Predicted_Sales_Classification']])

        except Exception as e:
            st.error(f"Terjadi kesalahan saat melakukan prediksi: {e}")