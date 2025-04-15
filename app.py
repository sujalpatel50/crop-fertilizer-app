import streamlit as st
import numpy as np
import pickle

# Load models and encoders with error handling
try:
    crop_model = pickle.load(open("crop_model.pkl", "rb"))
    fertilizer_model = pickle.load(open("fertilizer_model.pkl", "rb"))
    scaler_crop = pickle.load(open("scaler_crop.pkl", "rb"))
    label_encoder_crop = pickle.load(open("label_encoder.pkl", "rb"))
    label_encoder_soil = pickle.load(open("label_encoder_soil.pkl", "rb"))
    label_encoder_fertilizer = pickle.load(open("label_encoder_fertilizer.pkl", "rb"))
    label_encoder_crop_fertilizer = pickle.load(open("label_encoder_crop.pkl", "rb"))
    soil_classes = pickle.load(open("soil_classes.pkl", "rb"))
    fertilizer_classes = pickle.load(open("fertilizer_classes.pkl", "rb"))

except Exception as e:
    st.error(f"An error occurred while loading pickled files: {e}")
    st.stop()

# Update label_encoder_crop_fertilizer if needed
all_possible_crops = np.array(['rice', 'maize', 'chickpea', 'kidneybeans', 'pigeonpeas', 
                               'mothbeans', 'mungbean', 'blackgram', 'lentil', 'pomegranate', 
                               'banana', 'mango', 'grapes', 'watermelon', 'muskmelon', 
                               'apple', 'orange', 'papaya', 'coconut', 'cotton', 'jute', 'coffee'])

if len(set(all_possible_crops) - set(label_encoder_crop_fertilizer.classes_)) > 0:
    label_encoder_crop_fertilizer.fit(all_possible_crops)
    pickle.dump(label_encoder_crop_fertilizer, open("D:/ME research work/label_encoder_crop.pkl", "wb"))

# Streamlit App UI
st.title("🌾 Crop Prediction and Fertilizer Recommendation System 🌱")

# Collect user inputs
st.header("Input Parameters")
N = st.number_input("Nitrogen (kg/ha)", min_value=0, max_value=300, value=90)
P = st.number_input("Phosphorous (kg/ha)", min_value=0, max_value=300, value=42)
K = st.number_input("Potassium (kg/ha)", min_value=0, max_value=300, value=43)
temperature = st.number_input("Temperature (°C)", min_value=-10.0, max_value=60.0, value=20.0)
humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=82.0)
ph = st.number_input("pH Level", min_value=0.0, max_value=14.0, value=6.5)
rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=500.0, value=200.0)
soil_type = st.selectbox("Soil Type", options=soil_classes)

if st.button("Predict Crop and Recommend Fertilizer"):
    # Crop Prediction
    crop_input = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    crop_input_scaled = scaler_crop.transform(crop_input)
    crop_pred_encoded = crop_model.predict(crop_input_scaled)[0]
    predicted_crop = label_encoder_crop.inverse_transform([crop_pred_encoded])[0]

    # Check for unseen crop labels
    if predicted_crop not in label_encoder_crop_fertilizer.classes_:
        st.error(f"Crop prediction contains an unseen crop label: {predicted_crop}. "
                 f"Please update your models and encoders.")
    else:
        # Fertilizer Recommendation (Include the full 9 features)
        soil_encoded = label_encoder_soil.transform([soil_type])[0]
        crop_encoded_fertilizer = label_encoder_crop_fertilizer.transform([predicted_crop])[0]
        fertilizer_input = np.array([[N, P, K, temperature, humidity, ph, rainfall, soil_encoded, crop_encoded_fertilizer]])
        fertilizer_pred_encoded = fertilizer_model.predict(fertilizer_input)[0]
        recommended_fertilizer = label_encoder_fertilizer.inverse_transform([fertilizer_pred_encoded])[0]

        # Display the results
        st.success(f"🌾 Predicted Crop: **{predicted_crop}**")
        st.success(f"🌱 Recommended Fertilizer: **{recommended_fertilizer}**")
