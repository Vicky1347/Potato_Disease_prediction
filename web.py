import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load model function
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("trained_plant_disease_model.keras")

model = load_model()

# Function to predict disease
def model_prediction(test_image):
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.expand_dims(input_arr, axis=0)
    predictions = model.predict(input_arr)
    return np.argmax(predictions)

# Sidebar Navigation
st.sidebar.image("agroai.jpeg", use_container_width=True)
st.sidebar.title("🌿 Potato Disease Detection")
st.sidebar.markdown("### Created by: **Vicky Kumar**")
app_mode = st.sidebar.radio("Navigation", ["🏠 Home", "🔬 Disease Recognition", "📖 Disease Info"])

# Home Page
if app_mode == "🏠 Home":
    st.markdown("<h1 style='text-align: center;'>🌱 Plant Disease Detection System for Sustainable Agriculture</h1>", unsafe_allow_html=True)
    st.image("agroai.jpeg", use_container_width=True)
    st.markdown("""
        **Why use this app?**  
        - Detect plant diseases early.  
        - Increase agricultural productivity.  
        - Support sustainable farming.
    """)

# Disease Recognition Page
elif app_mode == "🔬 Disease Recognition":
    st.header("📷 Upload an Image for Disease Prediction")

    test_image = st.file_uploader("Choose an image:", type=["jpg", "jpeg", "png"])

    if test_image:
        col1, col2 = st.columns(2)

        with col1:
            st.image(test_image, caption="Uploaded Image", use_container_width=True)

        with col2:
            if st.button("🔍 Predict"):
                with st.spinner("Analyzing..."):
                    result_index = model_prediction(test_image)
                    class_name = ["Potato - Early Blight", "Potato - Late Blight", "Healthy Potato"]
                    st.success(f"✅ Model Prediction: **{class_name[result_index]}**")

# Disease Info Page
elif app_mode == "📖 Disease Info":
    st.header("🌿 Understanding Potato Diseases")
    
    st.subheader("🦠 1. Early Blight (*Alternaria solani*)")
    st.image("early_Blight.jpg", caption="Early Blight Symptoms", use_container_width=True)
    st.markdown("""
    - **Cause**: Fungus *Alternaria solani*  
    - **Symptoms**:
      - Dark brown **circular spots** with **concentric rings** on leaves 🎯  
      - Lower leaves affected first  
      - Leaves turn **yellow and dry out**  
      - Tubers may have **dark, sunken spots**  
    - **Prevention**:
      - Use **resistant varieties**  
      - Apply **fungicides**  
      - Rotate crops to prevent disease spread  
    """)

    st.subheader("🔥 2. Late Blight (*Phytophthora infestans*)")
    st.image("late_blight.jpg", caption="Late Blight Symptoms", use_container_width=True)
    st.markdown("""
    - **Cause**: Water mold *Phytophthora infestans*  
    - **Symptoms**:
      - **Irregular water-soaked lesions** on leaves  
      - Leaves **turn brown and die**  
      - White fungal growth in humid conditions  
      - Tubers develop **soft rot and foul smell**  
    - **Prevention**:
      - Avoid overhead watering  
      - Remove infected plants immediately  
      - Apply **copper-based fungicides**  
    """)

    st.subheader("📊 Comparison: Early vs. Late Blight")
    st.markdown("""
    | Feature | Early Blight | Late Blight |
    |---------|-------------|------------|
    | **Pathogen** | *Alternaria solani* | *Phytophthora infestans* |
    | **Lesions** | Circular with rings 🎯 | Irregular, water-soaked |
    | **Spread** | Slower | Fast, can destroy entire crops |
    | **Conditions** | Warm, humid | Cool, wet |
    | **Severity** | Moderate | Very severe, highly destructive |
    """)

    st.success("🌱 Proper **prevention & early detection** can help protect your crops!")

