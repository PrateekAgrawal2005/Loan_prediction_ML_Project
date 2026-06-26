from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from PIL import Image
import io
import numpy as np

from color_quantization import quantify_image_colors


st.set_page_config(page_title="Loan Predictor", page_icon="💰", layout="centered")

MODEL_PATH = Path(__file__).with_name("loan_classifier_pipeline.pkl")

app_mode = st.sidebar.selectbox(
    "Choose App",
    ["Loan Prediction", "Color Quantization"],
)

# ===================== LOAN PREDICTION =====================
if app_mode == "Loan Prediction":
    st.title("💰 Loan Approval Prediction")

    if not MODEL_PATH.exists():
        st.error("Model file not found")
        st.stop()

    model = joblib.load(MODEL_PATH)

    st.header("Enter Applicant Details")

    gender = st.selectbox("Gender", ["Male", "Female"])
    married = st.selectbox("Married", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self Employed", ["Yes", "No"])
    property_area = st.selectbox("Property Area", ["Urban", "Rural", "Semiurban"])

    app_income = st.number_input("Applicant Income", min_value=0)
    coapp_income = st.number_input("Coapplicant Income", min_value=0)
    loan_amount = st.number_input("Loan Amount", min_value=0)
    loan_term = st.number_input("Loan Amount Term", min_value=0)
    credit_history = st.selectbox("Credit History", [1.0, 0.0])

    if st.button("Predict"):
        input_data = pd.DataFrame([{
            "ApplicantIncome": app_income,
            "CoapplicantIncome": coapp_income,
            "LoanAmount": loan_amount,
            "Loan_Amount_Term": loan_term,
            "Credit_History": credit_history,
            "Gender_Male": 1 if gender == "Male" else 0,
            "Married_Yes": 1 if married == "Yes" else 0,
            "Dependents_1": 1 if dependents == "1" else 0,
            "Dependents_2": 1 if dependents == "2" else 0,
            "Dependents_3+": 1 if dependents == "3+" else 0,
            "Education_Not Graduate": 1 if education == "Not Graduate" else 0,
            "Self_Employed_Yes": 1 if self_employed == "Yes" else 0,
            "Property_Area_Semiurban": 1 if property_area == "Semiurban" else 0,
            "Property_Area_Urban": 1 if property_area == "Urban" else 0,
        }])

        if hasattr(model, "feature_names_in_"):
            input_data = input_data[list(model.feature_names_in_)]

        prediction = model.predict(input_data)
        proba = model.predict_proba(input_data)

        if prediction[0] == 1:
            st.success(f"Loan Approved (Confidence: {proba[0][1] * 100:.2f}%)")
        else:
            st.error(f"Loan Rejected (Confidence: {proba[0][0] * 100:.2f}%)")


# ===================== COLOR QUANTIZATION =====================
else:
    st.title("🎨 Color Quantization")

    uploaded_file = st.file_uploader("Upload an image")
    n_clusters = st.slider("Number of colors", min_value=2, max_value=16, value=6)

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")

        # 🔥 Processing
        original_image, quantized_image = quantify_image_colors(image, n_clusters)

        # 📏 Dimensions
        h, w, c = quantized_image.shape
        st.write(f"📏 Dimensions: {w} x {h} pixels")

        # ✅ Real original size
        orig_size = len(uploaded_file.getvalue()) / 1024

        # ✅ Quantized size (correct)
        buffer_quant = io.BytesIO()
        Image.fromarray(quantized_image.astype("uint8")).save(buffer_quant, format="PNG")
        quant_size = len(buffer_quant.getvalue()) / 1024

        st.write(f"📦 Original Size: {orig_size:.2f} KB")
        st.write(f"🗜️ Quantized Size: {quant_size:.2f} KB")

        # 📉 Reduction
        reduction = ((orig_size - quant_size) / orig_size) * 100
        st.write(f"📉 Size Reduced: {reduction:.2f}%")

        # 🖼️ Show images
        col1, col2 = st.columns(2)
        col1.image(original_image, caption="Original", use_container_width=True)
        col2.image(quantized_image, caption="Quantized", use_container_width=True)

        # 🎨 Plot
        fig, ax = plt.subplots()
        ax.imshow(quantized_image)
        ax.axis("off")
        st.pyplot(fig)