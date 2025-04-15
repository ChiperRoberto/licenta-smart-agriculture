import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
import joblib
import torch
from torchvision import transforms
from models.cnn_model import CNNModel

# Configurare pagina
st.set_page_config(page_title="Smart Agriculture Assistant", layout="wide")
st.title("🌱 Smart Agriculture Assistant")

# Selectare funcționalitate
option = st.sidebar.selectbox("Alege funcționalitatea", [
    "Recomandare cultură (Iterația 1)",
    "Detectare boli din frunze (Iterația 2)",
    "Estimare producție cultură (Iterația 3)"
])

# === Iterația 1 ===
if option == "Recomandare cultură (Iterația 1)":
    st.subheader("Completează caracteristicile solului")
    N = st.slider("Azot (N)", 0, 150, 50)
    P = st.slider("Fosfor (P)", 0, 150, 50)
    K = st.slider("Potasiu (K)", 0, 200, 50)
    temperature = st.number_input("Temperatura (°C)", value=25.0)
    humidity = st.number_input("Umiditate (%)", value=70.0)
    ph = st.number_input("pH", value=6.5)
    rainfall = st.number_input("Precipitații (mm)", value=100.0)

    if st.button("Recomandă cultură"):
        model = joblib.load("models/XGBoost_model.pkl")
        features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        prediction = model.predict(features)[0]
        label_encoder = joblib.load("models/crop_label_encoder.pkl")
        print(label_encoder.classes_)
        predicted_crop = label_encoder.inverse_transform([prediction])[0]
        st.success(f"🌾 Cultura recomandată: **{predicted_crop.upper()}**")

# === Iterația 2 ===
elif option == "Detectare boli din frunze (Iterația 2)":
    st.subheader("Încarcă o imagine a frunzei")
    image_file = st.file_uploader("Imagine", type=["jpg", "jpeg", "png"])

    if image_file:
        image = Image.open(image_file).convert("RGB")
        st.image(image, caption="Imagine încărcată", use_column_width=True)

        if st.button("Identifică boala"):
            class_names = [
                "Bacterial leaf blight",
                "Brown spot",
                "Healthy corn",
                "Infected",
                "Leaf smut",
                "Pepper__bell___Bacterial_spot",
                "Pepper__bell___healthy",
                "Potato___Early_blight",
                "Potato___Late_blight",
                "Potato___healthy",
                "Tomato___Bacterial_spot",
                "Tomato___Early_blight",
                "Tomato___Late_blight",
                "Tomato___Leaf_Mold",
                "Tomato___Septoria_leaf_spot",
                "Tomato___Spider_mites_Two_spotted_spider_mite",
                "Tomato___Target_Spot",
                "Tomato___YellowLeaf_Curl_Virus",
                "Tomato___Tomato_mosaic_virus",
                "Tomato___healthy"
            ]
            model = CNNModel(num_classes=len(class_names), img_height=64, img_width=64)
            model.load_state_dict(torch.load("models/final_model.pth", map_location="cpu"))
            model.eval()

            transform = transforms.Compose([
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
            input_tensor = transform(image).unsqueeze(0)
            with torch.no_grad():
                outputs = model(input_tensor)
                _, predicted_class = torch.max(outputs, 1)
                prediction = class_names[predicted_class.item()]

            st.success(f"🩺 Boala detectată: **{prediction}**")

# === Iterația 3 ===
# === Iterația 3 ===
elif option == "Estimare producție cultură (Iterația 3)":
    st.subheader("Introdu caracteristicile parcelei")

    # Încarcă structura coloanelor
    reference_data = pd.read_csv("datasets/Processed_Database.csv")
    feature_columns = reference_data.drop(columns=["Yield of CT", "Yield of NT"]).columns

    user_input = {}

    # Identificăm toate coloanele Crop_*
    crop_columns = [col for col in feature_columns if col.startswith("Crop_")]
    other_features = [col for col in feature_columns if col not in crop_columns]

    # Dropdown pentru alegerea culturii
    selected_crop = st.selectbox("Selectează cultura:", [col.replace("Crop_", "") for col in crop_columns])
    for crop_col in crop_columns:
        user_input[crop_col] = 1 if crop_col == f"Crop_{selected_crop}" else 0

    # Input pentru celelalte caracteristici
    for feature in other_features:
        if reference_data[feature].dtype in [np.float64, np.int64]:
            value = st.number_input(f"{feature}", value=float(reference_data[feature].mean()))
        else:
            value = st.selectbox(f"{feature}", options=sorted(reference_data[feature].unique()))
        user_input[feature] = value

    input_df = pd.DataFrame([user_input])

    if st.button("Prezice Randamentul"):
        try:
            model_ct = joblib.load("models/best_model_Yield_of_CT_XGBoost.pkl")
            model_nt = joblib.load("models/best_model_Yield_of_NT_XGBoost.pkl")

            # Reordonăm inputul pentru a se potrivi exact cu modelul
            input_df = input_df.reindex(columns=model_ct.feature_names_in_, fill_value=0)

            yield_ct = model_ct.predict(input_df)[0]
            yield_nt = model_nt.predict(input_df)[0]

            st.success("Predicție completă!")
            st.metric(label="Randament CT (kg/ha)", value=f"{yield_ct:,.2f}")
            st.metric(label="Randament NT (kg/ha)", value=f"{yield_nt:,.2f}")

            better = "CT" if yield_ct > yield_nt else "NT"
            st.info(f"**Sistemul recomandat** pentru această parcelă este: `{better}`")

        except Exception as e:
            st.error(f"A apărut o eroare la predicție: {e}")