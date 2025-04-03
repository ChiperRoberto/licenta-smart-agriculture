import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import os
import numpy as np

from models.cnn_model import CNNModel  # importăm modelul tău deja antrenat

# === Config ===
MODEL_PATH = 'models/best_model.pth'
CLASS_NAMES = ['Apple___Black_rot', 'Apple___healthy', 'Corn_(maize)___Cercospora_leaf_spot',
               'Corn_(maize)___healthy', 'Grape___Black_rot', 'Potato___Early_blight',
               'Potato___Late_blight', 'Potato___healthy', 'Tomato___Bacterial_spot',
               'Tomato___Early_blight', 'Tomato___healthy']  # <-- adaugă clasele reale din setul tău
IMG_SIZE = 64  # aceeași dimensiune ca la antrenare

# === Funcții auxiliare ===
@st.cache_resource
def load_model():
    model = CNNModel(num_classes=len(CLASS_NAMES), img_height=IMG_SIZE, img_width=IMG_SIZE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()
    return model

def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    image = transform(image).unsqueeze(0)  # shape: [1, 3, H, W]
    return image

# === Interfață Streamlit ===
st.title("🩺 Detecția bolilor în culturi (din imagini cu frunze)")

uploaded_file = st.file_uploader("Încarcă o imagine cu o frunză", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Imagine încărcată", use_column_width=True)

    st.write("🔄 Se procesează imaginea...")

    model = load_model()
    input_tensor = preprocess_image(image)
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted_class = torch.max(output, 1)
        prediction = CLASS_NAMES[predicted_class.item()]

    st.success(f"✅ Predicție: **{prediction}**")