import joblib
import numpy as np

def load_model(path="models/NB_model.pkl"):
    return joblib.load(path)

def load_label_encoder(path="models/crop_label_encoder.pkl"):
    return joblib.load(path)

def predict_crop(N, P, K, temperature, humidity, ph, rainfall, model, label_encoder):
    features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    prediction = model.predict(features)[0]
    crop = label_encoder.inverse_transform([prediction])[0]
    return crop.upper()
