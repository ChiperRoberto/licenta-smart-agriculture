import pandas as pd
import joblib

# Incarca modelul antrenat si encoderul de labeluri
model_path = "models/XGBoost_model.pkl"
label_encoder_path = "models/crop_label_encoder.pkl"

model = joblib.load(model_path)
label_encoder = joblib.load(label_encoder_path)


def recommend_crop(input_data: dict) -> str:
    """
    Functie care recomanda o cultura bazata pe caracteristicile solului si climatului.
    :param input_data: dictionar cu chei ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
    :return: cultura recomandata (str)
    """
    expected_features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']

    # Convertim dictionarul in DataFrame pentru predictie
    input_df = pd.DataFrame([input_data])

    # Verificam daca toate coloanele necesare sunt incluse
    if not all(col in input_df.columns for col in expected_features):
        raise ValueError(f"Lipsesc coloane in input_data. Coloanele necesare sunt: {expected_features}")

    # Predictie
    predicted_label_encoded = model.predict(input_df)[0]
    predicted_label = label_encoder.inverse_transform([predicted_label_encoded])[0]
    return predicted_label
