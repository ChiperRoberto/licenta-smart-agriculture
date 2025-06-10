import pandas as pd
import numpy as np
import joblib
import json

def load_features(json_path="models/reduced_feature_lists.json"):
    with open(json_path) as f:
        feat_dict = json.load(f)
    feature_columns = sorted(set(feat_dict["Yield of CT"]) | set(feat_dict["Yield of NT"]))
    crop_columns = [c for c in feature_columns if c.startswith("Crop_")]
    other_features = [c for c in feature_columns if c not in crop_columns]
    return feature_columns, crop_columns, other_features

def load_reference_data(columns, ct_path="datasets/Processed_Database_CT.csv", nt_path="datasets/Processed_Database_NT.csv"):
    df_ct = pd.read_csv(ct_path, usecols=lambda c: c in columns)
    df_nt = pd.read_csv(nt_path, usecols=lambda c: c in columns)
    return pd.concat([df_ct, df_nt], ignore_index=True)

def prepare_input(selected_crop, crop_columns, other_features, reference_data, form_data):
    user_input = {}
    for col in crop_columns:
        user_input[col] = 1 if col == f"Crop_{selected_crop}" else 0
    for feat in other_features:
        user_input[feat] = form_data[feat]
    return pd.DataFrame([user_input])

def predict_yields(input_df, model_ct_path, model_nt_path):
    model_ct = joblib.load(model_ct_path)
    model_nt = joblib.load(model_nt_path)

    input_ct = input_df.reindex(model_ct.feature_names_in_, axis=1, fill_value=0)
    input_nt = input_df.reindex(model_nt.feature_names_in_, axis=1, fill_value=0)

    yield_ct = model_ct.predict(input_ct)[0]
    yield_nt = model_nt.predict(input_nt)[0]
    return yield_ct, yield_nt
