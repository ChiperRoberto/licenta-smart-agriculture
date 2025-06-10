# tests/test_iteratia3.py
import pytest
import pandas as pd
import numpy as np
from app.utils.iteratia3 import load_features, load_reference_data, prepare_input, predict_yields

def test_load_features_and_columns(tmp_json_features):
    """
    Verificăm că load_features returnează corect listele de coloane
    (feature_columns, crop_columns, other_features).
    """
    feature_columns, crop_columns, other_features = load_features(json_path=tmp_json_features)
    assert set(feature_columns) == {"f1", "Crop_Grâu", "f2", "Crop_Orez"}
    assert set(crop_columns) == {"Crop_Grâu", "Crop_Orez"}
    assert set(other_features) == {"f1", "f2"}

def test_load_reference_data_success(dummy_reference_csv):
    """
    Verificăm că load_reference_data concatenează corect două CSV-uri.
    """
    ct_path, nt_path = dummy_reference_csv
    combined = load_reference_data(columns=["f1", "f2"], ct_path=ct_path, nt_path=nt_path)
    # CT: 2 rânduri, NT: 1 rând -> total 3 rânduri, 2 coloane
    assert combined.shape == (3, 2)
    assert list(combined["f1"]) == [1, 2, 5]
    assert list(combined["f2"]) == [3, 4, 6]

def test_prepare_input_and_predict_yields(tmp_json_features, dummy_reference_csv, dummy_models_iter3):
    """
    Test „end-to-end”:
     1. Încărcăm features
     2. Încărcăm date de referință
     3. Pregătim input_df
     4. Apelăm predict_yields cu modelele dummy
    """
    feature_columns, crop_columns, other_features = load_features(json_path=tmp_json_features)
    ct_path_csv, nt_path_csv = dummy_reference_csv
    reference_data = load_reference_data(columns=feature_columns, ct_path=ct_path_csv, nt_path=nt_path_csv)

    # Pregătim form_data pentru cele două other_features („f1” și „f2”)
    form_data = {"f1": 999, "f2": 888}
    selected_crop = "Grâu"
    input_df = prepare_input(
        selected_crop=selected_crop,
        crop_columns=crop_columns,
        other_features=other_features,
        reference_data=reference_data,
        form_data=form_data
    )

    # Verificăm DataFrame-ul
    assert set(input_df.columns) == {"Crop_Grâu", "Crop_Orez", "f1", "f2"}
    assert input_df.loc[0, "Crop_Grâu"] == 1
    assert input_df.loc[0, "Crop_Orez"] == 0
    assert input_df.loc[0, "f1"] == 999
    assert input_df.loc[0, "f2"] == 888

    # Apelăm predict_yields cu căile celor două modele dummy
    ct_model_path, nt_model_path = dummy_models_iter3
    yield_ct, yield_nt = predict_yields(input_df, model_ct_path=ct_model_path, model_nt_path=nt_model_path)

    assert yield_ct == pytest.approx(555.0)
    assert yield_nt == pytest.approx(999.0)
