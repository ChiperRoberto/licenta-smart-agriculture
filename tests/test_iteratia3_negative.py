# tests/test_iteratia3_negative.py
import pytest
import pandas as pd
import json
import joblib
from app.utils.iteratia3 import load_features, load_reference_data, prepare_input, predict_yields

# Definim BadModel la nivel de modul pentru a putea fi pickled
class BadModel:
    def predict(self, X):
        return [0]

def test_load_features_invalid_json(tmp_path):
    """
    Dacă fișierul nu e JSON valid -> json.JSONDecodeError.
    Dacă JSON-ul nu conține cheile „Yield of CT” sau „Yield of NT” -> KeyError.
    """
    # 1) Fișier text care nu e JSON
    bad_file = tmp_path / "bad.txt"
    bad_file.write_text("asta nu e JSON")
    with pytest.raises(json.JSONDecodeError):
        load_features(json_path=str(bad_file))

    # 2) JSON valid, dar fără cheile necesare -> KeyError
    invalid_json = tmp_path / "invalid.json"
    invalid_json.write_text("{}")
    with pytest.raises(KeyError):
        load_features(json_path=str(invalid_json))


def test_load_reference_data_invalid_columns(tmp_path):
    """
    - Dacă fișierele nu există => FileNotFoundError.
    - Dacă fișierele există dar nu conțin coloanele cerute => DataFrame gol.
    """
    # 1) Fișiere inexistențe => FileNotFoundError
    with pytest.raises(FileNotFoundError):
        load_reference_data(columns=["f1", "f2"], ct_path="ct_nonexistent.csv", nt_path="nt_nonexistent.csv")

    # 2) Creăm CSV-uri care nu conțin coloanele cerute
    df_ct = pd.DataFrame({"X": [1]})
    df_nt = pd.DataFrame({"Y": [2]})
    ct_file = tmp_path / "ct.csv"
    nt_file = tmp_path / "nt.csv"
    df_ct.to_csv(ct_file, index=False)
    df_nt.to_csv(nt_file, index=False)

    # Pandas produce DataFrame gol, iar funcția returnează tot un DataFrame gol
    combined = load_reference_data(columns=["f1", "f2"], ct_path=str(ct_file), nt_path=str(nt_file))
    assert isinstance(combined, pd.DataFrame)
    assert combined.empty


def test_prepare_input_missing_features():
    """
    - Dacă form_data lipsește vreun feature din other_features -> KeyError.
    - Dacă selected_crop nu se găsește în crop_columns, nu apare eroare Python.
    """
    crop_columns = ["Crop_Grâu"]
    other_features = ["f1", "f2"]
    reference_data = pd.DataFrame({"f1": [1], "f2": [2]})

    # Lipsă „f2” în form_data -> KeyError
    form_data = {"f1": 10}
    with pytest.raises(KeyError):
        prepare_input("Grâu", crop_columns, other_features, reference_data, form_data)

    # selected_crop vid -> DataFrame returnat, iar „Crop_Grâu” va fi 0
    form_data = {"f1": 10, "f2": 20}
    df = prepare_input("", crop_columns, other_features, reference_data, form_data)
    assert df.loc[0, "Crop_Grâu"] == 0


def test_predict_yields_invalid_inputs(tmp_path):
    """
    - Dacă input_df e gol și modelele nu există -> FileNotFoundError.
    - Dacă calea modelului nu există sau nu e .pkl -> FileNotFoundError.
    - Dacă model salvat nu are feature_names_in_ -> AttributeError.
    """
    # 1) input_df gol + modele inexistențe -> FileNotFoundError
    empty_df = pd.DataFrame()
    with pytest.raises(FileNotFoundError):
        predict_yields(empty_df, model_ct_path="models/ct.pkl", model_nt_path="models/nt.pkl")

    # 2) model_ct_path cu extensie greșită -> FileNotFoundError
    df = pd.DataFrame({"f1": [1], "f2": [2]})
    with pytest.raises(FileNotFoundError):
        predict_yields(df, model_ct_path="models/ct_model.txt", model_nt_path="models/nt_model.pkl")

    # 3) Salvăm un BadModel (fără feature_names_in_) pe disc -> joblib.load funcționează,
    #    dar accesul la feature_names_in_ ridică AttributeError
    bad_path = tmp_path / "bad.pkl"
    joblib.dump(BadModel(), bad_path)

    with pytest.raises(AttributeError):
        predict_yields(df, model_ct_path=str(bad_path), model_nt_path=str(bad_path))
