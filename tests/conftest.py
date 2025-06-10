# tests/conftest.py
import pytest
import pandas as pd
import numpy as np
import json
import joblib
import torch
from PIL import Image

# ----- ITERATIA 3: modele dummy la nivel de modul (pentru joblib.dump) -----

class DummyModelCT:
    def __init__(self):
        self.feature_names_in_ = np.array(["f1", "f2", "Crop_Grâu"])
    def predict(self, X):
        return np.array([555.0])

class DummyModelNT:
    def __init__(self):
        self.feature_names_in_ = np.array(["f1", "f2", "Crop_Orez"])
    def predict(self, X):
        return np.array([999.0])


@pytest.fixture
def tmp_json_features(tmp_path_factory):
    """
    Fixture pentru un JSON temporar (iterația 3) cu coloanele de test.
    """
    data = {
        "Yield of CT": ["f1", "Crop_Grâu"],
        "Yield of NT": ["f2", "Crop_Orez"]
    }
    d = tmp_path_factory.mktemp("data_json")
    file = d / "features.json"
    with open(file, "w", encoding="utf-8") as f:
        json.dump(data, f)
    return str(file)


@pytest.fixture
def dummy_reference_csv(tmp_path):
    """
    Fixture care creează două fișiere CSV temporare pentru CT și NT,
    cu coloanele f1, f2. Returnează (ct_path, nt_path).
    """
    df_ct = pd.DataFrame({"f1": [1, 2], "f2": [3, 4], "extra_ct": [9, 9]})
    df_nt = pd.DataFrame({"f1": [5], "f2": [6], "extra_nt": [8]})
    ct_file = tmp_path / "ct_test.csv"
    nt_file = tmp_path / "nt_test.csv"
    df_ct.to_csv(ct_file, index=False)
    df_nt.to_csv(nt_file, index=False)
    return str(ct_file), str(nt_file)


@pytest.fixture
def dummy_models_iter3(tmp_path):
    """
    Fixture care creează două modele DummyModelCT și DummyModelNT și le salvează cu joblib.
    Returnează (path_ct_model, path_nt_model).
    """
    ct = DummyModelCT()
    nt = DummyModelNT()
    ct_path = tmp_path / "dummy_ct.pkl"
    nt_path = tmp_path / "dummy_nt.pkl"
    joblib.dump(ct, ct_path)
    joblib.dump(nt, nt_path)
    return str(ct_path), str(nt_path)


@pytest.fixture
def dummy_image(tmp_path):
    """
    Fixture care creează o imagine PNG albă 64×64 și returnează un obiect PIL.Image.
    """
    img = Image.new("RGB", (64, 64), color="white")
    img_path = tmp_path / "leaf.png"
    img.save(img_path)
    return img  # întoarcem instanța PIL.Image, nu calea
