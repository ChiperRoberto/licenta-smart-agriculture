# tests/test_iteratia1.py
import pytest
import joblib
import numpy as np

from app.utils.iteratia1 import load_model, load_label_encoder, predict_crop

# Clase DummyModelNB și DummyLabelEncoder la nivel de modul
class DummyModelNB:
    def predict(self, X):
        # Așteptăm shape (1, 7)
        assert X.shape == (1, 7)
        return [2]  # întotdeauna „porumb”

class DummyLabelEncoder:
    def inverse_transform(self, labels):
        # Așteptăm labels == [2], întoarcem „porumb”
        assert labels == [2]
        return ["porumb"]


@pytest.fixture
def create_dummy_model_and_encoder(tmp_path):
    """
    Salvează cu joblib DummyModelNB și DummyLabelEncoder pentru testarea load_*
    """
    model = DummyModelNB()
    encoder = DummyLabelEncoder()
    model_path = tmp_path / "NB_model_dummy.pkl"
    encoder_path = tmp_path / "crop_label_encoder_dummy.pkl"
    joblib.dump(model, model_path)
    joblib.dump(encoder, encoder_path)
    return str(model_path), str(encoder_path)


def test_load_model_and_label_encoder(create_dummy_model_and_encoder, monkeypatch):
    """
    Verificăm că load_model și load_label_encoder apelează joblib.load corect.
    """
    model_path, encoder_path = create_dummy_model_and_encoder
    calls = []

    def fake_load(path):
        calls.append(path)
        return "dummy_return"

    monkeypatch.setattr(joblib, "load", fake_load)

    m = load_model(path=model_path)
    e = load_label_encoder(path=encoder_path)

    assert calls == [model_path, encoder_path]
    assert m == "dummy_return"
    assert e == "dummy_return"


@pytest.mark.parametrize(
    "N,P,K,temp,humid,ph,rainfall,expected",
    [
        (10, 5, 5, 20.0, 50.0, 6.5, 100.0, "PORUMB"),
        (0, 0, 0, 0.0, 0.0, 1.0, 0.0, "PORUMB"),
    ]
)
def test_predict_crop_happy_path(create_dummy_model_and_encoder, monkeypatch,
                                 N, P, K, temp, humid, ph, rainfall, expected):
    """
    Verificăm predict_crop cu DummyModelNB și DummyLabelEncoder,
    rezultatul trebuie să fie „PORUMB” (uppercase).
    """
    model_path, encoder_path = create_dummy_model_and_encoder
    dummy_model = joblib.load(model_path)
    dummy_encoder = joblib.load(encoder_path)

    # Forțăm predict() și inverse_transform() așa cum vrem
    monkeypatch.setattr(dummy_model, "predict", lambda X: [2])
    monkeypatch.setattr(dummy_encoder, "inverse_transform", lambda labels: ["porumb"])

    result = predict_crop(N, P, K, temp, humid, ph, rainfall, dummy_model, dummy_encoder)
    assert result == expected
