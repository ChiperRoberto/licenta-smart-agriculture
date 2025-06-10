# tests/test_iteratia1_negative.py
import pytest
from app.utils.iteratia1 import predict_crop

class DummyModel:
    def predict(self, X):
        return [0]

class DummyEncoder:
    def inverse_transform(self, labels):
        return ["grau"]

def test_predict_crop_model_missing_predict():
    """
    Dacă modelul nu are metoda predict, apare AttributeError.
    """
    class NoPredict: pass
    with pytest.raises(AttributeError):
        predict_crop(10, 10, 10, 25, 70, 6.5, 100, model=NoPredict(), label_encoder=DummyEncoder())

def test_predict_crop_encoder_missing_inverse_transform():
    """
    Dacă encoder-ul nu are metoda inverse_transform, apare AttributeError.
    """
    class NoInverse:
        def predict(self, X): return [0]
    with pytest.raises(AttributeError):
        predict_crop(10, 10, 10, 25, 70, 6.5, 100, model=DummyModel(), label_encoder=NoInverse())
