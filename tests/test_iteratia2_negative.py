# tests/test_iteratia2_negative.py
import pytest
import torch
from PIL import Image
from app.utils.iteratia2 import load_cnn_model, predict_disease, CLASS_NAMES

def test_predict_disease_invalid_image_type():
    """
    Dacă image nu e PIL.Image, torchvision.transforms.Resize va ridica TypeError.
    """
    class DummyCNN(torch.nn.Module):
        def __init__(self, num_classes):
            super().__init__()
            self.eval_called = False
        def eval(self): self.eval_called = True
        def forward(self, x):
            return torch.zeros((x.size(0), len(CLASS_NAMES)))

    model = DummyCNN(num_classes=len(CLASS_NAMES))
    with pytest.raises(TypeError):
        predict_disease(image="not_an_image", model=model)

def test_predict_disease_invalid_model_methods():
    """
    Dacă model nu e un cnn (nu e callable), se ridică TypeError ('model' object is not callable).
    """
    from PIL import Image
    img = Image.new("RGB", (64, 64))
    class NoCNN: pass
    with pytest.raises(TypeError):
        predict_disease(img, model=NoCNN())

def test_predict_disease_index_out_of_range(tmp_path, monkeypatch):
    """
    Simulăm un model care returnează un tensor cu dimensiune mai mare decât numarul de clase,
    astfel încât CLASS_NAMES[idx] duce la IndexError, dar instanțiem un CNNModel cu starea goală,
    forțăm torch.load să returneze un state_dict gol, provocând RuntimeError la load_state_dict.
    """
    from app.models.cnn_model import CNNModel
    from app.utils.iteratia2 import load_cnn_model

    class BadCNN(torch.nn.Module):
        def __init__(self):
            super().__init__()
        def eval(self): pass
        def forward(self, x):
            # Creăm un tensor [1, len(CLASS_NAMES)+1], astfel argmax va fi out-of-range
            out = torch.zeros((1, len(CLASS_NAMES) + 1))
            out[0, -1] = 1.0
            return out

    bad_model = BadCNN()
    bad_path = tmp_path / "bad_cnn.pth"
    torch.save(bad_model.state_dict(), bad_path)

    # Monkeypatch torch.load pentru load_cnn_model
    monkeypatch.setattr(torch, "load", lambda p, map_location=None: bad_model.state_dict())

    # În momentul load_state_dict, lipsesc cheile din state_dict => RuntimeError
    with pytest.raises(RuntimeError):
        load_cnn_model(path=str(bad_path), num_classes=len(CLASS_NAMES))
