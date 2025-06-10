# tests/test_iteratia2.py
import pytest
import torch
from app.utils.iteratia2 import load_cnn_model, predict_disease, CLASS_NAMES

# Dynamo CNNModel este definit în app/models/cnn_model.py
from app.models.cnn_model import CNNModel

@pytest.fixture(scope="module")
def dummy_cnn_state(tmp_path_factory):
    """
    Creăm un model CNNModel (același tip folosit în aplicație),
    salvăm state_dict într-un fișier .pth și returnăm (path, num_classes).
    """
    num_classes = len(CLASS_NAMES)
    model = CNNModel(num_classes=num_classes, img_height=64, img_width=64)
    # Salvăm structura (state_dict) fără antrenare efectivă
    path = tmp_path_factory.mktemp("cnn_models") / "dummy_cnn.pth"
    torch.save(model.state_dict(), path)
    return str(path), num_classes

@pytest.fixture
def dummy_image():
    """
    Creează o imagine albă 64×64 ca PIL.Image și o returnează.
    """
    from PIL import Image
    img = Image.new("RGB", (64, 64), color="white")
    return img

def test_load_cnn_model_and_predict(dummy_cnn_state, dummy_image):
    """
    Verificăm că încărcarea modelului și predicția funcționează fără erori.
    predict_disease returnează întotdeauna un element din CLASS_NAMES.
    """
    path, num_classes = dummy_cnn_state
    model = load_cnn_model(path=path, num_classes=num_classes)
    # Modelul trebuie să fie un obiect torch.nn.Module
    assert isinstance(model, torch.nn.Module)
    model.eval()

    # predict_disease returnează un string valid
    result = predict_disease(dummy_image, model)
    assert isinstance(result, str)
    assert result in CLASS_NAMES

def test_load_cnn_model_file_not_found():
    """
    Dacă path-ul nu există sau nu e .pth, apare FileNotFoundError.
    """
    with pytest.raises(FileNotFoundError):
        load_cnn_model(path="invalid_model.txt", num_classes=len(CLASS_NAMES))
