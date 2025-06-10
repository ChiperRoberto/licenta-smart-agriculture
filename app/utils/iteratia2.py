import torch
from torchvision import transforms
from PIL import Image

CLASS_NAMES = [
    "Bacterial leaf blight", "Brown spot", "Healthy corn", "Infected",
    "Leaf smut", "Pepper__bell___Bacterial_spot", "Pepper__bell___healthy",
    "Potato___Early_blight", "Potato___Late_blight", "Potato___healthy",
    "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Late_blight",
    "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites_Two_spotted_spider_mite", "Tomato___Target_Spot",
    "Tomato___YellowLeaf_Curl_Virus", "Tomato___Tomato_mosaic_virus", "Tomato___healthy"
]

def load_cnn_model(path="models/final_model.pth", num_classes=20):
    from app.models.cnn_model import CNNModel
    model = CNNModel(num_classes=num_classes, img_height=64, img_width=64)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model

def predict_disease(image: Image.Image, model):
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    input_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted_class = torch.max(outputs, 1)
        return CLASS_NAMES[predicted_class.item()]
