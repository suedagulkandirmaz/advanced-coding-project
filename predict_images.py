#predict_images.py 

import torch 
from torchvision import transforms
from PIL import Image
import os
from app.models.cnn_model import DiseaseCNN  


def load_model(model_path, total_classes):
    model = DiseaseCNN(total_classes=total_classes)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def predict_image(image_path, model, class_names):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0)  

    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)

    return class_names[predicted.item()]

if __name__ == "__main__":
    image_path = "test_images/leaf1.jpg"
    model_path = "trained_model/plant_disease_detection.pt"
    total_classes = 13


    class_names = sorted(os.listdir("resource/train"))

    model = load_model(model_path, total_classes)
    prediction = predict_image(image_path, model, class_names)
    print("Prediction:",(prediction))