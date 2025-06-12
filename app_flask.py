# app_flask.py #%74
from flask import Flask, request, jsonify, render_template
import torch
from app.models.cnn_model import DiseaseCNN 
import io
from PIL import Image
from torchvision import transforms

app = Flask(__name__)

class_names = [
    "Pepper_bell_Bacterial_spot",
    "Pepper_bell_healthy",
    "Potato_Early_blight",
    "Potato_healthy",
    "Potato_Late_blight",
    "Tomato_Target_Spot",
    "Tomato_Tomato_mosaic_virus",
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_healthy",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot"
]


# Upload model
model = DiseaseCNN(total_classes=13, rate=0.4)
model.load_state_dict(torch.load("trained_model/plant_disease_detection.pt", map_location=torch.device("cpu")))
model.eval()

# first, we define images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


# main page route
@app.route('/')
def home():
    return render_template('index.html')
@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    image = Image.open(io.BytesIO(file.read())).convert('RGB')
    input_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        output = model(input_tensor)  # Use input_tensor here
        predicted_class = torch.argmax(output, dim=1).item()
        predicted_label = class_names[predicted_class]
    
    return jsonify({
        "prediction_index": predicted_class,
        "prediction_label": predicted_label
    })

if __name__ == '__main__':
    app.run(debug=True)