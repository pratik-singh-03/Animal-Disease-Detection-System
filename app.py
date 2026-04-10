from flask import Flask, request, jsonify, render_template
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18
import os

app = Flask(__name__)

# ✅ Load model safely (correct path for Render)
model = resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, 6)

model_path = os.path.join(os.path.dirname(__file__), "animal_disease_weights.pt")

model.load_state_dict(
    torch.load(model_path, map_location=torch.device("cpu"))
)
model.eval()

# Labels
labels = [
    'Dermatitis',
    'Fungal_infections',
    'Healthy',
    'Hypersensitivity',
    'demodicosis',
    'ringworm'
]

# Disease info
disease_info = {
    "Dermatitis": "Skin inflammation causing redness, itching, and irritation.",
    "Fungal_infections": "Caused by fungi, leads to scaly patches and hair loss.",
    "Healthy": "No infection detected.",
    "Hypersensitivity": "Allergic reaction causing itching and redness.",
    "demodicosis": "Mite infection causing hair loss and skin damage.",
    "ringworm": "Fungal infection causing circular patches."
}

# Image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        image_file = request.files['image']
        image = Image.open(image_file).convert('RGB')
        image = transform(image).unsqueeze(0)

        with torch.no_grad():
            outputs = model(image)
            probs = torch.nn.functional.softmax(outputs[0], dim=0)
            confidence, predicted = torch.max(probs, 0)
            result = labels[predicted.item()]

        return jsonify({
            "infection": result,
            "confidence": float(confidence.item()),
            "description": disease_info[result]
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500