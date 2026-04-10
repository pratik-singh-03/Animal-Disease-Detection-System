from flask import Flask, request, jsonify, render_template
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights

app = Flask(__name__)

# Load model
model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
model.fc = torch.nn.Linear(model.fc.in_features, 6)

model.load_state_dict(
    torch.load("animal_disease_weights.pt", map_location=torch.device("cpu"))
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

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image"}), 400

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

if __name__ == '__main__':
    app.run(debug=True)

if __name__ == '__main__':
    app.run()