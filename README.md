# AI-Based Animal Disease Detection System Version 0.1

## Overview

This project is an AI-powered system that detects animal skin diseases using image classification. It analyzes uploaded images and predicts whether the animal is affected by a specific disease or is healthy.

The system is designed to assist in early detection and provide basic insights for quick action.

---

## Problem Statement

Animals often suffer from skin diseases such as infections, allergies, and parasites. Many pet owners and farmers:

* Cannot identify diseases early
* Lack access to immediate veterinary support
* Ignore symptoms until they become severe

This leads to delayed treatment and increased risk to animal health.

---

## Solution

This system uses Deep Learning to:

* Analyze animal images
* Detect patterns related to skin diseases
* Classify conditions such as infections, allergies, or healthy skin
* Provide prediction confidence and disease information

---

## Features

* Image-based disease detection
* Confidence score for predictions
* Supports multiple disease classes
* Simple and user-friendly web interface (Flask)
* Fast and real-time prediction
* Scalable for future healthcare modules

---

## Technologies Used

* Python
* PyTorch – Deep Learning model
* Torchvision – Image transformations
* Flask – Web application
* Pillow (PIL) – Image processing

---

## Project Structure

```
ANIMALPROJECT/
│
├── app.py                      # Flask backend
├── animal_disease_weights.pt   # Trained model (not included)
├── train_model.py              # Model training script
├── requirements.txt
├── README.md
├── .gitignore
│
├── templates/
│   └── index.html              # Frontend UI
│
├── static/                     # CSS, JS, Images
│
└── data/ (optional future use)
```

---

## How to Run

### 1. Clone the repository

```
git clone https://github.com/pratik-singh-03/Animalproject.git
cd Animalproject
```

### 2. Install dependencies

```
pip install -r requirements.txt
```

### 3. Add model file

Place the trained model file in the root directory:

```
animal_disease_weights.pt
```

### 4. Run the application

```
python app.py
```

### 5. Open in browser

```
http://127.0.0.1:5000/
```

---

## Future Enhancements

* First Aid module for emergency animal care
* Common medicines recommendation system
* Veterinary hospital locator integration
* Mobile-friendly UI
* Real-time AI-based treatment suggestions
* Multi-page dashboard interface

---

## Author

Pratik Singh
