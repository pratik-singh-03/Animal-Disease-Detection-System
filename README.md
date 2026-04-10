# Animal Disease Detection System

## Overview

This project is a web-based application designed to detect common animal skin conditions through image input. Users can upload an image, and the system analyzes it to identify possible issues and provide basic information.

The application is built to assist in early awareness and help users understand potential animal health problems.

---

## Problem Statement

Animals often suffer from skin-related diseases such as infections, allergies, and parasitic conditions. In many situations:

* Early symptoms are difficult to identify
* Immediate veterinary help is not always available
* Lack of awareness leads to delayed treatment

This creates a need for a simple and accessible system that can assist users in identifying possible health issues.

---

## Solution

The current system:

* Accepts animal images as input
* Processes the image using a trained model
* Identifies possible conditions
* Displays the result with confidence and basic description

---

## Current Status

This version focuses on core functionality and basic disease detection.

Future updates will enhance the system into a fully AI-powered platform with advanced features such as:

* Improved detection accuracy
* First aid and emergency guidance
* Nearby veterinary hospital locator
* Appointment booking system
* Doctor-recommended temporary treatment suggestions
* Mobile-friendly and scalable architecture

---

## Vision

The long-term goal is to build a complete animal healthcare support system that not only detects diseases but also assists users with treatment, emergency handling, and access to veterinary services.

---

## Project Structure

```
Animalproject/
│
├── app.py                      # Main Flask application
├── train_model.py              # Model training script
├── animal_disease_weights.pt   # Trained model (not included in repository)
├── requirements.txt            # Project dependencies
├── README.md                   # Documentation
├── LICENSE                     # Apache 2.0 License
├── .gitignore                  # Ignored files
│
├── templates/
│   └── index.html              # Frontend interface
│
├── static/                     # CSS, JavaScript, Images
│
└── data/                       # Reserved for future datasets
```

---

## How to Run

### 1. Clone the Repository

```
git clone https://github.com/pratik-singh-03/Animalproject.git
cd Animalproject
```

### 2. Install Dependencies

```
pip install -r requirements.txt
```

### 3. Add the Model File

Place the trained model file in the root directory:

```
animal_disease_weights.pt
```

Note: The model file is not included in the repository due to size limitations.

### 4. Run the Application

```
python app.py
```

### 5. Open in Browser

```
http://127.0.0.1:5000/
```

---

## Notes

* Python 3.8 or above is recommended
* Model file must be present for predictions to work
* The application runs locally without internet dependency
* Future updates will include advanced modules and integrations
