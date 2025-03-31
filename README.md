# Real vs Recorded/Artificial Sound Classification Engine

A machine learning system that classifies audio as bonafide (real human speech), spoof (artificially generated or manipulated), or non-speech.

## Features

- Audio classification using ensemble machine learning approach
- Support for WAV, MP3, OGG, FLAC audio formats
- Modern, responsive UI with drag-and-drop functionality
- Real-time analysis with confidence scores

## Tech Stack

- **Backend**: Flask, PyTorch, NumPy, Pandas, Librosa
- **Frontend**: HTML, CSS, JavaScript, Bootstrap 5
- **Models**: Ensemble of MLP, Random Forest, XGBoost, and SVM

## Installation

1. Clone the repository
```bash
git clone https://github.com/yourusername/audio-classification-engine.git
cd audio-classification-engine
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Run the application
```bash
python app.py
```

## API Routes

### `GET /`
- Renders the main application interface

### `POST /predict`
- Accepts audio file uploads for classification 
- Returns prediction results as JSON


## How It Works

1. **Audio Processing**: When an audio file is uploaded, the system extracts 73 audio features using Librosa, including MFCCs, spectral features, chroma, and onset features.

2. **Classification**: The extracted features are processed by an ensemble of four machine learning models:
   - Multi-Layer Perceptron (PyTorch)
   - Random Forest
   - XGBoost
   - Support Vector Machine

3. **Ensemble Decision**: The final classification is determined by majority voting among all models, providing robust prediction against individual model biases.

4. **Results**: The system returns the classification result (bonafide, spoof, or non-speech) along with confidence scores.
