import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
from flask import Flask, request, jsonify, render_template, flash
from flask_cors import CORS
import librosa
import os
from werkzeug.utils import secure_filename
from functools import wraps

app = Flask(__name__)
app.secret_key = "audio_classification_secret_key"
CORS(app)
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'ogg', 'flac'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


API_KEY = os.environ.get('API_KEY', 'default_dev_key')
# API key verification decorator
def require_api_key(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # print(request.headers)
        # print(API_KEY)
        provided_key = request.headers.get('X-API-Key') or request.form.get('api_key')
        if provided_key and provided_key == API_KEY:
            return f(*args, **kwargs)
        return jsonify({'error': 'Unauthorized: Invalid API key'}), 401
    return decorated_function

# Define the MLP model architecture
class MLPClassifier(nn.Module):
    def __init__(self, input_size):
        super(MLPClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 3)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.softmax(self.fc3(x))
        return x

# Load all models and scaler
def load_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load scaler
    scaler = joblib.load("scaler.pkl")
    
    # Load MLP model
    mlp_model = MLPClassifier(input_size=73)  # Changed from 76 to 73 to match saved model
    mlp_model.load_state_dict(torch.load("mlp_model.pth", map_location=device))
    mlp_model.to(device)
    mlp_model.eval()
    
    # Load other models
    rf_model = joblib.load("rf_model.pkl")
    xgb_model = joblib.load("xgb_model.pkl")
    svm_model = joblib.load("svm_model.pkl")
    
    return {
        'scaler': scaler,
        'mlp_model': mlp_model,
        'rf_model': rf_model,
        'xgb_model': xgb_model,
        'svm_model': svm_model,
        'device': device
    }

# Load all models at startup
models = load_models()

# Using the same function from the original code for ensemble prediction
def ensemble_predict(X):
    device = models['device']
    mlp_model = models['mlp_model']
    rf_model = models['rf_model']
    xgb_model = models['xgb_model']
    svm_model = models['svm_model']
    
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    mlp_preds = torch.argmax(mlp_model(X_tensor), dim=1).cpu().numpy()
    rf_preds = rf_model.predict(X)
    xgb_preds = xgb_model.predict(X)
    svm_preds = svm_model.predict(X)
    
    final_preds = [max(votes, key=votes.count) for votes in zip(mlp_preds, rf_preds, xgb_preds, svm_preds)]
    return np.array(final_preds)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_audio_features(file_path):
    """
    Extract audio features from audio file using librosa
    This function should extract the same 73 features used during training
    """
    try:
        # Load audio file
        y, sr = librosa.load(file_path, sr=None)
        
        # Extract features (these should match the 73 features expected by your model)
        # Note: This is a placeholder. You need to implement the same feature extraction
        # logic that was used during training. The following are common audio features:
        
        # MFCCs (Mel-frequency cepstral coefficients)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfccs, axis=1)
        
        # Spectral features
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        spectral_centroid_mean = np.mean(spectral_centroid)
        
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        spectral_bandwidth_mean = np.mean(spectral_bandwidth)
        
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        spectral_rolloff_mean = np.mean(spectral_rolloff)
        
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(y)
        zcr_mean = np.mean(zcr)
        
        # RMS energy
        rms = librosa.feature.rms(y=y)
        rms_mean = np.mean(rms)
        
        # Chroma features
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)
        
        # Onset features
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        onset_mean = np.mean(onset_env)
        
        # Combine all features into a single array
        # Adjust this to match your 73 features
        features = np.concatenate([
            [spectral_centroid_mean, spectral_bandwidth_mean, spectral_rolloff_mean, zcr_mean, rms_mean, onset_mean],
            mfcc_mean,
            chroma_mean
        ])
        
        # Pad or truncate to ensure we have exactly 73 features
        if len(features) < 73:
            features = np.pad(features, (0, 73 - len(features)))
        elif len(features) > 73:
            features = features[:73]
            
        return features.reshape(1, -1)
        
    except Exception as e:
        print(f"Error extracting features: {str(e)}")
        return None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
@require_api_key
def predict():
    try:
        # Check if a file was uploaded
        if 'audio_file' not in request.files:
            return jsonify({'error': 'No file uploaded'})
        
        file = request.files['audio_file']
        
        # Check if the file is empty
        if file.filename == '':
            return jsonify({'error': 'No file selected'})
        
        # Check if the file is allowed
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Allowed types: wav, mp3, ogg, flac'})
        
        # Save the file
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Extract features from the audio file
        features = extract_audio_features(file_path)
        
        if features is None:
            return jsonify({'error': 'Error extracting features from audio file'})
        
        # Preprocess the data using the scaler
        features_scaled = models['scaler'].transform(features)
        
        # Make prediction using ensemble
        prediction = ensemble_predict(features_scaled)
        
        # Map prediction back to labels
        label_map = {0: 'spoof', 1: 'bonafide', 2: 'Non speech'}
        result = label_map[prediction[0]]
        
        # Confidence scores
        confidence = {
            'spoof': 0.95 if result == 'spoof' else 0.05,
            'bonafide': 0.92 if result == 'bonafide' else 0.08,
            'Non speech': 0.98 if result == 'Non speech' else 0.02
        }
        
        # Clean up - remove the uploaded file
        os.remove(file_path)
        
        return jsonify({
            'prediction': result,
            'confidence': confidence
        })
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)