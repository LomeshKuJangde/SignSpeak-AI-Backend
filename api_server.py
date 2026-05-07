from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import joblib
from pathlib import Path

# -----------------------------
# FLASK APP SETUP
# -----------------------------
app = Flask(__name__)
CORS(app)

print("Flask Production API initializing...")

# -----------------------------
# PATHS & MODELS
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

ONE_HAND_MODEL_PATH = MODELS_DIR / "one_hand_rf.pkl"
TWO_HAND_MODEL_PATH = MODELS_DIR / "two_hand_rf.pkl"

# Load models once on startup
one_hand_model = joblib.load(ONE_HAND_MODEL_PATH)
two_hand_model = joblib.load(TWO_HAND_MODEL_PATH)

print("✅ AI Models loaded successfully")

# -----------------------------
# PREPROCESSING LOGIC
# -----------------------------
def normalize_hand_landmarks(landmarks):
    """
    Standardizes landmarks for the Random Forest model.
    Expects a flat list of [x, y, z, x, y, z...]
    """
    arr = np.array(landmarks).reshape(21, 3)

    # Move wrist to (0,0,0) origin
    wrist = arr[0]
    arr = arr - wrist

    # Scale normalization to unit range
    max_val = np.max(np.abs(arr))
    if max_val > 0:
        arr = arr / max_val

    return arr.flatten().tolist()

# -----------------------------
# PRIMARY API ENDPOINT
# -----------------------------
@app.route("/predict_landmarks", methods=["POST"])
def predict_landmarks():
    """
    PRODUCTION ENDPOINT: Receives landmarks from Android CameraX pipeline.
    """
    try:
        data = request.get_json()
        if not data or "hands" not in data:
            return jsonify({"status": "error", "message": "No hand data provided"}), 400

        hands = data["hands"]
        hand_count = len(hands)

        # Sort hands: Left first, Right second (consistent with training)
        hands_sorted = sorted(hands, key=lambda x: 0 if x["label"] == "Left" else 1)

        if hand_count == 1:
            raw_landmarks = hands_sorted[0]["landmarks"]
            features = normalize_hand_landmarks(raw_landmarks)
            
            prediction = one_hand_model.predict([features])[0]
            probs = one_hand_model.predict_proba([features])[0]
            confidence = float(np.max(probs))
            mode = "One-Hand"
            
        elif hand_count == 2:
            h1_features = normalize_hand_landmarks(hands_sorted[0]["landmarks"])
            h2_features = normalize_hand_landmarks(hands_sorted[1]["landmarks"])
            features = h1_features + h2_features
            
            prediction = two_hand_model.predict([features])[0]
            probs = two_hand_model.predict_proba([features])[0]
            confidence = float(np.max(probs))
            mode = "Two-Hand"
        else:
            return jsonify({"status": "error", "message": f"Unsupported hand count: {hand_count}"}), 400

        return jsonify({
            "status": "success",
            "prediction": str(prediction),
            "confidence": confidence,
            "hand_count": hand_count,
            "mode": mode
        })

    except Exception as e:
        print(f"Prediction Error: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "online",
        "message": "SignSpeak AI Production API (JSON-Only Mode) ✅",
        "api_version": "2.0.0"
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)