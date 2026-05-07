from flask import Flask, request, jsonify
from flask_cors import CORS

import cv2
import mediapipe as mp
import numpy as np
import joblib
from pathlib import Path
import tempfile
import os

# -----------------------------
# FLASK APP SETUP
# -----------------------------
app = Flask(__name__)
CORS(app)

print("Flask app initialized")

# -----------------------------
# PATHS
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

ONE_HAND_MODEL_PATH = MODELS_DIR / "one_hand_rf.pkl"
TWO_HAND_MODEL_PATH = MODELS_DIR / "two_hand_rf.pkl"

# -----------------------------
# LOAD MODELS
# -----------------------------
one_hand_model = joblib.load(ONE_HAND_MODEL_PATH)
two_hand_model = joblib.load(TWO_HAND_MODEL_PATH)

print("Models loaded successfully")

# -----------------------------
# MEDIAPIPE SETUP
# -----------------------------
mp_hands = mp.solutions.hands

# -----------------------------
# FEATURE EXTRACTION HELPERS
# -----------------------------
def extract_single_hand_landmarks(hand_landmarks):
    coords = []
    for lm in hand_landmarks.landmark:
        coords.extend([lm.x, lm.y, lm.z])
    return coords


def normalize_hand_landmarks(landmarks):
    arr = np.array(landmarks).reshape(21, 3)

    # Wrist as origin
    wrist = arr[0]
    arr = arr - wrist

    # Scale normalization
    max_val = np.max(np.abs(arr))
    if max_val > 0:
        arr = arr / max_val

    return arr.flatten().tolist()


def extract_features_from_results(results):
    """
    Returns:
        features: list or None
        hand_count: int
        status: success / error
        mode: One-Hand / Two-Hand / None
    """
    if not results.multi_hand_landmarks:
        return None, 0, "no_hands", None

    detected_hand_count = len(results.multi_hand_landmarks)

    # -------------------------
    # ONE-HAND CASE
    # -------------------------
    if detected_hand_count == 1:
        hand_landmarks = results.multi_hand_landmarks[0]
        features = extract_single_hand_landmarks(hand_landmarks)
        features = normalize_hand_landmarks(features)
        return features, 1, "success", "One-Hand"

    # -------------------------
    # TWO-HAND CASE
    # -------------------------
    elif detected_hand_count == 2:
        hand_data = []

        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            label = results.multi_handedness[idx].classification[0].label
            coords = extract_single_hand_landmarks(hand_landmarks)
            coords = normalize_hand_landmarks(coords)
            hand_data.append((label, coords))

        hand_data_sorted = sorted(hand_data, key=lambda x: 0 if x[0] == "Left" else 1)
        labels_found = [x[0] for x in hand_data_sorted]

        if labels_found != ["Left", "Right"]:
            return None, 2, f"invalid_handedness_{labels_found}", None

        combined_features = hand_data_sorted[0][1] + hand_data_sorted[1][1]
        return combined_features, 2, "success", "Two-Hand"

    return None, detected_hand_count, "unsupported_hand_count", None

# -----------------------------
# PREDICT API
# -----------------------------
@app.route("/predict_landmarks", methods=["POST"])
def predict_landmarks():
    """
    NEW: High-speed endpoint that receives landmarks directly from the mobile app.
    Avoids image processing overhead on the server.
    """
    try:
        data = request.get_json()
        if not data or "hands" not in data:
            return jsonify({"status": "error", "message": "No hand data provided"}), 400

        hands = data["hands"]
        hand_count = len(hands)

        # Python-Identical Sorting: Left hand always first, Right hand always second
        hands_sorted = sorted(hands, key=lambda x: 0 if x["label"] == "Left" else 1)

        if hand_count == 1:
            # Extract and normalize the single hand
            raw_landmarks = hands_sorted[0]["landmarks"]
            features = normalize_hand_landmarks(raw_landmarks)
            
            prediction = one_hand_model.predict([features])[0]
            
            # --- VECTOR DUMP REMOVED FOR DEPLOYMENT ---
            # (No local file writes in cloud environments)
                
            probs = one_hand_model.predict_proba([features])[0]
            confidence = float(np.max(probs))
            mode = "One-Hand"
            
        elif hand_count == 2:
            # Normalize each hand individually before concatenation
            h1_features = normalize_hand_landmarks(hands_sorted[0]["landmarks"])
            h2_features = normalize_hand_landmarks(hands_sorted[1]["landmarks"])
            
            # Concatenate to form the 126-feature vector
            features = h1_features + h2_features
            
            prediction = two_hand_model.predict([features])[0]
            probs = two_hand_model.predict_proba([features])[0]
            confidence = float(np.max(probs))
            mode = "Two-Hand"
        else:
            return jsonify({"status": "error", "message": f"Unsupported hand count: {hand_count}"}), 400

        print(f"ML_SYNC: Received {hand_count} hand(s). Mode: {mode} | Prediction: {prediction} | Confidence: {confidence:.2f}")
        return jsonify({
            "status": "success",
            "prediction": str(prediction),
            "confidence": confidence,
            "hand_count": hand_count,
            "mode": mode
        })

    except Exception as e:
        print(f"Prediction Error: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


@app.route("/predict", methods=["POST"])
def predict():
    """
    STABLE: Endpoint for image uploads (Capture/Gallery).
    """
    try:
        if "image" not in request.files:
            return jsonify({"status": "error", "message": "No image provided"}), 400

        file = request.files["image"]
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            file.save(temp_file.name)
            temp_path = temp_file.name

        image = cv2.imread(temp_path)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        with mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.6) as hands:
            results = hands.process(rgb_image)

        features, hand_count, status, mode = extract_features_from_results(results)
        os.remove(temp_path)

        if status != "success":
            return jsonify({"status": status, "prediction": None, "hand_count": hand_count})

        if hand_count == 1:
            pred = one_hand_model.predict([features])[0]
            probs = one_hand_model.predict_proba([features])[0]
        else:
            pred = two_hand_model.predict([features])[0]
            probs = two_hand_model.predict_proba([features])[0]

        confidence = float(np.max(probs))

        return jsonify({
            "status": "success",
            "prediction": str(pred),
            "confidence": confidence,
            "hand_count": hand_count,
            "mode": mode
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Sign Language Recognition API is running ✅"
    })

# -----------------------------
# RUN SERVER
# -----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
    
    