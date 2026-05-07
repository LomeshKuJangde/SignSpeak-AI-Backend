import cv2
import mediapipe as mp
import numpy as np
import joblib
from pathlib import Path
from collections import deque, Counter


# -----------------------------
# PATHS
# -----------------------------
PROJECT_DIR = Path(r"D:\sign_language_project")
MODELS_DIR = PROJECT_DIR / "models"

ONE_HAND_MODEL_PATH = MODELS_DIR / "one_hand_rf.pkl"
TWO_HAND_MODEL_PATH = MODELS_DIR / "two_hand_rf.pkl"

# -----------------------------
# LOAD MODELS
# -----------------------------
one_hand_model = joblib.load(ONE_HAND_MODEL_PATH)
two_hand_model = joblib.load(TWO_HAND_MODEL_PATH)

print("Models loaded successfully ✅")

# -----------------------------
# MEDIAPIPE SETUP
# -----------------------------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# -----------------------------
# CLASS LISTS
# -----------------------------
ONE_HAND_CLASSES = ["C", "I", "L", "O", "U", "V"]

TWO_HAND_CLASSES = [
    "A", "B", "D", "E", "F", "G", "H", "J", "K",
    "M", "N", "P", "Q", "R", "S", "T", "W", "X", "Y", "Z"
]

prediction_buffer = deque(maxlen=3)

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


def extract_features_from_frame(results):
    """
    Returns:
        features: list of features
        hand_count: number of detected hands
        status: success / error reason
    """

    if not results.multi_hand_landmarks:
        return None, 0, "no_hands"

    detected_hand_count = len(results.multi_hand_landmarks)

    # -------------------------
    # ONE-HAND CASE
    # -------------------------
    if detected_hand_count == 1:
        hand_landmarks = results.multi_hand_landmarks[0]
        label = results.multi_handedness[0].classification[0].label
        
        raw_features = extract_single_hand_landmarks(hand_landmarks)
        norm_features = normalize_hand_landmarks(raw_features)
        
        # --- FINGERPRINT AUDIT ---
        print("\n" + "="*30)
        print(f"PYTHON FINGERPRINT: {label}")
        print(f"RAW WRIST: {raw_features[0]:.6f}, {raw_features[1]:.6f}, {raw_features[2]:.6f}")
        print(f"RAW THUMB: {raw_features[3]:.6f}, {raw_features[4]:.6f}, {raw_features[5]:.6f}")
        print(f"NORM THUMB: {norm_features[3]:.6f}, {norm_features[4]:.6f}, {norm_features[5]:.6f}")
        print("="*30)
        
        return norm_features, 1, "success"

    # -------------------------
    # TWO-HAND CASE
    # -------------------------
    elif detected_hand_count == 2:
        hand_data = []

        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            label = results.multi_handedness[idx].classification[0].label  # Left / Right
            coords = extract_single_hand_landmarks(hand_landmarks)
            coords = normalize_hand_landmarks(coords)
            hand_data.append((label, coords))

        # Left first, Right second
        hand_data_sorted = sorted(hand_data, key=lambda x: 0 if x[0] == "Left" else 1)

        labels_found = [x[0] for x in hand_data_sorted]

        if labels_found != ["Left", "Right"]:
            return None, 2, f"invalid_handedness_{labels_found}"

        combined_features = hand_data_sorted[0][1] + hand_data_sorted[1][1]
        return combined_features, 2, "success"

    return None, detected_hand_count, "unsupported_hand_count"


# -----------------------------
# REAL-TIME PREDICTION LOOP
# -----------------------------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Cannot access webcam")
    exit()

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
) as hands:

    print("Starting webcam... Press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Flip for mirror view
        frame = cv2.flip(frame, 1)

        # Convert to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process with MediaPipe
        results = hands.process(rgb_frame)

        prediction_text = "No Hands Detected"

        # Draw landmarks
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

        # Extract features
        features, hand_count, status = extract_features_from_frame(results)

        # -----------------------------
        # PREDICTION LOGIC
        # -----------------------------
        if status == "success":
            if hand_count == 1:
                pred = one_hand_model.predict([features])[0]
                prediction_buffer.append(pred)

            elif hand_count == 2:
                pred = two_hand_model.predict([features])[0]
                prediction_buffer.append(pred)

            # Majority vote (light smoothing)
            if len(prediction_buffer) > 0:
                final_pred = Counter(prediction_buffer).most_common(1)[0][0]
                prediction_text = f"Pred: {final_pred}"
            else:
                prediction_text = "..."

        else:
            prediction_text = status

        # -----------------------------
        # DISPLAY UI
        # -----------------------------
        # Background rectangle for top info
        cv2.rectangle(frame, (10, 10), (500, 110), (0, 0, 0), -1)

        # Prediction text
        cv2.putText(
            frame,
            prediction_text,
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

        # Hand count text
        cv2.putText(
            frame,
            f"Hands Detected: {hand_count}",
            (20, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        # Show frame
        cv2.imshow("Sign Language Recognition", frame)

        # Exit key
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s') and status == "success":
            with open(PROJECT_DIR / "python_vector.txt", "w") as f:
                f.write(",".join([f"{x:.6f}" for x in features]))
            print(f"✅ Python Vector Saved! Length: {len(features)}")

cap.release()
cv2.destroyAllWindows()