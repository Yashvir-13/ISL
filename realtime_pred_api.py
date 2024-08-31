import cv2
import numpy as np
import joblib
import mediapipe as mp
from flask import Flask, render_template, request, jsonify

# Initialize Flask app
app = Flask(__name__)

# Load model and label encoder
model_dict = joblib.load('model.pkl')
model = model_dict['model']
with open('label_encoder.pkl', 'rb') as f:
    label_encoder = joblib.load(f)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.7)  # Adjusted confidence

def process_frame(frame):
    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    
    if results.multi_hand_landmarks:
        data_aux = [0] * 84  # Initialize with zeros for two hands
        hand_idx = 0  # Index to track the hand being processed

        for hand_landmarks in results.multi_hand_landmarks:
            if hand_idx >= 2:  # Skip if more than two hands are detected
                break

            x_ = []
            y_ = []

            for i, landmark in enumerate(hand_landmarks.landmark):
                x = landmark.x
                y = landmark.y

                x_.append(x)
                y_.append(y)

            min_x, max_x = min(x_), max(x_)
            min_y, max_y = min(y_), max(y_)

            normalized = []
            for x, y in zip(x_, y_):
                normalized.append((x - min_x) / (max_x - min_x))
                normalized.append((y - min_y) / (max_y - min_y))

            # Fill in the features for the current hand (42 features)
            data_aux[hand_idx * 42:(hand_idx + 1) * 42] = normalized
            hand_idx += 1

        prediction = model.predict([np.asarray(data_aux)])
        predicted_character = label_encoder.inverse_transform(prediction)

        # Draw bounding boxes and predictions on the frame
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, 
                hand_landmarks, 
                mp_hands.HAND_CONNECTIONS, 
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

        return predicted_character[0], frame
    else:
        return None, frame
@app.route('/')
def index():
    return render_template('index.html')
@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        file = request.files['image'].read()
        np_img = np.frombuffer(file, np.uint8)
        frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        prediction, processed_frame = process_frame(frame)

        if prediction is None:
            return jsonify({'error': 'No hand landmarks detected or feature length mismatch'}), 400

        # Encode the processed frame back to JPEG format
        _, buffer = cv2.imencode('.jpg', processed_frame)
        frame_data = buffer.tobytes()

        return jsonify({
            'prediction': prediction,
            'image': frame_data.hex()
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
