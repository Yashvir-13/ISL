import cv2
import numpy as np
import joblib
import mediapipe as mp
from translations import hindi_dict, bengali_dict, malayalam_dict, marathi_dict, punjabi_dict, tamil_dict, telugu_dict, kannada_dict, gujarati_dict, urdu_dict

# Load model and label encoder
model_dict = joblib.load('model.pkl')
model = model_dict['model']
with open('label_encoder.pkl', 'rb') as f:
    label_encoder = joblib.load(f)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.7)  # Adjusted confidence

# Camera setup
cap = cv2.VideoCapture(0)

# Choose the dictionary to use
selected_dict = hindi_dict  # Change this to your preferred dictionary

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image from camera.")
        break

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

            data_aux[hand_idx * 42:(hand_idx + 1) * 42] = normalized
            hand_idx += 1

        prediction = model.predict([np.asarray(data_aux)])
        predicted_label = label_encoder.inverse_transform(prediction)[0]

        print(f"Predicted Label: {predicted_label}")  # Debugging line

        try:
            predicted_character = selected_dict[predicted_label]
        except KeyError:
            predicted_character = "Unknown"
        
        print(f"Predicted Character: {predicted_character}")  # Debugging line

        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, 
                hand_landmarks, 
                mp_hands.HAND_CONNECTIONS, 
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
        
        x1, y1 = int(min_x * W) - 10, int(min_y * H) - 10
        cv2.putText(frame, predicted_character, (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    else:
        print("No hand landmarks detected in this frame.")

    cv2.imshow('Real-Time Hand Gesture Recognition', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
