import cv2
import streamlit as st
import numpy as np
import joblib
import mediapipe as mp

# Load model path of you device
model = joblib.load('mobilenet.pkl')
labels = ["quiet", "rich", "sad", "ugly"]

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

st.title("Live Gesture and Face Tracking with Prediction")
stframe = st.empty() 
cap = cv2.VideoCapture(0) 

with mp_holistic.Holistic(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to access the webcam. Please ensure it's connected and try again.")
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = holistic.process(frame_rgb)

        mp_drawing.draw_landmarks(
            frame,
            results.face_landmarks,
            mp_holistic.FACEMESH_CONTOURS,
            mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
            mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1),
        )
        mp_drawing.draw_landmarks(
            frame,
            results.left_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2),
        )
        mp_drawing.draw_landmarks(
            frame,
            results.right_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2),
        )

        input_frame = cv2.resize(frame, (224, 224))
        input_frame = input_frame / 255.0
        input_frame = np.expand_dims(input_frame, axis=0)

        predictions = model.predict(input_frame)
        predicted_class = labels[np.argmax(predictions)]

        cv2.putText(frame, f"Prediction: {predicted_class}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(rgb_frame, channels="RGB")

cap.release()
cv2.destroyAllWindows()