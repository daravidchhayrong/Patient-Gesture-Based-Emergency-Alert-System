import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import pygame
import time
from django.http import StreamingHttpResponse
import os
import threading

# Initialize pygame mixer
pygame.mixer.init()

# Correct path to Haar cascade for face detection
cascade_path = os.path.join(os.path.dirname(__file__), '..', 'savedmodel', 'haarcascade_frontalface_default.xml')
face_cascade = cv2.CascadeClassifier(cascade_path)

# Load the trained model for hand sign detection
try:
    model_path = os.path.join(os.path.dirname(__file__), '..', 'savedmodel', 'hand_sign_lstm_model.h5')
    model = tf.keras.models.load_model(model_path)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.8)
mp_drawing = mp.solutions.drawing_utils

# Global variables for multi-threading
global_frame = None
frame_lock = threading.Lock()

def capture_frames():
    global global_frame
    cap = cv2.VideoCapture(1)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        with frame_lock:
            global_frame = frame.copy()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()

def play_sound(audio_file):
    try:
        pygame.mixer.music.load(audio_file)
        pygame.mixer.music.play()
    except Exception as e:
        print(f"Error playing sound: {e}")

def stop_sound():
    pygame.mixer.music.stop()

def extract_hand_landmarks(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    if results.multi_hand_landmarks:
        landmarks = results.multi_hand_landmarks[0]
        landmarks_array = np.array([(lm.x, lm.y, lm.z) for lm in landmarks.landmark]).flatten()
        return landmarks_array, results
    return None, None

def gen_frames():
    x, y, w, h = 200, 100, 300, 300
    detection_interval_1 = 30  # seconds
    detection_interval_2 = 40  # seconds

    # Start the thread to capture frames
    frame_thread = threading.Thread(target=capture_frames)
    frame_thread.start()

    last_detection_time = time.time()
    head_detected = False
    current_audio_file = None

    while True:
        with frame_lock:
            if global_frame is None:
                continue
            frame = global_frame.copy()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        detected = any((fx >= x and fy >= y and fx + fw <= x + w and fy + fh <= y + h) for fx, fy, fw, fh in faces)
        elapsed_time = time.time() - last_detection_time

        if elapsed_time > detection_interval_2:
            circle_color = (0, 0, 255)
            audio_file = os.path.join(os.path.dirname(__file__), '..', 'savedmodel', 'alert-33762.mp3')
        elif elapsed_time > detection_interval_1:
            circle_color = (0, 255, 255)
            audio_file = os.path.join(os.path.dirname(__file__), '..', 'savedmodel', 'weeb-alert-182941.mp3')
        else:
            circle_color = (0, 0, 0)
            audio_file = None

        if audio_file != current_audio_file:
            stop_sound()
            if audio_file:
                play_sound(audio_file)
            current_audio_file = audio_file

        cv2.circle(frame, (100, 50), 20, circle_color, -1)
        cv2.putText(frame, "Yes" if detected else "No", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if detected else (0, 0, 255), 2)

        if detected:
            head_detected = True
            stop_sound()

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if head_detected:
            last_label = None
            current_audio_file = None
            last_hand_detection_time = time.time()
            hand_detection_timeout = 10  # seconds

            while True:
                with frame_lock:
                    if global_frame is None:
                        continue
                    image = global_frame.copy()

                landmarks, results = extract_hand_landmarks(image)
                if landmarks is not None:
                    last_hand_detection_time = time.time()
                    landmarks_array = landmarks.reshape(1, 1, -1)
                    prediction = model.predict(landmarks_array)
                    class_id = np.argmax(prediction)

                    class_labels = {0: 'Normal', 1: 'Sign 1', 2: 'Sign 2'}
                    label = class_labels.get(class_id, 'Unknown')

                    if class_id == 0:
                        label = 'Normal'
                        audio_file = None
                    elif class_id == 1:
                        label = 'Sign 1'
                        audio_file = os.path.join(os.path.dirname(__file__), '..', 'savedmodel', 'weeb-alert-182941.mp3')
                    elif class_id == 2:
                        label = 'Sign 2'
                        audio_file = os.path.join(os.path.dirname(__file__), '..', 'savedmodel', 'alert-33762.mp3')
                    else:
                        label = 'Unknown'
                        audio_file = None
                    
                    if audio_file != current_audio_file:
                        stop_sound()
                        if audio_file:
                            play_sound(audio_file)
                        current_audio_file = audio_file

                    mp_drawing.draw_landmarks(image, results.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)
                    cv2.putText(image, f"Detected: {label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(image, f'Confidence: {np.max(prediction):.2f}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

                    if label != last_label or audio_file != current_audio_file:
                        stop_sound()
                        if audio_file:
                            play_sound(audio_file)
                        last_label = label
                        current_audio_file = audio_file
                else:
                    if last_label:
                        stop_sound()
                        last_label = None
                        current_audio_file = None

                if time.time() - last_hand_detection_time > hand_detection_timeout:
                    head_detected = False
                    break

                ret, buffer = cv2.imencode('.jpg', image)
                frame = buffer.tobytes()

                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

                if cv2.waitKey(5) & 0xFF == 27:  # Press 'Esc' to exit
                    break

    cv2.destroyAllWindows()

def video_feed(request):
    return StreamingHttpResponse(gen_frames(), content_type='multipart/x-mixed-replace; boundary=frame')
