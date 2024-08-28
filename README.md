# Patient-Gesture-Based-Emergency-Alert-System

### This project is a real-time emergency alert system that leverages computer vision and machine learning to detect patient gestures and provide immediate alerts. The system is designed to recognize hand gestures and facial features to determine whether an emergency situation has occurred. It uses advanced technologies such as TensorFlow for hand sign recognition, OpenCV for image processing, and MediaPipe for hand landmark detection. The system also integrates audio alerts to notify caregivers or medical personnel when an emergency is detected.

## Features

* **Real-Time Gesture Recognition:** Utilizes TensorFlow to identify hand gestures and classify them into predefined categories.
* Face Detection:** Employs OpenCV's Haar cascade for detecting faces to ensure the system operates in the right context.
* **Audio Alerts:** Provides immediate audio notifications for various detected gestures to alert caregivers.
* **Streaming Video Feed:** Streams live video from a webcam and overlays detection results on the video feed.
* **Django Integration:** Implements a Django web application to serve the video feed and manage the system's operations.

## Technologies Used

* **Python:** Main programming language.
* **Django:** Web framework for serving the video feed and managing the application.
* **OpenCV:** Library for image and video processing.
* **TensorFlow:** Machine learning library for gesture classification.
* **MediaPipe:** Framework for hand and face landmark detection.
* **pygame:** Library for audio playback.
* **Python:** Main programming language.
* **Django:** Web framework for serving the video feed and managing the application.
* **OpenCV:** Library for image and video processing.
* **TensorFlow:** Machine learning library for gesture classification.
* **MediaPipe:** Framework for hand and face landmark detection.
* **pygame:** Library for audio playback.

## Installation

### 1. Clone the Repository:

```
git clone https://github.com/daravidchhayrong/Patient-Gesture-Based-Emergency-Alert-System.git
cd Patient-Gesture-Based-Emergency-Alert-System
```

### 2. Install Dependencies:

```
pip install -r requirements.txt
```

### 3. Run the Django Server:

```python
python manage.py runserver
```

## Usage

* **Video Feed:** The `/detection/video_feed/` endpoint streams the live video feed with gesture and face detection overlays.
* **Audio Alerts:** The system plays audio alerts based on detected gestures.

## License

### This project is licensed under the MIT License. See the [LICENSE]() file for details.
