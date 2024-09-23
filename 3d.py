import cv2
import mediapipe as mp
import tkinter as tk
from tkinter import messagebox
import numpy as np

# Initialize MediaPipe Face Detection, Hands Detection, and Face Mesh
mp_face_detection = mp.solutions.face_detection
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh

class FaceHandDetector:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.2)
        self.hands = mp_hands.Hands(min_detection_confidence=0.5)
        self.face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, max_num_faces=1)
        self.running = False

    def start_detection(self):
        self.running = True
        self.detect_faces_and_hands()

    def stop_detection(self):
        self.running = False
        self.cap.release()
        cv2.destroyAllWindows()

    def detect_faces_and_hands(self):
        while self.running:
            success, image = self.cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            face_results = self.face_detection.process(image_rgb)
            hand_results = self.hands.process(image_rgb)
            mesh_results = self.face_mesh.process(image_rgb)

            # Draw face detections
            face_count = 0
            expression = "Neutral"  # Default expression
            if face_results.detections:
                for detection in face_results.detections:
                    bboxC = detection.location_data.relative_bounding_box
                    h, w, _ = image.shape
                    x, y, width, height = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)
                    cv2.rectangle(image, (x, y), (x + width, y + height), (255, 0, 0), 2)
                    face_count += 1

            # Face Mesh landmarks for expression detection
            if mesh_results.multi_face_landmarks:
                for face_landmarks in mesh_results.multi_face_landmarks:
                    h, w, _ = image.shape
                    landmarks = face_landmarks.landmark

                    # Get relevant landmark coordinates for expression analysis
                    # Example: Calculate mouth aspect ratio (for detecting smile)
                    left_lip = landmarks[61]  # Left side of the mouth
                    right_lip = landmarks[291]  # Right side of the mouth
                    top_lip = landmarks[13]  # Upper lip
                    bottom_lip = landmarks[14]  # Lower lip

                    # Convert landmark positions to screen coordinates
                    left_lip_coord = np.array([int(left_lip.x * w), int(left_lip.y * h)])
                    right_lip_coord = np.array([int(right_lip.x * w), int(right_lip.y * h)])
                    top_lip_coord = np.array([int(top_lip.x * w), int(top_lip.y * h)])
                    bottom_lip_coord = np.array([int(bottom_lip.x * w), int(bottom_lip.y * h)])

                    # Calculate horizontal and vertical distances for mouth aspect ratio
                    horizontal_distance = np.linalg.norm(left_lip_coord - right_lip_coord)
                    vertical_distance = np.linalg.norm(top_lip_coord - bottom_lip_coord)

                    # Smile detection based on mouth aspect ratio (you can tweak thresholds)
                    if vertical_distance / horizontal_distance > 0.35:
                        expression = "Smiling"
                    else:
                        expression = "Neutral"

                    # Draw the expression above the face
                    cv2.putText(image, expression, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            # Draw hand detections
            hand_count = 0
            if hand_results.multi_hand_landmarks:
                hand_count = len(hand_results.multi_hand_landmarks)
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    for landmark in hand_landmarks.landmark:
                        h, w, _ = image.shape
                        cx, cy = int(landmark.x * w), int(landmark.y * h)
                        cv2.circle(image, (cx, cy), 5, (0, 255, 0), -1)

            # Display counts on the screen
            cv2.putText(image, f'Faces: {face_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(image, f'Hands: {hand_count}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # Show the image
            cv2.imshow('Face and Hand Detector with Expression', image)

            if cv2.waitKey(5) & 0xFF == ord('q'):
                self.stop_detection()
                break

class Application(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Face and Hand Detector with Expression Detection")
        self.geometry("300x150")
        self.detector = FaceHandDetector()
        
        self.start_button = tk.Button(self, text="Start Detection", command=self.start_detection)
        self.start_button.pack(pady=20)
        
        self.stop_button = tk.Button(self, text="Stop Detection", command=self.stop_detection)
        self.stop_button.pack(pady=20)

    def start_detection(self):
        self.start_button.config(state="disabled")
        self.stop_button.config(state="normal")
        self.detector.start_detection()

    def stop_detection(self):
        self.detector.stop_detection()
        self.start_button.config(state="normal")
        self.stop_button.config(state="disabled")
        messagebox.showinfo("Info", "Detection Stopped")

if __name__ == "__main__":
    app = Application()
    app.mainloop()
