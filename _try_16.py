import cv2
import mediapipe as mp
import playsound
import math

cap = cv2.VideoCapture(1)

mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

# Initial posture data (dictionary to store all landmark coordinates)
initial_pose = None

def calculate_normalized_variation(initial_value, current_value, image_dimension):
  """Calculates normalized variation between initial and current values."""
  print(image_dimension)
  ch=int(current_value - initial_value)
  print(ch/image_dimension)
  return abs(ch/image_dimension)

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)

    if results.pose_landmarks:
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)

        # Capture initial pose on first detection
        if initial_pose is None:
            initial_pose = {}
            for idx, landmark in enumerate(results.pose_landmarks.landmark):
                landmark_name = mpPose.PoseLandmark(idx).name  # Use index for name
                initial_pose[landmark_name] = (landmark.x * img.shape[1], landmark.y * img.shape[0])
            print("Initial posture captured!")

        # Check for variations from initial pose (normalized)
        threshold = 0.1  # You can adjust this threshold value (0-1)
        for landmark_name, (x, y) in initial_pose.items():
            current_landmark = results.pose_landmarks.landmark[mpPose.PoseLandmark[landmark_name]]
            current_x, current_y = int(current_landmark.x * img.shape[1]), int(current_landmark.y * img.shape[0])
            variation_x = calculate_normalized_variation(x, current_x, img.shape[1])
            variation_y = calculate_normalized_variation(y, current_y, img.shape[0])
            if variation_x > threshold and variation_y > threshold:
                playsound.playsound("beep-02.wav", False)
                break  # Only beep once per frame

    cv2.imshow("Image", img)
    # Space to capture initial posture, q to quit
    key = cv2.waitKey(1) & 0xFF
    if key == ord(' '):
        if results.pose_landmarks:
            initial_pose = {}
            for idx, landmark in enumerate(results.pose_landmarks.landmark):
                landmark_name = mpPose.PoseLandmark(idx).name  # Use index for name
                initial_pose[landmark_name] = (landmark.x, landmark.y)
            print("Initial posture captured!")
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
