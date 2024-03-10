import cv2
import mediapipe as mp
import playsound
import math

cap = cv2.VideoCapture(1)

mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

# Initial posture data (dictionary)
initial_pose = None
# Reference shoulder points (modify these based on your needs)
left_shoulder_landmark = mpPose.PoseLandmark.LEFT_SHOULDER
right_shoulder_landmark = mpPose.PoseLandmark.RIGHT_SHOULDER

def calculate_triangle_area(point1, point2, point3):
  """Calculates the area of a triangle using Heron's formula."""
  a = math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
  b = math.sqrt((point2[0] - point3[0])**2 + (point2[1] - point3[1])**2)
  c = math.sqrt((point3[0] - point1[0])**2 + (point3[1] - point1[1])**2)
  s = int(a + b + c) / 2
  print(s)
  return math.sqrt(s * (s - a) * (s - b) * (s - c))

def is_significant_variation(variation, threshold):
  """Checks if the variation exceeds a threshold (considering noise)."""
  return variation > threshold * 2  # Adjust threshold multiplier based on noise

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)

    if results.pose_landmarks:
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)

        # Capture initial pose on first detection
        if initial_pose is None:
            initial_pose = {
                "left_triangle_area": 0,  # Initialize with None
                "right_triangle_area": 0,  # Initialize with None
                # ... other initial pose data if needed
            }
            for idx, landmark in enumerate(results.pose_landmarks.landmark):
                landmark_name = mpPose.PoseLandmark(idx).name  # Use index for name
                initial_pose[landmark_name] = (landmark.x, landmark.y)
            print("Initial posture captured!")
        # Handle potential KeyError
        # Calculate triangle areas (left and right)
        left_eye = results.pose_landmarks.landmark[mpPose.PoseLandmark.LEFT_EYE]
        right_eye = results.pose_landmarks.landmark[mpPose.PoseLandmark.RIGHT_EYE]
        nose = results.pose_landmarks.landmark[mpPose.PoseLandmark.NOSE]
        left_shoulder = results.pose_landmarks.landmark[left_shoulder_landmark]
        right_shoulder = results.pose_landmarks.landmark[right_shoulder_landmark]
        left_triangle_area = calculate_triangle_area(
            (left_eye.x, left_eye.y), (nose.x, nose.y), (left_shoulder.x, left_shoulder.y))
        right_triangle_area = calculate_triangle_area(
            (right_eye.x, right_eye.y), (nose.x, nose.y), (right_shoulder.x, right_shoulder.y))

        # Check for significant variations in areas (normalized by image dimension)
        threshold = 0.1  # Adjust threshold value (0-1)
        image_area = img.shape[0] * img.shape[1]  # Total image area
        
        left_variation = abs(left_triangle_area - initial_pose["left_triangle_area"]) / image_area
        right_variation = abs(right_triangle_area - initial_pose["right_triangle_area"]) / image_area

        if is_significant_variation(left_variation, threshold) or is_significant_variation(right_variation, threshold):
            playsound.playsound("beep-02.wav", False)

        # Update initial pose areas for next frame comparison
        initial_pose["left_triangle_area"] = left_triangle_area
        initial_pose["right_triangle_area"] = right_triangle_area

    cv2.imshow("Image", img)
    # Space to capture initial posture, q to quit
    key = cv2.waitKey(1) & 0xFF
    if key == ord(' '):
        if results.pose_landmarks:
            # Reset initial pose data on space key press
            initial_pose = None
            print("Resetting initial posture...")
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()