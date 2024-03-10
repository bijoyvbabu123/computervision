import cv2
import mediapipe as mp
import playsound

cap = cv2.VideoCapture(1)

mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

# Initial posture reference point (modify these for your desired posture)
initial_nose_y = None  # You can set a numeric value (y-coordinate) here

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)

    if results.pose_landmarks:
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)

        nose = results.pose_landmarks.landmark[mpPose.PoseLandmark.NOSE]
        h, w, c = img.shape
        x, y = int(nose.x * w), int(nose.y * h)

        cv2.line(img, (0, int(0.7*h)), (w, int(0.7*h)), (0, 255, 0), 2)

        # Set initial posture on first detection
        if initial_nose_y is None:
            initial_nose_y = y

        # Check for deviation from initial posture (allow for a threshold)
        threshold = 75  # You can adjust this threshold value as needed (in pixels)
        if abs(y - initial_nose_y) > threshold:
            playsound.playsound("beep-02.wav", False)

    cv2.imshow("Image", img)
    # Space to capture initial posture, q to quit
    key = cv2.waitKey(1) & 0xFF
    if key == ord(' '):
        if results.pose_landmarks:
            initial_nose_y = y
            print("Initial posture captured!")
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
