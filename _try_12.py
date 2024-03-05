# take the live feed from the camera. Using mediapose, find the nodes and show the nodes in the live feed. 
# draw a horizontal line at the center of the screen. If the nose goes below the line, play the beep-02.wav file
# using the playsound module.

import cv2
import mediapipe as mp
import playsound

cap = cv2.VideoCapture(0)

mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils


while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)
    # print(results.pose_landmarks)

    if results.pose_landmarks:
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)

    nose = results.pose_landmarks.landmark[mpPose.PoseLandmark.NOSE]
    h, w, c = img.shape
    x, y = int(nose.x * w), int(nose.y * h)

    cv2.line(img, (0, int(0.7*h)), (w, int(0.7*h)), (0, 255, 0), 2)

    if y > 0.7*h:
        playsound.playsound("beep-02.wav", False)

    cv2.imshow("Image", img)
    # q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
