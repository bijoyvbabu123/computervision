import mediapipe as mp
import cv2
import numpy as np

from pprint import pprint
import inspect

# Initialize MediaPipe Pose.
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


cap = cv2.VideoCapture(0)

# printing the attributes of the cap object 
for attr in inspect.getmembers(cap):
    print(attr)

with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:
    
    """
    printing the attributes of the pose object

    for attr in inspect.getmembers(pose):
        print(attr)
    """

    while cap.isOpened():
        success, frame = cap.read()

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        results = pose.process(image)
        print(results)
        print(dir(results))
        for attr in dir(results):
            if not attr.startswith('_'):
                print(attr, getattr(results, attr))

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )


        cv2.imshow('MediaPipe Pose', image)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()