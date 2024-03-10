import cv2
import mediapipe as mp

# Initialize MediaPipe Pose.
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils




# setting the left limits
left_results = None
# Capture video from the webcam.
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Convert the BGR image to RGB, flip the image around y-axis for correct handedness output
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    
    # To improve performance, optionally mark the image as not writeable to pass by reference.
    image.flags.writeable = False
    left_results = pose.process(image)

    # Draw the pose annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if left_results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image=image, 
            landmark_list=left_results.pose_landmarks, 
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 245, 0))
        )

    cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))
    if cv2.waitKey(5) & 0xFF == 27:  # Press 'ESC' to exit
        break

cap.release()
cv2.destroyAllWindows()




# setting the right limits
right_results = None
# Capture video from the webcam.
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Convert the BGR image to RGB, flip the image around y-axis for correct handedness output
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    
    # To improve performance, optionally mark the image as not writeable to pass by reference.
    image.flags.writeable = False
    right_results = pose.process(image)

    # Draw the pose annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if right_results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image=image, 
            landmark_list=right_results.pose_landmarks,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 245))
        )

    cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))
    if cv2.waitKey(5) & 0xFF == 27:  # Press 'ESC' to exit
        break

cap.release()
cv2.destroyAllWindows()




# setting the right limits
current_results = None
# Capture video from the webcam.
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Convert the BGR image to RGB, flip the image around y-axis for correct handedness output
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    
    # To improve performance, optionally mark the image as not writeable to pass by reference.
    image.flags.writeable = False
    current_results = pose.process(image)

    # Draw the pose annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if current_results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image=image, 
            landmark_list=current_results.pose_landmarks,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(245, 0, 0))
        )
        mp_drawing.draw_landmarks(
            image=image, 
            landmark_list=right_results.pose_landmarks,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 245))
        )
        mp_drawing.draw_landmarks(
            image=image, 
            landmark_list=left_results.pose_landmarks, 
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 245, 0))
        )

    cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))
    if cv2.waitKey(5) & 0xFF == 27:  # Press 'ESC' to exit
        break

cap.release()
cv2.destroyAllWindows()
