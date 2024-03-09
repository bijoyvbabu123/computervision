import cv2
import mediapipe as mp
import pyautogui

# Initialize MediaPose and drawing utilities
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

# Define function to close the window on button click
def on_click(x, y, button, pressed):
    if pressed and button == cv2.EVENT_FLAG_LBUTTON:  # Check for left mouse click
        cv2.destroyAllWindows()

# Create a button using PyAutoGUI
button_width, button_height = 100, 50  # Adjust button size as needed
button_x, button_y = 10, 10  # Adjust button position as needed
pyautogui.click(x=button_x, y=button_y, duration=0.01)  # Simulate a click to create the button
pyautogui.rectangle((button_x, button_y, button_width, button_height), fill='gray', border='black', thickness=2)
pyautogui.text('Close', (button_x + button_width // 2 - 15, button_y + button_height // 2 + 5), color='white', font=('Arial', 12))

# Start video capture and MediaPose
cap = cv2.VideoCapture(0)
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Convert BGR image to RGB for MediaPose
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False  # Set image to read-only for MediaPose
        results = pose.process(image)

        # Set image back to BGR for OpenCV display
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Draw keypoints only without connecting lines
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=results.pose_landmarks,
                connections=None,  # Set connections to None to avoid lines
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmark_drawing_spec(),
                connection_drawing_spec=mp_drawing_styles.get_default_pose_connection_drawing_spec()
            )

        # Set a callback function for button clicks
        cv2.setMouseCallback('Live Feed', on_click)

        # Display the image with the button
        cv2.imshow('Live Feed', image)

        # Exit on 'q' key press or window close
        if cv2.waitKey(5) & 0xFF == ord('q') or cv2.getWindowProperty('Live Feed', cv2.WND_PROP_AUTOSIZE) == -1:
            break

cap.release()
