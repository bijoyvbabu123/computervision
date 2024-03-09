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

# Define function to create a button using PyAutoGUI
def create_button(x, y, width, height, text, color='gray', border='black', thickness=2):
    """
    Creates a button on the screen using PyAutoGUI.

    Args:
        x (int): X-coordinate of the top-left corner.
        y (int): Y-coordinate of the top-left corner.
        width (int): Width of the button.
        height (int): Height of the button.
        text (str): Text displayed on the button.
        color (str, optional): Button background color. Defaults to 'gray'.
        border (str, optional): Border color. Defaults to 'black'.
        thickness (int, optional): Border thickness. Defaults to 2.
    """

    pyautogui.click(x=x, y=y, duration=0.01)  # Simulate a click
    pyautogui.fillRectangle((x, y, width, height), fillColor=color)  # Fill the region with color
    pyautogui.drawRectangle((x, y, width, height), borderColor=border, thickness=thickness)  # Draw the border
    pyautogui.text(text, (x + width // 2 - len(text) * 5, y + height // 2 + 5), color='white', font=('Arial', 12))  # Center the text

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
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
                # connection_drawing_spec=mp_drawing_styles.get_default_conn()
            )

        # Create the button using the new function
        create_button(button_x, button_y, button_width, button_height, 'Close')

        # Set a callback function for button clicks
        cv2.setMouseCallback('Live Feed', on_click)

        # Display the image with the button
        cv2.imshow('Live Feed', image)

        # Exit on 'q' key press or window close
        if cv2.waitKey(5) & 0xFF == ord('q') or cv2.getWindowProperty('Live Feed', cv2.WND_PROP_AUTOSIZE) == -1:
            break

cap.release()
