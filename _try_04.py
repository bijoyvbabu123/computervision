import cv2

# Initialize the camera (0 is the default camera)
cap = cv2.VideoCapture(0)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

frame_number = 0  # Initialize frame counter

# Loop to continuously capture frames
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    # If frame is read correctly, ret is True
    if not ret:
        print("Error: Can't receive frame (stream end?). Exiting ...")
        break
    
    # Increment frame number
    frame_number += 1
    
    # Put frame number on the frame
    cv2.putText(frame, f'Frame: {frame_number}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                1, (0, 255, 0), 2, cv2.LINE_AA)
    
    # Display the resulting frame
    cv2.imshow('Live Feed with Frame Number', frame)
    
    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) == ord('q'):
        break

# When everything is done, release the capture and close any open windows
cap.release()
cv2.destroyAllWindows()
