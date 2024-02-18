import cv2

cap = cv2.VideoCapture(0) # capture frames from a camera

while (1): #If there is no camera then this loop will not be executed
    hasFrame, frame = cap.read() # reads frames from a camera
    # frame = frame[100:300, 100:300] # Crop the frame
    cv2.imshow("Original", frame) # Display an original Frames
    # cv2.waitKey(1) # wait for keyPress for just 1 millisecond and it will continue to refresh and read frame.
    if cv2.waitKey(1) == ord("q"):
        break
cap.release() # Close the capture window
cv2.destroyAllWindows()
