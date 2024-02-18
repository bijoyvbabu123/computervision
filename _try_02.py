import cv2 
import numpy as np

cap = cv2.VideoCapture(0) # capture frames from a camera 

while (1): #If there is no camera then this loop will not be executed 
    ret, frame = cap.read() # reads frames from a camera
    cv2.imshow("Original", frame) # Display an original Frames
    edges = cv2.Canny(frame, 100, 200,True) # discovers edges in input Frames
    cv2.imshow("Edges", edges) # Display edges in a frame
    #waitKey(0) will pause your screen will not refresh the frame(cap.read()) using your WebCam.
    #waitKey(1) will wait for keyPress for just 1 millisecond and it will continue to refresh and read frame.
    if cv2.waitKey(1) == ord("q"): 
        break
cap.release() # Close the capture window 
cv2.destroyAllWindows()