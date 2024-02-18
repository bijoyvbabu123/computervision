import cv2
img = cv2.imread("cvlogo.png",1)
cv2.imshow("image",img)
cv2.waitKey() 
cv2.destroyAllWindows()