import cv2
import tf_pose

# Assuming you've successfully installed tf-pose-estimation
estimator = tf_pose.get_estimator()


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Apply PoseNet to detect poses
    poses = estimator.infer(frame)
    
    # Draw poses on the frame
    frame_with_poses = tf_pose.draw_poses(frame, poses)
    
    # Display the frame
    cv2.imshow('PoseNet Live Feed', frame_with_poses)
    
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
