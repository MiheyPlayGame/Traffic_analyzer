import cv2
import sys
sys.path.append('Project-task-CV-2-37')
from Road_detector import road_marking_detection

cap = cv2.VideoCapture("Traffic.mp4")

ret, frame = cap.read()

cap.release()

result = road_marking_detection(frame)

# Save the result
cv2.imwrite("video_result.jpg", result)
print("Result saved as 'video_result.jpg'")