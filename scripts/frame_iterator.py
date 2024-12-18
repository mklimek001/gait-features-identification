import cv2
from math import inf

def video_frame_iterator(video_path, frames_n=inf, frames_per_sec=25):
    frame_duration_ms = 1000/frames_per_sec
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Error: Could not open video.")
        
    iteration = 0
    while True and iteration < frames_n:
        ret, frame = cap.read()
        if not ret:
            break
            
        timestamp = int(frame_duration_ms*iteration)
        iteration += 1
        yield timestamp, frame
        
    cap.release()