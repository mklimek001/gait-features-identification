import numpy as np
import mediapipe as mp

class MediaPipeEstimator:
    def __init__(self):
        model_path = 'mediapipe_models/pose_landmarker_heavy.task'
        self.video_fps = 25
        self.frame_duration_ms = 1000/self.video_fps
        self.landmarks_num = 33
        
        BaseOptions = mp.tasks.BaseOptions
        PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.VIDEO)

        self.landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(options)

    def predict_for_frame(self, frame_num, frame):
        frame_ts_ms = int(self.frame_duration_ms * frame_num)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        frame_landmarks = self.landmarker.detect_for_video(mp_image, frame_ts_ms)

        frame_h, frame_w, _ = frame.shape
        
        if frame_landmarks.pose_landmarks:
            x_norm = np.array([frame_landmarks.pose_landmarks[0][i].x for i in range(len(frame_landmarks.pose_landmarks[0]))])
            y_norm = np.array([frame_landmarks.pose_landmarks[0][i].y for i in  range(len(frame_landmarks.pose_landmarks[0]))])
            if len(frame_landmarks.pose_landmarks[0]) != self.landmarks_num:
                print(f"[Frame {frame_num}] Not all landmarks found!")
 
            x_frame = frame_w * x_norm
            y_frame = frame_h * y_norm
            
            points = list(zip(np.round(x_frame).astype(int), np.round(y_frame).astype(int)))
            return points
            
        else:
            print(f"[Frame {frame_num}] Landmarks not found!")
            return []
