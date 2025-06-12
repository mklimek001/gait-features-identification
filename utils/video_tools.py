from scripts.parsers import parse_sequences
import numpy as np
import mediapipe as mp
import cv2

def get_video_files(sequence_key):
    file_path = './gait3d/ListOfSequences.txt'
    sequence_info = parse_sequences(file_path)[sequence_key]
    avi_file_names =[
        f"c{camera_number}_{(4 - len(str(sequence_info['start_frame']))) * '0' + str(sequence_info['start_frame'])}" 
        for camera_number in range(1, 5)
        ]
    
    avi_seq_paths = [
        f"./gait3d/Sequences/{sequence_key}/Images/{avi_file_name}.avi"
        for avi_file_name in avi_file_names
        ]
    
    return avi_seq_paths


def get_camera_calibration_files(sequence_key):
    calibraton_file_path_base = "./gait3d/Sequences/{sequence_key}/Calibration/c{camera_number}.xml"
    camera_file_paths = [calibraton_file_path_base.format(sequence_key=sequence_key, camera_number=c_num) for c_num in range(1, 5)]
    return camera_file_paths


def extract_frame(video_path, frame_number, rgb=False):
    video_capture = cv2.VideoCapture(video_path)
    if not video_capture.isOpened():
        print("Error: Could not open video.")
        return None

    # Printing some video stats
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total number of frames: {total_frames}") 
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    print(f"Frames Per Second (FPS): {fps}")
    frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Frame Height: {frame_height}, frame Width: {frame_width}")
    
    video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    success, frame = video_capture.read()

    if not success:
        print(f"Error: Could not read frame {frame_number}.")
        return None

    video_capture.release()

    if rgb:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    return frame


class MediaPipeEstimator:
    def __init__(self):
        model_path = './mediapipe_models/pose_landmarker_heavy.task'
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
                # print(f"[Frame {frame_num}] Not all landmarks found!")
                return []
 
            x_frame = frame_w * x_norm
            y_frame = frame_h * y_norm
            
            points = list(zip(np.round(x_frame).astype(int), np.round(y_frame).astype(int)))
            return points
            
        else:
            # print(f"[Frame {frame_num}] Landmarks not found!")
            return []


def save_frame_to_file(video_path, frame_number, output_image):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Cannot open video file.")
        exit()
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    
    if ret:
        cv2.imwrite(output_image, frame)
        print(f"Frame {frame_number} saved as {output_image}")
    else:
        print(f"Error: Could not read frame {frame_number}")
    
    cap.release()