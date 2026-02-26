from scripts.parsers import parse_sequences
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