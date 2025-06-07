import re

def parse_sequences(file_path: str) -> dict:
    sequence_dict = {}

    details_pattern = re.compile(
        r"start frame: (\d+), number of frames: (\d+), frames offset: (-?\d+), MoCap data: (\w+)"
    )

    with open(file_path, 'r') as file:
        for line in file:
            if line.strip().startswith('* p'):
                last_key = line.split(' ')[1]
            
            detail_match = details_pattern.search(line.strip())
            if detail_match:
                sequence_dict[last_key] = {
                    "start_frame": int(detail_match.group(1)),
                    "number_of_frames": int(detail_match.group(2)),
                    "frame_offset": int(detail_match.group(3)),
                    "MoCap_data": detail_match.group(4) == "Yes"
                }
                
                
    
    return sequence_dict
