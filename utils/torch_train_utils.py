import random
import torch
from torch.utils.data import Dataset
import torch.nn as nn
import numpy as np


def get_train_valid_test_set(sequences: list, seed: int = None):
    """
    Function to obtaining train, test and valid set from dataset described in sequences file.
    Testing and validation set will contain 10 gait sequences (5 participants, 2 sequence each).
    Training dataset will containg remaining 56 sequences.
    """

    if seed:
        random.seed(seed)

    mocap_keys = []
    par_cam_keys = []
    par_cam_person = set()
    par_after_cloth_change_keys = []
    par_after_cloth_change_person = set()

    for key, params in sequences.items():
        if params['MoCap_data']:
            mocap_keys.append(key)
            if key[-1] in ["1", "3", "5", "7"]:
                par_cam_keys.append(key)
                par_cam_person.add(key[:-2])
            if key[-1] in ["5", "7"]:
                par_after_cloth_change_keys.append(key)
                par_after_cloth_change_person.add(key[:-2])
    
    without_clothing_change = []
    while len(without_clothing_change) < 6:
        random_person = random.choice(list(par_cam_person))
        if random_person not in par_after_cloth_change_person:
            without_clothing_change.append(random_person)
            par_cam_person.remove(random_person)

    with_clothing_change = []
    while len(with_clothing_change) < 4:
        random_person = random.choice(list(par_after_cloth_change_person))
        with_clothing_change.append(random_person)
        par_after_cloth_change_person.remove(random_person)


    test_seq_set = ([f'{p_seq}s{seq_idx}' for p_seq in without_clothing_change[:3] for seq_idx in [1, 3]] +
                    [f'{p_seq}s{seq_idx}' for p_seq in with_clothing_change[:2] for seq_idx in [5, 7]])

    valid_seq_set = ([f'{p_seq}s{seq_idx}' for p_seq in without_clothing_change[3:] for seq_idx in [1, 3]] +
                    [f'{p_seq}s{seq_idx}' for p_seq in with_clothing_change[2:] for seq_idx in [5, 7]])
    

    train_seq_set = ([f'{p_seq}s{seq_idx}' for p_seq in list(par_cam_person) for seq_idx in [1, 3]] +
                    [f'{p_seq}s{seq_idx}' for p_seq in list(par_after_cloth_change_person) for seq_idx in [5, 7]])

    return train_seq_set, valid_seq_set, test_seq_set


class MoCapInputDataset(Dataset):
    """Dataset for nn to predict 3D points based on 2D images markup"""
    def __init__(self, seq_keys_list, sequences, selected_names, raw_input, raw_output):
        self.input_frames_data = {f"c{c_idx}": [] for c_idx in range(1, 5)}
        self.output_frames_data = []
        self.not_found = 0
              
        for seq_key in seq_keys_list:
            for f_idx in range(sequences[seq_key]['number_of_frames']):
                curr_output_array = []
                output_frame_dict = raw_output[seq_key][f_idx]
                for point_idx, joint_name in selected_names.items():
                    curr_output_array.append(output_frame_dict[joint_name])
        
                curr_output_array_np = np.array(curr_output_array)*255
                # 255 multiplier added to mocap to obtain distance in mm
                curr_input_arrays = {f"c{c_idx}": [] for c_idx in range(1, 5)}
        
                all_found = True
                
                for c_idx in range(1, 5):
                    input_frame_list = raw_input[seq_key][f"c{c_idx}"][str(f_idx)]
                    if [None, None] in input_frame_list:
                        all_found = False
                        break
                        
                    for point_idx, joint_name in selected_names.items(): 
                        pixel_coords = input_frame_list[int(point_idx)]
                        curr_input_arrays[f"c{c_idx}"].append(pixel_coords)

                if all_found:
                    for c_idx in range(1, 5):
                        self.input_frames_data[f"c{c_idx}"].append(np.array(curr_input_arrays[f"c{c_idx}"]))
 
                    self.output_frames_data.append(curr_output_array_np)
                else:
                    self.not_found += 1

        self.length = len(self.output_frames_data)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        inputs = [torch.from_numpy(self.input_frames_data[f"c{c_idx}"][idx]).float() for c_idx in range(1, 5)]  # each: (12, 2)
        target = torch.from_numpy(self.output_frames_data[idx]).float()  # (12, 3)
        return inputs, target
    

class MPJPE(nn.Module):
    """Mean Per Joint Position Error (MPJPE) loss function implementation"""
    def __init__(self):
        super().__init__()

    def forward(self, predictions, targets):
        # shape (batch, 12, 3)
        # compute euclidean distance for each point pair
        distances = torch.norm(predictions - targets, dim=2)
        mean_distance = distances.mean()
        return mean_distance
