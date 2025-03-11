from pathlib import Path
import os
from tqdm import tqdm


def convert_asf_amc_to_bvh(dataset_path):
    # base command for windows dowwnloaded from https://github.com/thcopeland/amc2bvh/releases
    base_command = 'amc2bvh.exe {source_asf} {source_amc} -o {dest_bvh}'
    
    for seq_directory in tqdm(dataset_path.iterdir(), desc="Processing directories"):
        mocap_seq_directory = seq_directory.joinpath('MoCap')
        
        if mocap_seq_directory.exists() \
            and (asf_file := next(mocap_seq_directory.glob("*.asf"), False)) \
            and (amc_file := next(mocap_seq_directory.glob("*.amc"), False)):

            new_bvh_file = mocap_seq_directory.joinpath(f'{Path(asf_file).stem}.bvh')
            command = base_command.format(source_asf=asf_file, source_amc=amc_file, dest_bvh=new_bvh_file)
            result = os.system(command)


if __name__ == "__main__":
    dataset_path = Path('./gait3d/Sequences')
    convert_asf_amc_to_bvh(dataset_path)
