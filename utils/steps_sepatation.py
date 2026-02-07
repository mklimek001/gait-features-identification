from utils.gait_parameters_extractor_v2 import GaitParametersExtractorV2
from utils.gait_parameters_extractor import CoordinatesIdx


def separate_gait_sequences_to_cycles(
    data_3d,
    window_size: int = 32,
    minima_window_size: int = 10,
    coordinates_index: CoordinatesIdx = CoordinatesIdx(),
    scale_factor: int = 255,
    smooth_butterworth: bool = False,
    running_average_window_size: int = 1,
    # butterworth filter parameters
    cutoff_frequency: int = 5,
    order: int = 2,
    fs: int = 200,
    print_stats: bool = True,
):
    results = {}
    cum_step_frames = []
    side = ""
    fail_counter = 0

    for seq_key in list(data_3d.keys()):
        gpe = GaitParametersExtractorV2(
            sequence_parameters=data_3d[seq_key],
            coordintates_idx=coordinates_index,
            scale_factor=scale_factor,
            minima_window_size=minima_window_size,
            running_average_window_size=running_average_window_size,
            smooth_butterworth=smooth_butterworth,
            cutoff_frequency=cutoff_frequency,
            order=order,
            fs=fs,
        )
        if len(gpe.l_steps) > 1 and len(gpe.r_steps) > 1:
            steps_sequence = []
            if gpe.l_steps[0] < gpe.r_steps[0]:
                cum_step_frames += _calc_step_frames(gpe.l_steps)
                steps_sequence = _fragment_step_frames(gpe.l_steps, window_size)
                side = "L"
            else:
                cum_step_frames += _calc_step_frames(gpe.r_steps)
                steps_sequence = _fragment_step_frames(gpe.r_steps, window_size)
                side = "R"

            if print_stats:
                print(f"{seq_key} [{side}] - {steps_sequence}")
            for i, (start_frame, end_frame) in enumerate(steps_sequence):
                results[f"{seq_key}c{i}"] = gpe.seq_params[start_frame:end_frame]
        else:
            fail_counter += 1

    if print_stats:
        print("   Steps: ", len(cum_step_frames))
        print("    Mean: ", sum(cum_step_frames) / len(cum_step_frames))
        print("     Max: ", max(cum_step_frames))
        print("     Min: ", min(cum_step_frames))
        print("  Median: ", sorted(cum_step_frames)[len(cum_step_frames) // 2])
        print(" Over 32: ", sum(1 for step in cum_step_frames if step > 32))
        print("  Failed: ", fail_counter)

    return results


def _calc_step_frames(steps: list) -> list[int]:
    step_frames = []
    for i in range(len(steps) - 1):
        step_frames.append(steps[i + 1] - steps[i])
    return step_frames


def _fragment_step_frames(steps: list, window_size: int = 32) -> list[tuple[int, int]]:
    step_frames = []
    for i in range(len(steps) - 1):
        center = steps[i] + (steps[i + 1] - steps[i]) // 2
        start_frame = center - window_size // 2
        end_frame = center + window_size // 2
        step_frames.append((start_frame, end_frame))
    return step_frames
