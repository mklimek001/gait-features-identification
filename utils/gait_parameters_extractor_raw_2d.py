import math
from typing import Sequence, Mapping, Tuple, NamedTuple
from dataclasses import dataclass, field, fields
import numpy as np


@dataclass
class CoordinatesIdx2D:
    x: int = field(default=0)
    y: int = field(default=1)

    def __post_init__(self):
        assert self.x != self.y
        for _field in fields(self):
            assert getattr(self, _field.name) in [0, 1]


class GaitAngles2D(NamedTuple):
    legs_angles: Sequence[float]
    left_knee_angles: Sequence[float]
    right_knee_angles: Sequence[float]
    left_hip_angles: Sequence[float]
    right_hip_angles: Sequence[float]
    left_humerus_angles: Sequence[float]
    right_humerus_angles: Sequence[float]
    left_elbow_angles: Sequence[float]
    right_elbow_angles: Sequence[float]


class DistanceParameters2D(NamedTuple):
    ankle_distances: Sequence[float]
    knee_distances: Sequence[float]
    elbow_distances: Sequence[float]
    hand_distances: Sequence[float]


class PelvicParameters2D(NamedTuple):
    center_of_gravity_height_change: Sequence[float]


class GaitParametersExtractorRaw2D:
    """
    Class to extract basic gait parameters based on single gait cycle in 2D sequence.
    """

    def __init__(
        self,
        sequence_parameters: Sequence[Mapping],
        selected_joints: Mapping[
            str, str
        ],  # mapping between indexes and selected joint names in frame array
        coordinates_idx: CoordinatesIdx2D = CoordinatesIdx2D(),
        window_size: int = 1,
    ):
        self.seq_params = self._smooth_data(
            sequence_parameters, window_size=window_size
        )
        self.name_idx_mapping = self._reverse_idx_mapping(selected_joints)
        self.c_idx = coordinates_idx

    def get_gait_parameters(self) -> np.array:
        """
        Get gait parameters as numpy array,
        ready to use as input to neural network
        """
        return np.array(
            [
                *self.get_gait_angles(),
                *self.get_joint_distances(),
                self.get_pelvic_parameters(),
            ]
        )

    def _smooth_data(
        self, sequence_parameters: Sequence[Mapping], window_size: int = 1
    ) -> Sequence[Mapping]:
        """Smooth sequence parameters data with running average."""
        if window_size % 2 == 0:

            print(
                f"Provided window size ({window_size}) is even, bigger window size ({window_size + 1}) will be used instead."
            )
            window_size += 1

        if window_size == 1:
            return sequence_parameters

        margin = window_size // 2
        smoothed_data = {}
        for idx, _ in sequence_parameters.items():
            smoothed_frame = []
            
            frames_window = list(sequence_parameters.values())[
                max(0, int(idx) - margin) : min(
                    int(idx) + margin + 1, len(sequence_parameters)
                )
            ]

            for i in range(len(frames_window[0])):
                smoothed_frame.append([
                    self._mean([frame[i][0] for frame in frames_window]),
                    self._mean([frame[i][1] for frame in frames_window]),
                ])

            smoothed_data[idx] = smoothed_frame

        assert smoothed_data.keys() == sequence_parameters.keys()

        return smoothed_data

    def get_step_frames(self, window_size: int = 10) -> Tuple[Sequence, Sequence]:
        """
        Function to find step frames (minimum foot marker position in sequence.
        Output as two lists - first with frames number with left foot steps, second for right foot.
        """
        lfoot_height_z = [
            frame[self.name_idx_mapping["lfoot"]][self.c_idx.y]
            for frame in self.seq_params.values()
        ]
        rfoot_height_z = [
            frame[self.name_idx_mapping["rfoot"]][self.c_idx.y]
            for frame in self.seq_params.values()
        ]
        left_minima = self.__find_local_minima(lfoot_height_z, window_size)
        right_minima = self.__find_local_minima(rfoot_height_z, window_size)

        if len(left_minima) <= 1 or len(right_minima) <= 1:
            print("Failed to find proper step frame keys")
            return [], []

        if not self.__check_if_left_right_alternately(left_minima, right_minima):
            # If not left right alternately check without first or last item on list - start and end of sequence might be problematic
            if left_minima[0] <= right_minima[
                0
            ] and self.__check_if_left_right_alternately(left_minima[1:], right_minima):
                left_minima = left_minima[1:]
                print(
                    "First left step recognized as probably marked incorrectly and removed"
                )
            elif right_minima[0] <= left_minima[
                0
            ] and self.__check_if_left_right_alternately(left_minima, right_minima[1:]):
                right_minima = right_minima[1:]
                print(
                    "First right step recognized as probably marked incorrectly and removed"
                )
            elif left_minima[-1] <= right_minima[
                -1
            ] and self.__check_if_left_right_alternately(
                left_minima[:-1], right_minima
            ):
                left_minima = left_minima[:-1]
                print(
                    "Last left step recognized as probably marked incorrectly and removed"
                )
            elif right_minima[-1] <= left_minima[
                -1
            ] and self.__check_if_left_right_alternately(
                left_minima, right_minima[:-1]
            ):
                right_minima = right_minima[:-1]
                print(
                    "Last right step recognized as probably marked incorrectly and removed"
                )

        if not self.__check_if_left_right_alternately(left_minima, right_minima):
            print("Failed to find proper step frame keys")
            return [], []

        return left_minima, right_minima

    def get_gait_parameters_names(self) -> Sequence[str]:
        """
        Get gait parameters names in same order as in results of
        `get_gait_parameters` function call()

        """

        return [
            *GaitAngles2D._fields,
            *DistanceParameters2D._fields,
            *PelvicParameters2D._fields,
        ]

    def get_gait_angles(self) -> GaitAngles2D:
        """
        Calculate vector of angles in each frame of gait cycle:
        - Legs angle
        - Left and right knee angle
        - Left and right hip angle
        - Left and right humerus angle
        - Left and right elbow angle
        -
        """
        legs_angles = self.get_legs_angle()
        left_knee_angles, right_knee_angles = self.get_l_r_joint_angle(
            "femur", "tibia", "foot"
        )
        left_hip_angles, right_hip_angles = self.get_l_r_joint_angle(
            "humerus", "femur", "tibia"
        )
        left_humerus_angles, right_humerus_angles = self.get_l_r_joint_angle(
            "femur", "humerus", "radius"
        )
        left_elbow_angles, right_elbow_angles = self.get_l_r_joint_angle(
            "humerus", "radius", "wrist"
        )
        # left_ankle_angles, right_ankle_angles = self.get_l_r_joint_angle(
        #    "tibia", "foot", "toes"
        # )

        return (
            legs_angles,
            left_knee_angles,
            right_knee_angles,
            left_hip_angles,
            right_hip_angles,
            left_humerus_angles,
            right_humerus_angles,
            left_elbow_angles,
            right_elbow_angles,
        )

    def get_joint_distances(self) -> DistanceParameters2D:
        """
        Calculate distance between pair of joints across gait cycle:
        - Ankle
        - Knee
        - Elbow
        - Hand
        """

        ankle_distances = self._get_l_r_joint_distance("foot")
        knee_distances = self._get_l_r_joint_distance("tibia")
        elbow_distances = self._get_l_r_joint_distance("radius")
        hand_distances = self._get_l_r_joint_distance("wrist")

        return ankle_distances, knee_distances, elbow_distances, hand_distances

    def get_pelvic_parameters(
        self,
    ) -> PelvicParameters2D:
        """
        Calculate:
        - Center of gravity height change - change of distance from center of gravity
          (center of pelvis) to ground level.
        """
        center_of_gravity_height = [
            (
                frame[self.name_idx_mapping["lfemur"]][self.c_idx.y]
                + frame[self.name_idx_mapping["rfemur"]][self.c_idx.y]
            )
            * 0.5
            for frame in self.seq_params.values()
        ]

        center_of_gravity_height_change = [0] + [
            center_of_gravity_height[i] - center_of_gravity_height[i - 1]
            for i in range(1, len(center_of_gravity_height))
        ]

        return center_of_gravity_height_change

    def _calculate_angle_between_joints(
        self,
        joint_pair_1: tuple,
        joint_pair_2: tuple,
        frame_number: int,
        degrees: bool = True,
    ) -> float:
        """
        Calculate angle between lines connecting two joint pairs.
        Points A, B, C are given as [x, y].

        If degrees=True, returns angle in degrees.
        Otherwise, returns angle in radians.
        """

        frame = self.seq_params[str(frame_number)]
        A = frame[self.name_idx_mapping[joint_pair_1[0]]]
        B = frame[self.name_idx_mapping[joint_pair_1[1]]]
        C = frame[self.name_idx_mapping[joint_pair_2[0]]]
        D = frame[self.name_idx_mapping[joint_pair_2[1]]]

        BA = (A[0] - B[0], A[1] - B[1])
        CD = (C[0] - D[0], C[1] - D[1])

        dot = BA[0] * CD[0] + BA[1] * CD[1]
        mag_ba = math.hypot(BA[0], BA[1])
        mag_cd = math.hypot(CD[0], CD[1])

        if mag_ba == 0 or mag_cd == 0:
            # print("Angle is undefined for zero-length line.")
            return 0  # maybe not the best option but hard to find better one

        cos_theta = max(-1.0, min(1.0, dot / (mag_ba * mag_cd)))
        angle = math.acos(cos_theta)

        return math.degrees(angle) if degrees else angle

    def get_legs_angle(self):
        """
        Calculate legs angle.
        """
        legs_angles = []
        for i in range(len(self.seq_params)):
            legs_angles.append(
                self._calculate_angle_between_joints(
                    ("rtibia", "rfemur"), ("ltibia", "lfemur"), i
                )
            )
        return legs_angles

    def get_l_r_joint_angle(self, joint1: str, joint2: str, joint3: str):
        l_angles = []
        r_angles = []
        for i in range(len(self.seq_params)):
            l_angles.append(
                self._calculate_angle_between_joints(
                    (f"l{joint1}", f"l{joint2}"), (f"l{joint3}", f"l{joint2}"), i
                )
            )
            r_angles.append(
                self._calculate_angle_between_joints(
                    (f"r{joint1}", f"r{joint2}"), (f"r{joint3}", f"r{joint2}"), i
                )
            )
        return l_angles, r_angles

    def calculate_euclidean_distance(
        self, vector1: Sequence[float], vector2: Sequence[float]
    ):
        """
        Calculate euclidean distance between two vectors.
        """
        return math.sqrt(
            sum((val1 - val2) ** 2 for val1, val2 in zip(vector1, vector2))
        )

    def _get_l_r_joint_distance(self, joint: str):
        distances = []
        for frame_data in self.seq_params.values():
            distances.append(
                self.calculate_euclidean_distance(
                    frame_data[self.name_idx_mapping[f"l{joint}"]],
                    frame_data[self.name_idx_mapping[f"r{joint}"]],
                )
            )

        return distances

    @staticmethod
    def _mean(sequence: list) -> float:
        return sum(sequence) / len(sequence)

    @staticmethod
    def _reverse_idx_mapping(idx_name_mapping: Mapping[str, str]) -> Mapping[str, int]:
        return {joint_name: int(idx) for idx, joint_name in idx_name_mapping.items()}

    @staticmethod
    def __find_local_minima(data, window_size: int = 5) -> Sequence[int]:
        local_minima_indices = []

        for i in range(window_size, len(data) - window_size):
            window_prev = data[i - window_size : i]
            window_next = data[i + 1 : i + window_size + 1]
            current = data[i]

            if current < min(window_prev) and current < min(window_next):
                local_minima_indices.append(i)

        return local_minima_indices

    @staticmethod
    def __check_if_left_right_alternately(left_minima, right_minima):
        sorted_minima = sorted(right_minima + left_minima)
        order = ""

        for minim in sorted_minima:
            if minim in left_minima:
                order += "L"
            else:
                order += "R"

        alternately = True
        for i in range(len(order) - 1):
            if order[i] == order[i + 1]:
                alternately = False
                break

        return alternately
