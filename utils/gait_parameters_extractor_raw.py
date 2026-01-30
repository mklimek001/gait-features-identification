import math
from typing import Sequence, Mapping, Tuple, NamedTuple

import numpy as np

from utils.gait_parameters_extractor import CoordinatesIdx


class GaitAngles(NamedTuple):
    legs_angles: Sequence[float]
    left_knee_angles: Sequence[float]
    right_knee_angles: Sequence[float]
    left_hip_angles: Sequence[float]
    right_hip_angles: Sequence[float]
    left_humerus_angles: Sequence[float]
    right_humerus_angles: Sequence[float]
    left_elbow_angles: Sequence[float]
    right_elbow_angles: Sequence[float]


class DistanceParameters(NamedTuple):
    ankle_distances: Sequence[float]
    knee_distances: Sequence[float]
    elbow_distances: Sequence[float]
    hand_distances: Sequence[float]


class PelvicParameters(NamedTuple):
    center_of_gravity_height_change: Sequence[float]
    lateral_pelvic_tilt: Sequence[float]
    pelvis_rotation: Sequence[float]


class GaitParametersExtractorRaw:
    """
    Class to extract basic gait parameters based on single gait cycle.
    """

    def __init__(
        self,
        sequence_parameters: Sequence[Mapping],
        coordinates_idx: CoordinatesIdx = CoordinatesIdx(),
        scale_factor: int = 255,
        running_average_window_size: int = 1,
    ):
        self.seq_params = self._smooth_data(
            sequence_parameters, window_size=running_average_window_size
        )
        self.scale_factor = scale_factor
        self.c_idx = coordinates_idx
        self.start_position, self.finish_position = (
            self._find_start_and_finish_position()
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
        smoothed_data = []
        for idx, frame_parameters in enumerate(sequence_parameters):
            smoothed_frame = {}
            frames_window = sequence_parameters[
                max(0, idx - margin) : min(idx + margin + 1, len(sequence_parameters))
            ]
            for joint_name, joint_values in frame_parameters.items():
                # lfoot, [1,2 3]
                smoothed_joint_values = []
                for j in range(len(joint_values)):
                    smoothed_joint_values.append(
                        self._mean([frame[joint_name][j] for frame in frames_window])
                    )

                assert len(smoothed_joint_values) == 3
                smoothed_frame[joint_name] = smoothed_joint_values

            assert frame_parameters.keys() == smoothed_frame.keys()
            smoothed_data.append(smoothed_frame)

        assert len(sequence_parameters) == len(smoothed_data)

        return smoothed_data

    def get_gait_parameters(self) -> np.array:
        """
        Get gait parameters as numpy array,
        ready to use as input to neural network
        """
        return np.array(
            [
                *self.get_gait_angles(),
                *self.get_joint_distances(),
                *self.get_pelvic_parameters(),
            ]
        )

    def get_gait_parameters_wo_hands(self) -> np.array:
        """
        Get gait parameters as numpy array, ready to use as input to neural network.
        Select only parameters without hands (not involving elbows and wrists markers).
        """

        gait_angles = self.get_gait_angles()
        joint_distances = self.get_joint_distances()
        pelvic_parameters = self.get_pelvic_parameters()

        return np.array(
            [
                gait_angles.left_hip_angles,
                gait_angles.right_hip_angles,
                gait_angles.right_knee_angles,
                gait_angles.left_knee_angles,
                gait_angles.legs_angles,
                joint_distances.ankle_distances,
                joint_distances.knee_distances,
                pelvic_parameters.center_of_gravity_height_change,
                pelvic_parameters.lateral_pelvic_tilt,
                pelvic_parameters.pelvis_rotation,
            ]
        )

    def get_gait_parameters_names(self) -> Sequence[str]:
        """
        Get gait parameters names in same order as in results of
        `get_gait_parameters` function call()

        """

        return [
            *GaitAngles._fields,
            *DistanceParameters._fields,
            *PelvicParameters._fields,
        ]

    def get_gait_parameters_names_wo_hands(self) -> np.array:
        """
        Get gait parameters as numpy array, ready to use as input to neural network.
        Select only parameters without hands (not involving elbows and wrists markers).
        """

        return [
            "left_hip_angles",
            "right_hip_angles",
            "right_knee_angles",
            "left_knee_angles",
            "legs_angles",
            "ankle_distances",
            "knee_distances",
            "center_of_gravity_height_change",
            "lateral_pelvic_tilt",
            "pelvis_rotation",
        ]

    def get_gait_angles(self) -> GaitAngles:
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

        return GaitAngles(
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

    def get_joint_distances(self) -> DistanceParameters:
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

        return DistanceParameters(
            ankle_distances, knee_distances, elbow_distances, hand_distances
        )

    def get_pelvic_parameters(
        self,
    ) -> PelvicParameters:
        """
        Calculate:
        - Center of gravity height change - change of distance from center of gravity
          (center of pelvis) to ground level.
        - Lateral pelvic tilt - angle between vertical and line going through hips
        - Pelvis rotation - angle between line going from start to end position of the walker,
          and line going through hips.
        """
        center_of_gravity_height = [
            (frame["lfemur"][self.c_idx.z] + frame["rfemur"][self.c_idx.z])
            * self.scale_factor
            * 0.5
            for frame in self.seq_params
        ]

        center_of_gravity_height_change = [0] + [
            center_of_gravity_height[i] - center_of_gravity_height[i - 1]
            for i in range(1, len(center_of_gravity_height))
        ]

        lfemur_x_z = [
            (
                frame["lfemur"][self.c_idx.x] * self.scale_factor,
                frame["lfemur"][self.c_idx.z] * self.scale_factor,
            )
            for frame in self.seq_params
        ]
        rfemur_x_z = [
            (
                frame["rfemur"][self.c_idx.x] * self.scale_factor,
                frame["rfemur"][self.c_idx.z] * self.scale_factor,
            )
            for frame in self.seq_params
        ]

        lateral_pelvic_tilt = []

        for p1, p2 in zip(lfemur_x_z, rfemur_x_z):
            angle_radians = math.atan2(p2[0] - p1[0], p2[1] - p1[1])
            angle_degrees = math.degrees(angle_radians)
            lateral_pelvic_tilt.append(angle_degrees)

        lfemur_x_y = [
            (
                frame["lfemur"][self.c_idx.x] * self.scale_factor,
                frame["lfemur"][self.c_idx.y] * self.scale_factor,
            )
            for frame in self.seq_params
        ]
        rfemur_x_y = [
            (
                frame["rfemur"][self.c_idx.x] * self.scale_factor,
                frame["rfemur"][self.c_idx.y] * self.scale_factor,
            )
            for frame in self.seq_params
        ]

        pelvis_rotation = []

        for p1, p2 in zip(lfemur_x_y, rfemur_x_y):
            pelvis_rotation.append(
                self.__angle_between_vectors(
                    v1=(p1[0] - p2[0], p1[1] - p2[1]),
                    v2=(
                        self.start_position[0] - self.finish_position[0],
                        self.start_position[1] - self.finish_position[1],
                    ),
                )
            )

        return PelvicParameters(
            center_of_gravity_height_change, lateral_pelvic_tilt, pelvis_rotation
        )

    def _find_start_and_finish_position(
        self,
    ) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """
        Function to find start and finish position of sequence - useful to obtain
        """
        start_position = (
            (
                self.seq_params[0]["lfoot"][self.c_idx.x]
                + self.seq_params[0]["rfoot"][self.c_idx.x]
            )
            / 2
            * self.scale_factor,
            (
                self.seq_params[0]["lfoot"][self.c_idx.y]
                + self.seq_params[0]["rfoot"][self.c_idx.y]
            )
            / 2
            * self.scale_factor,
        )

        finish_position = (
            (
                self.seq_params[-1]["lfoot"][self.c_idx.x]
                + self.seq_params[-1]["rfoot"][self.c_idx.x]
            )
            / 2
            * self.scale_factor,
            (
                self.seq_params[-1]["lfoot"][self.c_idx.y]
                + self.seq_params[-1]["rfoot"][self.c_idx.y]
            )
            / 2
            * self.scale_factor,
        )

        return start_position, finish_position

    @staticmethod
    def _mean(sequence: list) -> float:
        return sum(sequence) / len(sequence)

    @staticmethod
    def __project_point_on_plane(point, plane_point, plane_normal):
        """
        Projects a 3D point onto a plane.Returns projected 3D point on the plane.
        """
        plane_normal = plane_normal / np.linalg.norm(plane_normal)
        vector_to_point = point - plane_point
        distance = np.dot(vector_to_point, plane_normal)
        projected_point = point - distance * plane_normal
        return projected_point

    @staticmethod
    def __angle_between_vectors(v1, v2):
        """
        Calculates the angle in degrees between two N-dimension vectors.
        """
        dot_product = np.dot(v1, v2)
        magnitude_v1 = np.linalg.norm(v1)
        magnitude_v2 = np.linalg.norm(v2)

        if magnitude_v1 == 0 or magnitude_v2 == 0:
            return 0.0

        cosine_angle = dot_product / (magnitude_v1 * magnitude_v2)
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
        angle_rad = np.arccos(cosine_angle)
        angle_deg = np.degrees(angle_rad)
        return angle_deg

    def _calculate_angle_between_joints(
        self, joint_pair_1: tuple, joint_pair_2: tuple, frame_number: int
    ):
        """
        Calculate angle between joints in sagittal plane.
        """
        p1 = np.array([self.start_position[0], self.start_position[1], 0.0])
        p2 = np.array([self.start_position[0], self.start_position[1], 1.0])
        p3 = np.array([self.finish_position[0], self.finish_position[1], 0.0])

        A_orig = (
            np.array(self.seq_params[frame_number][joint_pair_1[0]]) * self.scale_factor
        )
        B_orig = (
            np.array(self.seq_params[frame_number][joint_pair_1[1]]) * self.scale_factor
        )
        C_orig = (
            np.array(self.seq_params[frame_number][joint_pair_2[0]]) * self.scale_factor
        )
        D_orig = (
            np.array(self.seq_params[frame_number][joint_pair_2[1]]) * self.scale_factor
        )

        new_order_indices = [self.c_idx.x, self.c_idx.y, self.c_idx.z]

        A = A_orig[new_order_indices]
        B = B_orig[new_order_indices]
        C = C_orig[new_order_indices]
        D = D_orig[new_order_indices]

        v1 = p2 - p1
        v2 = p3 - p1
        plane_normal = np.cross(v1, v2)

        if np.linalg.norm(plane_normal) == 0:
            print(
                "Error: The points p1, p2, and p3 are collinear and do not define a unique plane."
            )
            return 0
        else:
            A_proj = self.__project_point_on_plane(A, p1, plane_normal)
            B_proj = self.__project_point_on_plane(B, p1, plane_normal)
            C_proj = self.__project_point_on_plane(C, p1, plane_normal)
            D_proj = self.__project_point_on_plane(D, p1, plane_normal)

            line1_direction = B_proj - A_proj
            line2_direction = D_proj - C_proj

            if (
                np.linalg.norm(line1_direction) == 0
                or np.linalg.norm(line2_direction) == 0
            ):
                print(
                    "One or both projected lines are effectively points (start and end points are the same). Angle is undefined or 0."
                )
                return 0
            else:
                angle = self.__angle_between_vectors(line1_direction, line2_direction)
                return angle

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
        return self.scale_factor * math.sqrt(
            sum((val1 - val2) ** 2 for val1, val2 in zip(vector1, vector2))
        )

    def _get_l_r_joint_distance(self, joint: str):
        distances = []
        for frame_data in self.seq_params:
            distances.append(
                self.calculate_euclidean_distance(
                    frame_data[f"l{joint}"], frame_data[f"r{joint}"]
                )
            )

        return distances
