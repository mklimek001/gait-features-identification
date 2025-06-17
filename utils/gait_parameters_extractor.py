import math
import numpy as np
from dataclasses import dataclass, field, fields
from typing import Sequence, Mapping, Tuple


@dataclass
class CoordinatesIdx:
    x: int = field(default=0)
    y: int = field(default=1)
    z: int = field(default=2)

    def __post_init__(self):
        assert self.x != self.y
        assert self.x != self.z
        assert self.y != self.z
        for field in fields(self):
            assert getattr(self, field.name) in [0, 1, 2]


class StepsNotFoundException(Exception):
    pass

class ParametersExtractionException(Exception):
    pass

class GaitParametersExtractor:
    FPS = 25
    FRAME_TIME = 1 / FPS
    """
    Class to extract basic gait parameters based on 3D frame.
    """

    def __init__(
        self,
        sequence_parameters: Sequence[Mapping],
        coordintates_idx: CoordinatesIdx = CoordinatesIdx,
        scale_factor: int = 255,
        minima_window_size: int = 10,
    ):
        self.seq_params = sequence_parameters
        self.scale_factor = scale_factor
        self.c_idx = coordintates_idx
        self.l_steps, self.r_steps = self._find_step_frames(minima_window_size)
        self.all_steps = sorted(self.l_steps + self.r_steps)
        self.start_position, self.finish_position = (
            self._find_start_and_finish_position()
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

    def _find_step_frames(self, window_size: int) -> Tuple[Sequence, Sequence]:
        """
        Function to find step frames (minimum foot marker position in sequence.
        Output as two lists - first with frames number with left foot steps, second for right foot.
        """
        lfoot_height_z = [
            frame["lfoot"][self.c_idx.z] * self.scale_factor
            for frame in self.seq_params
        ]
        rfoot_height_z = [
            frame["rfoot"][self.c_idx.z] * self.scale_factor
            for frame in self.seq_params
        ]
        left_minima = self.__find_local_minima(lfoot_height_z, window_size)
        right_minima = self.__find_local_minima(rfoot_height_z, window_size)

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
            raise StepsNotFoundException("Falied to find proper step frame keys")
        
        if len(left_minima) < 2:
            raise ParametersExtractionException("Less than two steps from left leg found")
        if len(right_minima) < 2:
            raise ParametersExtractionException("Less than two steps from right leg found")

        return left_minima, right_minima

    def calculate_mean_stride_time_and_length(self) -> Sequence[float]:
        """
        Stride time - the time between the consecutive initial contacts of the same foot.
        Stride length - the distance between the consecutive initial contacts of the same foot.
        Output as six floats: left foot mean stride time, right foot mean stride time, mean stride time,
        left foot mean stride length, right foot mean stride length, mean stride length.
        Time unit is [s], distance unit is [mm].
        """

        l_stride_times = []
        l_stride_dist = []
        for i in range(len(self.l_steps) - 1):
            l_stride_times.append(
                self.FRAME_TIME * (self.l_steps[i + 1] - self.l_steps[i])
            )
            l_stride_dist.append(
                self._calculate_distance_between_projections(
                    self.seq_params[self.l_steps[i + 1]]["lfoot"],
                    self.seq_params[self.l_steps[i]]["lfoot"],
                )
            )

        r_stride_times = []
        r_stride_dist = []
        for i in range(len(self.r_steps) - 1):
            r_stride_times.append(
                self.FRAME_TIME * (self.r_steps[i + 1] - self.r_steps[i])
            )
            r_stride_dist.append(
                self._calculate_distance_between_projections(
                    self.seq_params[self.r_steps[i + 1]]["rfoot"],
                    self.seq_params[self.r_steps[i]]["rfoot"],
                )
            )

        stride_times = r_stride_times + l_stride_times
        l_mean_stride_time = sum(l_stride_times) / len(l_stride_times)
        r_mean_stride_time = sum(r_stride_times) / len(r_stride_times)
        total_mean_stride_time = sum(stride_times) / len(stride_times)

        stride_dist = r_stride_dist + l_stride_dist
        l_mean_stride_dist = sum(l_stride_dist) / len(l_stride_dist)
        r_mean_stride_dist = sum(r_stride_dist) / len(r_stride_dist)
        total_mean_stride_dist = sum(stride_dist) / len(stride_dist)

        return (
            l_mean_stride_time,
            r_mean_stride_time,
            total_mean_stride_time,
            l_mean_stride_dist,
            r_mean_stride_dist,
            total_mean_stride_dist,
        )

    def calculate_mean_step_time_length_and_width(self) -> Sequence[float]:
        """
        Step time - the time between the initial contact of one foot and the initial contact of the contralateral foot.
        Step length - the distance between the initial contact of one foot and the initial contact of the contralateral foot.
        Step width - the distance between the points of initial contact of opposite feet during a step.
        Output as nine floats: left to right foot mean step time, right to left foot mean step time, total mean step time,
        left to right foot mean step length, right to left foot mean step length, total mean step length,
        left to right foot mean step width, right to left foot mean step width, total mean step width.
        Time unit is [s], distance unit is [mm].
        """

        step_times = []
        l_r_step_dist = []
        r_l_step_dist = []
        l_r_step_width = []
        r_l_step_width = []
        for i in range(len(self.all_steps) - 1):
            step_times.append(
                self.FRAME_TIME * (self.all_steps[i + 1] - self.all_steps[i])
            )
            if self.all_steps[i] in self.l_steps:
                l_r_step_dist.append(
                    self._calculate_distance_between_projections(
                        self.seq_params[self.all_steps[i]]["lfoot"],
                        self.seq_params[self.all_steps[i + 1]]["rfoot"],
                    )
                )

                l_r_step_width.append(
                    self._calculate_distance_to_gait_axis(
                        self.seq_params[self.all_steps[i + 1]]["rfoot"]
                    )
                    + self._calculate_distance_to_gait_axis(
                        self.seq_params[self.all_steps[i]]["lfoot"]
                    )
                )

            else:
                r_l_step_dist.append(
                    self._calculate_distance_between_projections(
                        self.seq_params[self.all_steps[i]]["rfoot"],
                        self.seq_params[self.all_steps[i + 1]]["lfoot"],
                    )
                )

                r_l_step_width.append(
                    self._calculate_distance_to_gait_axis(
                        self.seq_params[self.all_steps[i + 1]]["lfoot"]
                    )
                    + self._calculate_distance_to_gait_axis(
                        self.seq_params[self.all_steps[i]]["rfoot"]
                    )
                )

        if self.all_steps[0] == self.l_steps[0]:
            l_r_step_time = step_times[0::2]
            r_l_step_time = step_times[1::2]
        else:
            l_r_step_time = step_times[1::2]
            r_l_step_time = step_times[0::2]

        l_r_mean_step_times = sum(l_r_step_time) / len(l_r_step_time)
        r_l_mean_step_times = sum(r_l_step_time) / len(r_l_step_time)
        total_mean_step_times = sum(step_times) / len(step_times)

        step_dist = l_r_step_dist + r_l_step_dist
        total_mean_step_distance = sum(step_dist) / len(step_dist)
        l_r_mean_step_dist = sum(l_r_step_dist) / len(l_r_step_dist)
        r_l_mean_step_dist = sum(r_l_step_dist) / len(r_l_step_dist)

        step_width = l_r_step_width + r_l_step_width
        total_mean_step_width = sum(step_width) / len(step_width)
        l_r_mean_step_width = sum(l_r_step_width) / len(l_r_step_width)
        r_l_mean_step_width = sum(r_l_step_width) / len(r_l_step_width)

        return (
            l_r_mean_step_times,
            r_l_mean_step_times,
            total_mean_step_times,
            l_r_mean_step_dist,
            r_l_mean_step_dist,
            total_mean_step_distance,
            l_r_mean_step_width,
            r_l_mean_step_width,
            total_mean_step_width,
        )

    def _calculate_distance_to_gait_axis(self, point: Sequence[float]) -> float:
        """
        Calculate the distance from given point to the line defined by start and end participant position.
        Each point is a tuple of (x, y).
        """
        x1, y1 = self.start_position
        x2, y2 = self.finish_position

        x0, y0 = (
            point[self.c_idx.x] * self.scale_factor,
            point[self.c_idx.y] * self.scale_factor,
        )

        numerator = abs((x2 - x1) * (y1 - y0) - (x1 - x0) * (y2 - y1))
        denominator = math.hypot(x2 - x1, y2 - y1)

        return numerator / denominator

    def _project_point_onto_line(
        self, point: Tuple[float, float]
    ) -> Tuple[float, float]:
        """
        Projects a point onto the line defined by p_start and p_stop.
        """
        p_start = np.array(self.start_position)
        p_stop = np.array(self.finish_position)
        point = np.array(point)

        line_vec = p_stop - p_start
        line_unit = line_vec / np.linalg.norm(line_vec)

        point_vec = point - p_start
        projection_length = np.dot(point_vec, line_unit)

        projection = p_start + projection_length * line_unit
        return projection

    def _calculate_distance_between_projections(
        self, point_a: Sequence[float], point_b: Sequence[float]
    ):
        """
        Projects a and b onto the line through start and finish position,
        and returns the distance between those projections.
        """

        proj_a = self._project_point_onto_line(
            (
                point_a[self.c_idx.x] * self.scale_factor,
                point_a[self.c_idx.y] * self.scale_factor,
            )
        )
        proj_b = self._project_point_onto_line(
            (
                point_b[self.c_idx.x] * self.scale_factor,
                point_b[self.c_idx.y] * self.scale_factor,
            )
        )

        return np.linalg.norm(proj_b - proj_a)

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

    import numpy as np

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
        Calculates the angle in degrees between two 3D vectors.
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

    @staticmethod
    def _mean(sequence: list) -> float:
        return sum(sequence) / len(sequence)

    def _calculate_angle_between_joints(
        self, joint_pair_1: tuple, joint_pair_2: tuple, frame_number: int
    ):
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
        legs_angles = []
        for i in range(len(self.seq_params)):
            legs_angles.append(
                self._calculate_angle_between_joints(
                    ("rtibia", "rfemur"), ("ltibia", "lfemur"), i
                )
            )
        return legs_angles

    def get_knees_angle(self):
        l_knee_angles = []
        r_knee_angles = []
        for i in range(len(self.seq_params)):
            r_knee_angles.append(
                self._calculate_angle_between_joints(
                    ("rfemur", "rtibia"), ("rfoot", "rtibia"), i
                )
            )
            l_knee_angles.append(
                self._calculate_angle_between_joints(
                    ("lfemur", "ltibia"), ("lfoot", "ltibia"), i
                )
            )
        return l_knee_angles, r_knee_angles

    def get_hip_angle(self):
        l_hip_angles = []
        r_hip_angles = []
        for i in range(len(self.seq_params)):
            r_hip_angles.append(
                self._calculate_angle_between_joints(
                    ("rhumerus", "rfemur"), ("rtibia", "rfemur"), i
                )
            )
            l_hip_angles.append(
                self._calculate_angle_between_joints(
                    ("lhumerus", "lfemur"), ("ltibia", "lfemur"), i
                )
            )
        return l_hip_angles, r_hip_angles

    def calculate_avg_max_hip_angle_per_stride(self) -> tuple[float, float, float]:
        """
        Calculate max hip angle for each step in sequence
        Returns mean of those max hip angle for steps of left, right and both feet.
        """

        l_hip_angles, r_hip_angles = self.get_hip_angle()
        (
            l_max_hip_angle_diff,
            r_max_hip_angle_diff,
        ) = (
            [],
            [],
        )

        for i in range(len(self.l_steps) - 1):
            l_max_hip_angle_diff.append(
                max(l_hip_angles[self.l_steps[i] : self.l_steps[i + 1]])
                - min(l_hip_angles[self.l_steps[i] : self.l_steps[i + 1]])
            )

        for i in range(len(self.r_steps) - 1):
            r_max_hip_angle_diff.append(
                max(r_hip_angles[self.r_steps[i] : self.r_steps[i + 1]])
                - min(r_hip_angles[self.r_steps[i] : self.r_steps[i + 1]])
            )

        return (
            self._mean(l_max_hip_angle_diff),
            self._mean(r_max_hip_angle_diff),
            self._mean(l_max_hip_angle_diff + r_max_hip_angle_diff),
        )

    def calculate_avg_max_knee_angle_per_stride(self) -> tuple[float, float, float]:
        """
        Calculate max knee angle for each stride in sequence
        Returns mean of those max knee angle for steps of left, right and both feet.
        """

        l_knee_angles, r_knee_angles = self.get_knees_angle()
        (
            l_max_knee_angle_diff,
            r_max_knee_angle_diff,
        ) = (
            [],
            [],
        )

        for i in range(len(self.l_steps) - 1):
            l_max_knee_angle_diff.append(
                max(l_knee_angles[self.l_steps[i] : self.l_steps[i + 1]])
                - min(l_knee_angles[self.l_steps[i] : self.l_steps[i + 1]])
            )

        for i in range(len(self.r_steps) - 1):
            r_max_knee_angle_diff.append(
                max(r_knee_angles[self.r_steps[i] : self.r_steps[i + 1]])
                - min(r_knee_angles[self.r_steps[i] : self.r_steps[i + 1]])
            )

        return (
            self._mean(l_max_knee_angle_diff),
            self._mean(r_max_knee_angle_diff),
            self._mean(l_max_knee_angle_diff + r_max_knee_angle_diff),
        )

    def calculate_avg_max_legs_angle_for_steps(self) -> tuple[float, float, float]:
        """
        Calculate max legs angle in each step.
        Returns mean max legs angle for left to right foot step, right to left foot step and all steps.
        """
        l_r_step_l_angle, r_l_step_l_angle = [], []
        legs_angle = self.get_legs_angle()
        for i in range(len(self.all_steps) - 1):
            if self.all_steps[i] in self.l_steps:
                l_r_step_l_angle.append(
                    max(
                        legs_angle[self.all_steps[i]], legs_angle[self.all_steps[i] + 1]
                    )
                )

            else:
                r_l_step_l_angle.append(
                    max(
                        legs_angle[self.all_steps[i]], legs_angle[self.all_steps[i] + 1]
                    )
                )

        return (
            self._mean(l_r_step_l_angle),
            self._mean(r_l_step_l_angle),
            self._mean(r_l_step_l_angle + l_r_step_l_angle),
        )
