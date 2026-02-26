import json
from dataclasses import dataclass, field, fields
from typing import Sequence, Mapping, Literal


@dataclass
class CoordinatesIdx:
    """Class to store coordinates indexes."""

    x: int = field(default=0)
    y: int = field(default=1)
    z: int = field(default=2)

    def __post_init__(self):
        assert self.x != self.y
        assert self.x != self.z
        assert self.y != self.z
        for class_field in fields(self):
            assert getattr(self, class_field.name) in [0, 1, 2]

    def __getitem__(self, coordinate: Literal["x", "y", "z"]) -> int:
        return self.__getattribute__(coordinate)


class GaitParameters:
    """
    Simple class that might be a base for operations on gait parameters.
    """

    def __init__(
        self,
        sequence_parameters: Sequence[Mapping[str, Sequence[float]]],
        coordinates_idx: CoordinatesIdx = CoordinatesIdx,
        scale_factor: int = 255,
    ):
        self.seq_params = sequence_parameters
        self.scale_factor = scale_factor
        self.c_idx = coordinates_idx

    def get_joint_coordinate(
        self,
        frame: int,
        joint_name: str,
        coordinate: Literal["x", "y", "z"],
    ):
        """
        Method to get given coordinate (xyz) of given joint in given sequence frame
        where (0,0,0) point is located in the middle of walking scene on the ground level,
        Z is a height axis, X might be used to obtain step width and Y for step length calculation (for pXs1 sequence)
        """

        return (
            self.seq_params[frame][joint_name][self.c_idx[coordinate]]
            * self.scale_factor
        )


if __name__ == "__main__":
    SEQUENCE_KEY = "p1s1"

    with open("./data/dataset_yolo26_triang.json", "r", encoding="utf-8") as file:
        data_yolo = json.load(file)

    gp_yolo = GaitParameters(
        sequence_parameters=data_yolo[SEQUENCE_KEY],
        coordinates_idx=CoordinatesIdx(0, 1, 2),
    )

    print(f"{gp_yolo.get_joint_coordinate(5, 'lfoot', 'z') = }")

    with open("./data/dataset_mocap.json", "r", encoding="utf-8") as file:
        data_mocap = json.load(file)

    gp_mocap = GaitParameters(
        sequence_parameters=data_mocap[SEQUENCE_KEY],
        coordinates_idx=CoordinatesIdx(2, 0, 1),
    )

    print(f"{gp_mocap.get_joint_coordinate(5, 'lfoot', 'z') = }")
