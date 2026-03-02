import random
import torch
import re
import json
import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from typing import Sequence, Mapping, Literal
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.base import TransformerMixin


DATASET_PARAMETER_FILES = {
    "mocap": "datasets/mocap/calculated_parameters_v3.json",
    "openpose": "datasets/openpose/calculated_parameters_openpose_triangulated_v2.json",
    "mediapipe": "datasets/mediapipe/calculated_parameters_v4_triangulated.json",
    "hrnet": "datasets/mmpose/calculated_parameters_hrnet_triangulated.json",
    "rtmpose": "datasets/mmpose/calculated_parameters_rtmpose_triangulated.json",
    "vitpose": "datasets/mmpose/calculated_parameters_vitpose_triangulated.json",
    "movenet_lightning": "datasets/movenet/calculated_parameters_lightning_triangulated.json",
    "movenet_thunder": "datasets/movenet/calculated_parameters_thunder_triangulated.json",
    "yolo_v26": "datasets/yolo/calculated_parameters_v3_yolo26.json",
    "yolo_v11": "datasets/yolo/calculated_parameters_v2_triangulated_v3.json",
}

DATASET_PARAMETER_FILES_BUTTERWORTH = datasets_parameters = {
    "mocap": "datasets/mocap/calculated_parameters_butterworth_v3.json",
    "openpose": "datasets/openpose/calculated_parameters_butterworth_openpose_triangulated_v2.json",
    "mediapipe": "datasets/mediapipe/calculated_parameters_butterworth_v4_triangulated.json",
    "hrnet": "datasets/mmpose/calculated_parameters_butterworth_hrnet_triangulated.json",
    "rtmpose": "datasets/mmpose/calculated_parameters_butterworth_rtmpose_triangulated.json",
    "vitpose": "datasets/mmpose/calculated_parameters_butterworth_vitpose_triangulated.json",
    "movenet_lightning": "datasets/movenet/calculated_parameters_butterworth_lightning_triangulated.json",
    "movenet_thunder": "datasets/movenet/calculated_parameters_butterworth_thunder_triangulated.json",
    "yolo_v26": "datasets/yolo/calculated_parameters_butterworth_v3_yolo26.json",
    "yolo_v11": "datasets/yolo/calculated_parameters_butterworth_v2_triangulated_v3.json",
}


SKELETON_FORMATS = {
    "mocap": ["mocap"],
    "body_25": ["openpose"],
    "mediapipe": ["mediapipe"],
    "coco": [
        "hrnet",
        "rtmpose",
        "vitpose",
        "yolo_v11",
        "yolo_v26",
        "movenet_lightning",
        "movenet_thunder",
    ],
}

PARAMETER_NAMES = [
    "legs_angles",
    "left_knee_angles",
    "right_knee_angles",
    "left_hip_angles",
    "right_hip_angles",
    "left_humerus_angles",
    "right_humerus_angles",
    "left_elbow_angles",
    "right_elbow_angles",
    "ankle_distances",
    "knee_distances",
    "elbow_distances",
    "hand_distances",
    "center_of_gravity_height_change",
    "lateral_pelvic_tilt",
    "pelvis_rotation",
]


class SiameseGaitDatasetRaw(Dataset):
    def __init__(
        self,
        selected_participants: Sequence[int],
        all_participants: Sequence[int],
        features: Sequence[np.ndarray],
        sequence_length: int = 32,  # X
        feature_dim: int = 16,
    ):
        self.selected_participants = selected_participants
        self.all_participants = all_participants
        self.features = features
        self.sequence_length = sequence_length
        self.feature_dim = feature_dim
        self.data = []

        self.regenerate_pairs()

    def _to_sequence(self, row):
        """
        Convert flat row -> (X, 32)
        """
        arr = row.values.astype("float32")
        return torch.tensor(
            arr.reshape(self.sequence_length, self.feature_dim),
            dtype=torch.float32,
        )

    def regenerate_pairs(self):
        self.data = []

        for participant in self.selected_participants:
            participant_indices = [
                idx
                for idx, tmp_ptcp in enumerate(self.all_participants)
                if tmp_ptcp == participant
            ]
            other_selected_participant_indices = [
                idx
                for idx, tmp_ptcp in enumerate(self.all_participants)
                if tmp_ptcp != participant and tmp_ptcp in self.selected_participants
            ]

            for i in range(len(participant_indices)):
                for j in range(i + 1, len(participant_indices)):

                    # positive pair
                    self.data.append(
                        (
                            self.features[i].astype(np.float32).T,
                            self.features[j].astype(np.float32).T,
                            torch.tensor(0.0),
                        )
                    )

                    rand_ptcp_idx = random.choice(participant_indices)
                    rand_other_ptcp_idx = random.choice(
                        other_selected_participant_indices
                    )

                    # negative pair
                    self.data.append(
                        (
                            self.features[rand_ptcp_idx].astype(np.float32).T,
                            self.features[rand_other_ptcp_idx].astype(np.float32).T,
                            torch.tensor(1.0),
                        )
                    )

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)


def get_person_from_seq_key(sequence_key: str) -> int:
    match = re.search(r"p(\d+)s", sequence_key)
    if match:
        person_id = int(match.group(1))
        return person_id
    else:
        raise Exception(
            f"Person identifier cannot be extracted from provided key ({sequence_key})"
        )


def get_scaler(
    selected_datasets: Sequence[str] | str | None,
    training_set: Sequence[int],
    use_butterworth: bool = False,
) -> TransformerMixin:
    """
    Function prepare standard scaler on training set,
    It can be later passed to SiameseGaitDatasetRawMultipleDatasets
    """

    if isinstance(selected_datasets, str):
        selected_datasets = [selected_datasets]
    if selected_datasets == None:
        selected_datasets = list(DATASET_PARAMETER_FILES.keys())

    param_ds = {}

    if use_butterworth:
        for name, file_path in DATASET_PARAMETER_FILES_BUTTERWORTH.items():
            with open(file_path, "r", encoding="utf-8") as file:
                param_ds[name] = json.load(file)
    else:
        for name, file_path in DATASET_PARAMETER_FILES.items():
            with open(file_path, "r", encoding="utf-8") as file:
                param_ds[name] = json.load(file)

    combined_test_features = {parameter_name: [] for parameter_name in PARAMETER_NAMES}

    for dataset_type, sequence_parameters in param_ds.items():
        if dataset_type in selected_datasets:
            for sequence_key, parameters in sequence_parameters.items():
                person_id = get_person_from_seq_key(sequence_key)
                if person_id in training_set:
                    for parameter_name in PARAMETER_NAMES:
                        combined_test_features[parameter_name] += parameters[
                            parameter_name
                        ]

    features_df = pd.DataFrame(combined_test_features)

    scaler = StandardScaler()
    scaler.fit(features_df.values)

    return scaler


class SiameseGaitDatasetRawMultipleDatasets(Dataset):
    def __init__(
        self,
        selected_participants: Sequence[int],
        selected_datasets: Sequence[str] | str | None,
        scaler: TransformerMixin | None = None,
        diff_dataset_usage: Literal[
            "MIX_ALL", "SKELETON_TYPE", "DATASET_TYPE"
        ] = "MIX_ALL",
        sequence_length: int = 32,
        feature_dim: int = 16,
        use_butterworth_smoothed: bool = False,
    ):
        self.selected_participants = selected_participants
        if isinstance(selected_datasets, str):
            self.selected_datasets = [selected_datasets]
        elif isinstance(selected_datasets, Sequence):
            self.selected_datasets = selected_datasets
        else:
            self.selected_datasets = list(DATASET_PARAMETER_FILES.keys())

        self.scaler = scaler
        self.use_butterworth_smoothed = use_butterworth_smoothed
        self.diff_dataset_usage = diff_dataset_usage
        self.features_by_dataset = self._prepare_features_list()
        self.sequence_length = sequence_length
        self.feature_dim = feature_dim
        self.data = []

        self.regenerate_pairs()

    def _prepare_features_list(
        self,
    ) -> Mapping[str, Mapping[int, np.array]]:
        param_ds = {}

        if self.use_butterworth_smoothed:
            for name, file_path in DATASET_PARAMETER_FILES_BUTTERWORTH.items():
                with open(file_path, "r", encoding="utf-8") as file:
                    param_ds[name] = json.load(file)

        else:
            for name, file_path in DATASET_PARAMETER_FILES.items():
                with open(file_path, "r", encoding="utf-8") as file:
                    param_ds[name] = json.load(file)

        return {
            name: self._get_steps_features(features)
            for name, features in param_ds.items()
        }

    def _find_local_maxima(
        self, data: Sequence[float], window_size: int = 7
    ) -> Sequence[int]:
        local_maxima_indices = []

        for i in range(window_size, len(data) - window_size):
            window_prev = data[i - window_size : i]
            window_next = data[i + 1 : i + window_size + 1]
            current = data[i]

            if current > max(window_prev) and current > max(window_next):
                local_maxima_indices.append(i)

        return local_maxima_indices

    def _get_person_from_seq_key(self, sequence_key: str) -> int:
        match = re.search(r"p(\d+)s", sequence_key)
        if match:
            person_id = int(match.group(1))
            return person_id
        else:
            raise Exception(
                f"Person identifier cannot be extracted from provided key ({sequence_key})"
            )

    def _get_steps_features(self, dataset):
        extracted_paramters = {i: [] for i in range(1, 33)}
        for sequence_key, sequence_parameters in dataset.items():
            person_id = self._get_person_from_seq_key(sequence_key)
            maxima = self._find_local_maxima(sequence_parameters["ankle_distances"])
            if len(maxima) % 2:
                stride_centers = [
                    int(maxima[2 * i] + 0.5 * (maxima[2 * i + 2] - maxima[2 * i]))
                    for i in range(len(maxima) // 2)
                ]
            else:
                stride_centers = [
                    int(maxima[2 * i] + 0.5 * (maxima[2 * i + 2] - maxima[2 * i]))
                    for i in range(len(maxima) // 2 - 1)
                ]
            for c_idx in stride_centers:
                stride_matrix = np.array(
                    [
                        sequence_parameters[parameter][c_idx - 16 : c_idx + 16]
                        for parameter in PARAMETER_NAMES
                    ]
                )

                if self.scaler:
                    extracted_paramters[person_id].append(
                        np.array(self.scaler.transform(stride_matrix.T)).astype(
                            np.float32
                        )
                    )
                else:
                    extracted_paramters[person_id].append(
                        stride_matrix.astype(np.float32).T
                    )

        return extracted_paramters

    def _to_sequence(self, row):
        """
        Convert flat row -> (X, 32)
        """
        arr = row.values.astype("float32")
        return torch.tensor(
            arr.reshape(self.sequence_length, self.feature_dim),
            dtype=torch.float32,
        )

    def regenerate_pairs(self):
        self.data = []

        if self.diff_dataset_usage == "DATASET_TYPE":
            for dataset in self.selected_datasets:
                for positive_participant in self.selected_participants:
                    participant_matrices = self.features_by_dataset[dataset][
                        positive_participant
                    ]
                    other_selected_participants_matrices = []

                    for tmp_ptcp in self.selected_participants:
                        if tmp_ptcp != positive_participant:
                            other_selected_participants_matrices += (
                                self.features_by_dataset[dataset][tmp_ptcp]
                            )

                    for i in range(len(participant_matrices)):
                        for j in range(i + 1, len(participant_matrices)):

                            # positive pair
                            self.data.append(
                                (
                                    participant_matrices[i],
                                    participant_matrices[j],
                                    torch.tensor(0.0),
                                )
                            )

                            rand_positive_ptcp_sample = random.choice(
                                participant_matrices
                            )
                            rand_negative_ptcp_sample = random.choice(
                                other_selected_participants_matrices
                            )

                            # negative pair
                            self.data.append(
                                (
                                    rand_positive_ptcp_sample,
                                    rand_negative_ptcp_sample,
                                    torch.tensor(1.0),
                                )
                            )

        if self.diff_dataset_usage == "MIX_ALL":
            for positive_participant in self.selected_participants:
                participant_matrices = []
                other_selected_participants_matrices = []
                for dataset in self.selected_datasets:
                    for tmp_ptcp in self.selected_participants:
                        if tmp_ptcp != positive_participant:
                            other_selected_participants_matrices += (
                                self.features_by_dataset[dataset][tmp_ptcp]
                            )
                        else:
                            participant_matrices += self.features_by_dataset[dataset][
                                tmp_ptcp
                            ]

                for i in range(len(participant_matrices)):
                    for j in range(i + 1, len(participant_matrices)):

                        # positive pair
                        self.data.append(
                            (
                                participant_matrices[i],
                                participant_matrices[j],
                                torch.tensor(0.0),
                            )
                        )

                        rand_positive_ptcp_sample = random.choice(participant_matrices)
                        rand_negative_ptcp_sample = random.choice(
                            other_selected_participants_matrices
                        )

                        # negative pair
                        self.data.append(
                            (
                                rand_positive_ptcp_sample,
                                rand_negative_ptcp_sample,
                                torch.tensor(1.0),
                            )
                        )

        if self.diff_dataset_usage == "SKELETON_TYPE":
            for ds_with_common_skeleton in SKELETON_FORMATS.values():
                selected_ds_with_common_skeleton = [
                    ds for ds in ds_with_common_skeleton if ds in self.selected_datasets
                ]
                for positive_participant in self.selected_participants:
                    participant_matrices = []
                    other_selected_participants_matrices = []
                    for dataset in selected_ds_with_common_skeleton:
                        for tmp_ptcp in self.selected_participants:
                            if tmp_ptcp != positive_participant:
                                other_selected_participants_matrices += (
                                    self.features_by_dataset[dataset][tmp_ptcp]
                                )
                            else:
                                participant_matrices += self.features_by_dataset[
                                    dataset
                                ][tmp_ptcp]

                    for i in range(len(participant_matrices)):
                        for j in range(i + 1, len(participant_matrices)):

                            # positive pair
                            self.data.append(
                                (
                                    participant_matrices[i],
                                    participant_matrices[j],
                                    torch.tensor(0.0),
                                )
                            )

                            rand_positive_ptcp_sample = random.choice(
                                participant_matrices
                            )
                            rand_negative_ptcp_sample = random.choice(
                                other_selected_participants_matrices
                            )

                            # negative pair
                            self.data.append(
                                (
                                    rand_positive_ptcp_sample.T,
                                    rand_negative_ptcp_sample.T,
                                    torch.tensor(1.0),
                                )
                            )

    def get_raw_sequences(self, participants_id: Sequence[int], dataset_type: str):
        selected_ptcp_ids = []
        selected_ptcp_params = []

        if self.use_butterworth_smoothed:
            file_path = DATASET_PARAMETER_FILES_BUTTERWORTH[dataset_type]
        else:
            file_path = DATASET_PARAMETER_FILES[dataset_type]
        with open(file_path, "r", encoding="utf-8") as file:
            dataset_parameters = json.load(file)

        dataset_features = self._get_steps_features(dataset_parameters)

        for participant in participants_id:
            features_for_ptcp = dataset_features[participant]
            selected_ptcp_params += features_for_ptcp
            selected_ptcp_ids += [participant for _ in range(len(features_for_ptcp))]

        return selected_ptcp_ids, selected_ptcp_params

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)


class SiameseNetworkLSTM(nn.Module):
    def __init__(self, input_size=16, embedding_size=10):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
        )

        self.fc = nn.Linear(64, embedding_size)

    def forward_once(self, x):
        # x: (batch, 32, input_size)
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1])

    def forward(self, x1, x2):
        return self.forward_once(x1), self.forward_once(x2)


class SiameseNetworkConv1D(nn.Module):
    def __init__(self, input_size=16, embedding_size=10):
        super().__init__()

        self.encoder = nn.Sequential(
            # (batch, input_size, 32)
            nn.Conv1d(
                in_channels=input_size, out_channels=64, kernel_size=3, padding=1
            ),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),  # (batch, 256, 1)
        )

        self.fc = nn.Linear(256, embedding_size)

    def forward_once(self, x):
        # x: (batch, 32, input_size)
        x = x.transpose(1, 2)  # -> (batch, input_size, 32)
        x = self.encoder(x).squeeze(-1)
        return self.fc(x)

    def forward(self, x1, x2):
        return self.forward_once(x1), self.forward_once(x2)


class SiameseNetworkConv1DwithBactchNorm(nn.Module):
    def __init__(self, input_size=16, embedding_size=128):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv1d(
                in_channels=input_size, out_channels=64, kernel_size=3, padding=1
            ),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )

        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, embedding_size),
        )

    def forward_once(self, x):
        if x.dim() == 3 and x.shape[1] != self.encoder[0].in_channels:
            x = x.transpose(1, 2)

        x = self.encoder(x)
        x = self.fc(x)
        return x

    def forward(self, x1, x2):
        return self.forward_once(x1), self.forward_once(x2)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=32):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        # x: [batch, 32, 16]
        return x + self.pe[:, : x.size(1)]


class SiameseNetworkTransformer(nn.Module):
    def __init__(self, input_size=16, embedding_size=10, nhead=4, num_layers=2):
        super().__init__()

        self.pos_encoder = PositionalEncoding(input_size)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_size,
            nhead=nhead,
            dim_feedforward=64,
            dropout=0.3,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc = nn.Linear(input_size, embedding_size)

    def forward_once(self, x):
        x = self.pos_encoder(x)
        x = self.transformer(x)
        # Global Average Pooling to cmpress 32 steps into 1 vector
        x = x.mean(dim=1)

        return self.fc(x)

    def forward(self, x1, x2):
        return self.forward_once(x1), self.forward_once(x2)


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss function used in siamese neural network.
    """

    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, out1, out2, label):
        euclidean_distance = F.pairwise_distance(out1, out2)
        loss = torch.mean(
            (1 - label) * torch.pow(euclidean_distance, 2)
            + label
            * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        )
        return loss


def compute_similarity(x1, x2, model):
    model.eval()
    device = next(model.parameters()).device

    with torch.no_grad():
        if isinstance(x1, np.ndarray):
            x1 = torch.from_numpy(x1)
        if isinstance(x2, np.ndarray):
            x2 = torch.from_numpy(x2)

        x1 = x1.float().unsqueeze(0).to(device)
        x2 = x2.float().unsqueeze(0).to(device)

        out1, out2 = model(x1, x2)
        distance = F.pairwise_distance(out1, out2)

    return distance
