import random
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from typing import Sequence


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
            nn.Conv1d(in_channels=input_size, out_channels=64, kernel_size=3, padding=1),
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
