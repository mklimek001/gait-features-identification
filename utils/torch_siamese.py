import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


class SiameseGaitDataset(Dataset):
    def __init__(self, all_participants, participants, features_scaled_df):
        self.all_participants = all_participants
        self.participants = participants
        self.features_scaled_df = features_scaled_df
        self.data = []

        self.regenerate_pairs()

    def regenerate_pairs(self):
        self.data = []

        for participant in self.all_participants:
            mask = self.participants.isin([participant])
            participant_features = self.features_scaled_df[mask].reset_index(drop=True)

            # Create similar pairs
            for i in range(len(participant_features)):
                for j in range(i + 1, len(participant_features)):
                    self.data.append((
                        torch.tensor(participant_features.iloc[i].values, dtype=torch.float32),
                        torch.tensor(participant_features.iloc[j].values, dtype=torch.float32),
                        torch.tensor(0.0)
                    ))

                    # Dynamic negative (dissimilar) pair
                    rand_participant = random.choice(self.all_participants)
                    while rand_participant == participant:
                        rand_participant = random.choice(self.all_participants)

                    rand_mask = self.participants.isin([rand_participant])
                    rand_features = self.features_scaled_df[rand_mask].reset_index(drop=True)

                    if len(rand_features) == 0:
                        continue

                    k = random.randrange(len(participant_features))
                    m = random.randrange(len(rand_features))

                    self.data.append((
                        torch.tensor(participant_features.iloc[k].values, dtype=torch.float32),
                        torch.tensor(rand_features.iloc[m].values, dtype=torch.float32),
                        torch.tensor(1.0)
                    ))

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)
    

class SiameseNetwork(nn.Module):
    def __init__(self, input_size=24, embedding_size=10):
        super(SiameseNetwork, self).__init__()
        self.embedding = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, embedding_size)
        )

    def forward_once(self, x):
        return self.embedding(x)

    def forward(self, x1, x2):
        return self.forward_once(x1), self.forward_once(x2)
    

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, out1, out2, label):
        euclidean_distance = F.pairwise_distance(out1, out2)
        loss = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                          label * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss


def compute_similarity(x1, x2, model):
    model.eval()
    with torch.no_grad():
        out1, out2 = model(x1, x2)
        distance = F.pairwise_distance(out1, out2)
    return distance
    