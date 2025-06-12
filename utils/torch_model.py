import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomNet(nn.Module):
    def __init__(self):
        super(CustomNet, self).__init__()
        dropout = 0
        # shape (8, 2) -> reshape to (2, 8) 
        self.conv1d1 = nn.ModuleList([
            nn.Conv1d(in_channels=2, out_channels=8, kernel_size=2, padding=1) for _ in range(4)
        ])
        self.conv1d2 = nn.ModuleList([
            nn.Conv1d(in_channels=8, out_channels=1, kernel_size=2) for _ in range(4)
        ])
        
        self.bn1 = nn.BatchNorm1d(32)
        self.fc1 = nn.Linear(32, 48)
        self.dropout1 = nn.Dropout(p=dropout)
        self.bn2 = nn.BatchNorm1d(48)
        self.fc2 = nn.Linear(48, 32)
        self.dropout2 = nn.Dropout(p=dropout)
        self.bn3 = nn.BatchNorm1d(32)
        self.fc3 = nn.Linear(32, 24)

    def forward(self, x):
        # x: 4 tensors of shape (batch, 8, 2)
        conv_outs = []
        for i, xi in enumerate(x):
            xi = xi.permute(0, 2, 1)  # reshape to (batch, 2, 8) 
            # conv = self.conv1d[i](xi)     # (batch, 1, 8)
            conv = self.conv1d1[i](xi)
            conv = self.conv1d2[i](conv)
            conv = conv.squeeze(1)     # (batch, 8)
            conv_outs.append(conv)

        concat = torch.cat(conv_outs, dim=1)  # (batch, 28)

        out = self.bn1(concat)
        out = F.relu(self.bn2(self.fc1(out)))
        out = self.dropout1(out)
        out = F.relu(self.bn3(self.fc2(out)))
        out = self.dropout2(out)
        out = self.fc3(out)  # (batch, 24)
        out = out.view(-1, 8, 3)  # reshape to (batch, 8, 3)
        return out
