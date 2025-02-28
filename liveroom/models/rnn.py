import torch
import torch.nn as nn
import torch.nn.functional as F


def safe_softmax(mat):
    return F.softmax(mat - mat.max(dim=1, keepdim=True).values, dim=1)


class SalesPredictionModel(nn.Module):
    def __init__(self,
                 tabular_input_dim,
                 llm_input_dim,
                 hidden_dim,
                 temperature=500,
                 dropout=0.2):
        super().__init__()
        self.temperature = temperature

        self.base = nn.Linear(tabular_input_dim + llm_input_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.classifier1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 8)
        )

        self.classifier2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 8)
        )

        self.out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 4)
        )

    def forward(self, x):
        x = self.base(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        logits1 = self.classifier1(x) / self.temperature
        probs1 = safe_softmax(logits1)  # [batch_size, num_classes]
        logits2 = self.classifier2(x) / self.temperature
        probs2 = safe_softmax(logits2)
        out = self.out(x)
        return torch.cat((out, probs1, probs2), dim=1)
