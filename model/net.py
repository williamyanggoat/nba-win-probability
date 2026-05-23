import torch
import torch.nn as nn

FEATURE_COLS = [
    "score_diff", "secs_left", "time_elapsed_norm",
    "home_fouls", "away_fouls", "foul_diff",
    "home_timeouts", "away_timeouts",
    "possession_home", "is_overtime", "period",
]

class WinProbNet(nn.Module):
    def __init__(self, input_dim=len(FEATURE_COLS), hidden=128, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),           # outputs win probability in [0, 1]
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)