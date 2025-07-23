# agents/critic.py
import torch
import torch.nn as nn

class CentralizedCritic(nn.Module):
    def __init__(self, total_obs, total_actions):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(total_obs + total_actions, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)  # joint Q-value
        )

    def forward(self, obs_cat, act_cat):
        x = torch.cat([obs_cat, act_cat], dim=-1)
        return self.fc(x)
