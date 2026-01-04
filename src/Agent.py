import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.distributions.categorical import Categorical



class Agent(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        # backbone convolutional layers
        self.conv1 = nn.Conv2d(7, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        
        # Actor and critic heads
        self.actorfc1 = nn.Linear(in_features=(64*9*9), out_features=512)
        self.actorfc2 = nn.Linear(in_features=(512), out_features=81)

        self.criticfc1 = nn.Linear(in_features=(64*9*9), out_features=512)
        self.criticfc2 = nn.Linear(in_features=(512), out_features=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))

        flattened_x = torch.flatten(x, start_dim=1)

        # Policy
        p = F.relu(self.actorfc1(flattened_x))
        logits = self.actorfc2(p)

        # Value
        v = F.relu(self.criticfc1(flattened_x))
        value = self.criticfc2(v)

        return logits, value
    
    def get_training_info(self, obs: torch.Tensor, action: torch.Tensor=None):
        """
        Computes info for PPO given observations

        Args:
            x (torch.Tensor): The input observation tensor of shape (Batch_Size, 7, 9, 9).
            action (torch.Tensor, optional): If provided, the method will compute the 
                log probabilities and entropy for these specific actions. Used during 
                the PPO update phase.

        Returns:
            action (torch.Tensor): The sampled actions from the distribution.
            logprob (torch.Tensor): Log probabilities of the sampled (or provided) actions.
            entropy (torch.Tensor): The entropy of the masked distribution 
            value (torch.Tensor): The scalar value estimate from the critic head.
        """
        logits, value = self.forward(obs)
        action_mask = torch.flatten(obs[:, 3, :, :])
        masked_logits = torch.where(action_mask == 0, torch.tensor(-1e8).to(logits.device), logits)

        action_dist = Categorical(logits=masked_logits)
        
        if action is None:
            action = action_dist.sample()
        
        logprobs = action_dist.log_prob(action)
        entropy = action_dist.entropy()

        return action, logprobs, entropy, value

    @torch.no_grad()
    def get_action(self, obs: torch.Tensor, action: torch.Tensor=None):
        """
        Inference-only method to get the best move without training overhead.
        """
        self.eval()
        
        logits, _ = self.forward(obs)
        action_mask = torch.flatten(obs[:, 3, :, :])
        masked_logits = torch.where(action_mask == 0, torch.tensor(-1e8).to(logits.device), logits)

        action_dist = Categorical(logits=masked_logits)
        return action_dist.sample().item()
    
    @torch.no_grad()
    def get_value(self, obs):
        _, value = self.forward(obs)
        return value