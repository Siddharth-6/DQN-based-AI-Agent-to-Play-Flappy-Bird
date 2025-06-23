import torch
import torch.nn as nn
import torch.optim as optim
import time
import numpy as np

from training_config import RLConfig
from model import DQNNetwork

class RLTrainer:
    """Main RL training template"""
    
    def __init__(self, state_dim, action_dim, config=None):
        self.config = config or RLConfig()
        self.gamma = self.config.DQN_CONFIG['gamma']
        self.lr = self.config.DQN_CONFIG['learning_rate']
        self.model = DQNNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
    
    def train_step(self, states, actions, rewards, next_states, dones):
        # Convert to numpy arrays first, then tensors
        states = torch.from_numpy(np.array(states)).float()
        next_states = torch.from_numpy(np.array(next_states)).float()
        actions = torch.from_numpy(np.array(actions)).float()
        rewards = torch.from_numpy(np.array(rewards)).float()
    
    # Rest of your existing code...
        # Handle batch vs single sample
        if not isinstance(states, torch.Tensor):
            states = torch.tensor(states, dtype=torch.float)
        if not isinstance(next_states, torch.Tensor):
            next_states = torch.tensor(next_states, dtype=torch.float)
        if not isinstance(actions, torch.Tensor):
            actions = torch.tensor(actions, dtype=torch.float)  # Keep as float for one-hot
        if not isinstance(rewards, torch.Tensor):
            rewards = torch.tensor(rewards, dtype=torch.float)
        
        # Ensure batch dimensions
        if len(states.shape) == 1:
            states = torch.unsqueeze(states, 0)
            next_states = torch.unsqueeze(next_states, 0)
            actions = torch.unsqueeze(actions, 0)
            rewards = torch.unsqueeze(rewards, 0)
            dones = [dones] if not isinstance(dones, (list, tuple)) else dones

        # Get current Q values
        current_q_values = self.model(states)
        
        # Get next Q values (for non-terminal states)
        with torch.no_grad():  # Don't compute gradients for target
            next_q_values = self.model(next_states)
            max_next_q_values = torch.max(next_q_values, dim=1)[0]
        
        # Compute target Q values
        target_q_values = current_q_values.clone()
        
        for idx in range(len(dones)):
            # Get action index from one-hot encoding
            action_idx = torch.argmax(actions[idx]).item()
            
            if dones[idx]:
                # Terminal state: Q_target = reward only
                target_q_values[idx][action_idx] = rewards[idx]
            else:
                # Non-terminal: Q_target = reward + gamma * max(Q_next)
                target_q_values[idx][action_idx] = rewards[idx] + self.gamma * max_next_q_values[idx]
        
        # Compute loss and update
        self.optimizer.zero_grad()
        loss = self.criterion(current_q_values, target_q_values)
        loss.backward()
        
        # Optional: Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        return loss.item()  # Return loss for monitoring
