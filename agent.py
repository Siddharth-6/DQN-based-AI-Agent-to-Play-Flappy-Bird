import torch
import torch.nn as nn
import torch.optim as optim
from model import DQNNetwork
from train_rl import RLTrainer
from collections import deque
from game_runner import FlappyBirdGame
from config import Config
import matplotlib.pyplot as plot
import math
import random
import json
import numpy as np
import os
from helper import plot, plot_survival


MAX_MEMORY = 50000  # Increased from 10000
BATCH_SIZE = 32     # Reduced from 1000 for more frequent training
MIN_REPLAY_SIZE = 1000  # Wait before starting training


class ReplayBuffer:
    def __init__(self):
        self.memory = deque(maxlen=MAX_MEMORY)

    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = list(self.memory)

        return(zip(*mini_sample))

    def __len__(self):
        return len(self.memory)
    
    def save_experiences(self, filename="replay_buffer.json"):
        """Save replay buffer experiences to JSON file"""
        experiences = []
        for state, action, reward, next_state, done in self.memory:
            experiences.append({
                'state': state.tolist() if isinstance(state, np.ndarray) else state,
                'action': action,
                'reward': float(reward),
                'next_state': next_state.tolist() if isinstance(next_state, np.ndarray) else next_state,
                'done': bool(done)
            })
        
        with open(filename, 'w') as f:
            json.dump(experiences, f, indent=2)
        print(f"Saved {len(experiences)} experiences to {filename}")
    
    def load_experiences(self, filename="replay_buffer.json"):
        """Load replay buffer experiences from JSON file"""
        try:
            with open(filename, 'r') as f:
                experiences = json.load(f)
            
            self.memory.clear()
            for exp in experiences:
                self.memory.append((
                    np.array(exp['state']),
                    exp['action'],
                    exp['reward'],
                    np.array(exp['next_state']),
                    exp['done']
                ))
            print(f"Loaded {len(experiences)} experiences from {filename}")
            return True
        except FileNotFoundError:
            print(f"No replay buffer file found at {filename}")
            return False
        except Exception as e:
            print(f"Error loading replay buffer: {e}")
            return False


class DQNAgent:
    """DQN Agent with checkpoint functionality"""

    def __init__(self,config=None):
        self.state_dim = 4
        self.action_dim = 2
        self.config = config
        self.trainer = RLTrainer(self.state_dim,self.action_dim,self.config)
        self.replaybuffer = ReplayBuffer()
        self.n_games = 0
        
        # Enhanced epsilon strategy with exponential decay
        self.eps_start = 0.1
        self.eps_end = 0.1
        self.eps_decay = 0.9985  # Exponential decay rate
        self.epsilon = self.eps_start
        
        # Training parameters
        self.train_freq = 4
        self.step_count = 0

    def _get_epsilon(self):
        # Use exponential decay instead of games-based decay
        return self.epsilon

    def update_epsilon(self):
        """Update epsilon with exponential decay"""
        self.epsilon = max(self.eps_end, self.epsilon * self.eps_decay)

    def select_action(self, state):
        '''
        state: a flat list or array of features
        returns: one-hot [0,1] or [1,0]
        '''
        eps = self._get_epsilon()
        final_move = [0, 0]

        #if random.random() < eps:
            # explore
            #move = random.randint(0, 1)
        #else:
            # exploit
        self.trainer.model.eval()  # Set model to evaluation mode
        state_v = torch.tensor(state, dtype=torch.float32)
            # add batch dim if your model expects it:
        if state_v.dim() == 1:
            state_v = state_v.unsqueeze(0)
        with torch.no_grad():
            q_values = self.trainer.model(state_v)
        self.trainer.model.train()  # Set model back to training mode
            # q_values shape: [1, action_size] → pick the max
        move = torch.argmax(q_values, dim=1).item()
        final_move[move] = 1
        return final_move
    
    def train_if_ready(self):
        """Train the agent if enough experiences are available"""
        if len(self.replaybuffer) < MIN_REPLAY_SIZE:
            return
        
        if self.step_count % self.train_freq == 0:
            states, actions, rewards, next_states, dones = self.replaybuffer.sample()
            self.trainer.train_step(states, actions, rewards, next_states, dones)
    
    def save_checkpoint(self, filename="checkpoint.json"):
        """Save complete training state"""
        checkpoint = {
            'n_games': self.n_games,
            'epsilon': self.epsilon,
            'step_count': self.step_count,
            'eps_start': self.eps_start,
            'eps_end': self.eps_end,
            'eps_decay': self.eps_decay,
            'train_freq': self.train_freq,
            'buffer_size': len(self.replaybuffer),
            'model_path': f'model_checkpoint_{self.n_games}.pth'
        }
        
        # Save agent state
        with open(filename, 'w') as f:
            json.dump(checkpoint, f, indent=2)
        
        # Save model weights
        self.trainer.model.save(checkpoint['model_path'])
        
        # Save replay buffer
        buffer_filename = filename.replace('.json', '_buffer.json')
        self.replaybuffer.save_experiences(buffer_filename)
        
        print(f"Checkpoint saved: {filename}")
    
    def load_checkpoint(self, filename="checkpoint.json"):
        """Load complete training state"""
        try:
            # Load agent state
            with open(filename, 'r') as f:
                checkpoint = json.load(f)
            
            self.n_games = checkpoint['n_games']
            self.epsilon = checkpoint['epsilon']
            self.step_count = checkpoint['step_count']
            self.eps_start = checkpoint.get('eps_start', self.eps_start)
            self.eps_end = checkpoint.get('eps_end', self.eps_end)
            self.eps_decay = checkpoint.get('eps_decay', self.eps_decay)
            self.train_freq = checkpoint.get('train_freq', self.train_freq)
            
            # Load model weights
            if 'model_path' in checkpoint:
                try:
                    self.trainer.model.load_state_dict(torch.load(checkpoint['model_path']))
                    print(f"Model weights loaded from {checkpoint['model_path']}")
                except:
                    print(f"Could not load model weights from {checkpoint['model_path']}")
            
            # Load replay buffer
            buffer_filename = filename.replace('.json', '_buffer.json')
            self.replaybuffer.load_experiences(buffer_filename)
            
            print(f"Checkpoint loaded: {filename}")
            print(f"Resumed from game {self.n_games}, epsilon: {self.epsilon:.3f}")
            return True
            
        except FileNotFoundError:
            print(f"No checkpoint found at {filename}")
            return False
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            return False


plot_scores = []
plot_mean_scores = []
plot_survival_times = []
plot_mean_survival_times = []
total_score = 0
total_survival_time = 0

def train():
    global plot_scores, plot_mean_scores, plot_survival_times, plot_mean_survival_times
    global total_score, total_survival_time
    
    agent = DQNAgent()
    game = FlappyBirdGame()
    
    # Define checkpoint file path
    checkpoint_file = "training_checkpoint.json"
    MIN_REPLAY_SIZE = 1000  # Define this constant
    
    # Initialize survival tracking
    current_episode_steps = 0
    
    # Load existing checkpoint if available
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'r') as f:
                checkpoint_data = json.load(f)
                print(f"Loaded checkpoint from game {checkpoint_data.get('n_games', 0)}")
        except:
            print("Could not load checkpoint, starting fresh")
    
    while True:
        old_state = game.get_state_array()
        action = agent.select_action(old_state)
        
        # Convert one-hot to action index
        action_idx = 1 if action[1] == 1 else 0
        new_state, reward, done = game.step(action_idx)
        
        agent.replaybuffer.push(old_state, action, reward, new_state, done)
        agent.step_count += 1
        current_episode_steps += 1  # Track survival time
        
        # Train if ready
        agent.train_if_ready()
               
        # Update game display
        #if agent.n_games % 10 == 0:
        game.draw()
        
        if done:
            agent.n_games += 1
            agent.update_epsilon()
            
            # Store survival time for this episode
            plot_survival_times.append(current_episode_steps)
            total_survival_time += current_episode_steps
            mean_survival_time = total_survival_time / agent.n_games
            plot_mean_survival_times.append(mean_survival_time)
            
            # Additional training on game over
            if len(agent.replaybuffer) > MIN_REPLAY_SIZE:
                states, actions, rewards, next_states, dones = agent.replaybuffer.sample()
                agent.trainer.train_step(states, actions, rewards, next_states, dones)
            
            # Save model on new high score
            if game.score >= game.high_score:
                agent.trainer.model.save(f'model_score_{game.score}.pth')
                
            print('Game', agent.n_games, 'Score', game.score, 
                  f'Survived: {current_episode_steps} steps', 
                  f'Epsilon: {agent.epsilon:.3f}')
            
            # Update score tracking
            plot_scores.append(game.score)
            total_score += game.score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            
            # Reset episode step counter
            current_episode_steps = 0
            
            # Auto-save checkpoint every 50 games
            if agent.n_games % 50 == 0:
                # Check if agent has save_checkpoint method, if not create basic save
                try:
                    agent.save_checkpoint(checkpoint_file)
                except AttributeError:
                    # Basic checkpoint save if method doesn't exist
                    checkpoint_data = {
                        'n_games': agent.n_games,
                        'epsilon': getattr(agent, 'epsilon', 0.1),
                        'step_count': getattr(agent, 'step_count', 0)
                    }
                    with open(checkpoint_file, 'w') as f:
                        json.dump(checkpoint_data, f, indent=2)
                
                # Save training statistics
                stats = {
                    'scores': plot_scores,
                    'mean_scores': plot_mean_scores,
                    'survival_times': plot_survival_times,
                    'mean_survival_times': plot_mean_survival_times,
                    'total_score': total_score,
                    'total_survival_time': total_survival_time,
                    'last_updated': agent.n_games
                }
                stats_file = checkpoint_file.replace('.json', '_stats.json')
                with open(stats_file, 'w') as f:
                    json.dump(stats, f, indent=2)
                print(f"Auto-saved checkpoint at game {agent.n_games}")
            
            # Plot every 10 games
            if agent.n_games % 5 == 0:
                print("Epsilon:", getattr(agent, 'epsilon', 'N/A'))
                avg_score = sum(plot_scores[-10:]) / min(10, len(plot_scores))
                avg_survival = sum(plot_survival_times[-10:]) / min(10, len(plot_survival_times))
                print(f"Average score (last 10 games): {avg_score:.2f}")
                print(f"Average survival time (last 10 games): {avg_survival:.1f} steps")
                
                # Plot scores
                #plot(plot_scores, plot_mean_scores)
                #plot_survival(plot_survival_times,plot_mean_survival_times)
            game.restart_game()
train()