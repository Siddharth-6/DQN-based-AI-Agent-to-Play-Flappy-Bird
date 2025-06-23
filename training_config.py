class RLConfig:
    """Configuration template for RL algorithms"""
    
    # Environment settings
    ENV_CONFIG = {
        'state_type': 'vector',  # 'vector' or 'image'
        'render_mode': 'rgb_array',
        'max_episode_steps': 10000,
    }
    
    # DQN Configuration Template
    DQN_CONFIG = {
        # TODO: Set hyperparameters for DQN
        'learning_rate': 0.001,
        'gamma': 0.9,  # Discount factor
        'epsilon_start': None,  # Initial exploration rate
        'epsilon_end': None,   # Final exploration rate
        'epsilon_decay': None, # Exploration decay steps
        'memory_size': None,   # Replay buffer size
        'batch_size': None,    # Training batch size
        'target_update': None, # Target network update frequency
        'learning_starts': None, # When to start learning
        'train_freq': None,    # Training frequency
    }
    
    # Training settings
    TRAINING_CONFIG = {
        'total_timesteps': 1000000,
        'eval_freq': 10000,
        'eval_episodes': 10,
        'save_freq': 50000,
        'log_freq': 1000,
    }
