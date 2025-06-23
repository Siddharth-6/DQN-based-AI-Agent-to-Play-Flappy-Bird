# 🐦🤖 DQN-based-AI-Agent-to-Play-Flappy-Bird
This project implements a Deep Q-Network (DQN) agent to play the classic game Flappy Bird. The AI agent learns to navigate the bird through the pipes by interacting with the game environment and improving its policy through trial and error.

## Features:
 - Deep Q-Network (DQN) implementation in PyTorch.
 - Preprocessing of game frames for efficient training.
 - Experience replay buffer for stable learning.
 - Target network to improve training stability.
 - Real-time training with episodic reward logging.
 - Checkpointing and model saving for resuming training or inference.

## No. of Steps Survived vs No. of game played graph:

![pic](https://github.com/user-attachments/assets/8934a3ed-23df-4890-82e9-09ce94ac0879)

## Video of game Played by AI Agent:

https://github.com/user-attachments/assets/9afa3d49-cda5-449a-accd-76eb8eb1499c

## Requirements
 - Python 3.x
 - PyTorch
 - Pygame
 - NumPy
 - Matplotlib
