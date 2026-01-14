import gymnasium as gym
import torch
import numpy as np
from stable_baselines3 import PPO
from app.models.saint_model import SAINT
from app.models.rl_agent import StudentRSInv # The custom environment we built earlier
import os

def train_rl_policy():
    print("🤖 Initializing Reinforcement Learning Training...")

    # 1. Load the SAINT model we just trained
    # This acts as the 'Encoder' that provides the state to the RL agent
    saint_model = SAINT(num_concepts=1000, num_interactions=20)
    try:
        saint_model.load_state_dict(torch.load("app/ml_assets/saint_weights.pt"))
        saint_model.eval()
        print("✅ Pre-trained SAINT weights loaded.")
    except FileNotFoundError:
        print("⚠️ SAINT weights not found. Using random initialization for now.")

    # 2. Setup the Environment
    # This environment simulates student responses based on their persona
    env = StudentRSInv(saint_model)

    # 3. Define the Policy (PPO)
    # We use MlpPolicy because our input is a flat 128-d vector
    # We use a slightly higher learning rate for the initial 'exploration' phase
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1, 
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99, # Focus on long-term engagement
        device="cpu"
    )

    # 4. The Learning Phase
    # We will simulate 20,000 interactions across our 100 students
    print("🚀 Training RL Agent for 20,000 timesteps...")
    model.learn(total_timesteps=20000)

    # 5. Save the 'Policy'
    model_path = "app/ml_assets/ppo_student_policy"
    model.save(model_path)
    print(f"✅ RL Policy saved to {model_path}.zip")

if __name__ == "__main__":
    train_rl_policy()