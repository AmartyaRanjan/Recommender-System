import torch
import os
from stable_baselines3 import PPO
from app.models.saint_model import SAINT
from app.models.rl_agent import StudentRSInv

def init_models():
    print("🚀 Initializing Dummy Models for Cold Start...")
    
    # Ensure directory exists
    os.makedirs("app/ml_assets", exist_ok=True)

    # 1. SAINT Model
    print("🧠 Saving SAINT weights...")
    saint = SAINT(num_concepts=1000, num_interactions=20)
    # Save state dict
    torch.save(saint.state_dict(), "app/ml_assets/saint_weights.pt")

    # 2. RL Agent (PPO)
    print("🤖 Saving RL Agent...")
    # Need the env to initialize PPO
    env = StudentRSInv(saint)
    model = PPO("MlpPolicy", env)
    model.save("app/ml_assets/ppo_student_policy")

    print("✅ Models initialized and saved to app/ml_assets/")

if __name__ == "__main__":
    init_models()
