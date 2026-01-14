import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import PPO

class StudentRSInv(gym.Env):
    """Custom Environment for Student Recommender System"""
    def __init__(self, saint_model):
        super(StudentRSInv, self).__init__()
        self.saint = saint_model
        
        # Action Space: 0: No Action, 1: Recommend Video, 2: Start Chatbot, 3: Flashcards, 4: Roadmap Shift
        self.action_space = spaces.Discrete(5)
        
        # Observation Space: The 128-d Personality Vector from SAINT
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(128,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Curriculum Learning: Initialize with a specific scenario 90% of the time
        initial_obs = self._generate_scenario()
        self.last_obs = initial_obs
        return initial_obs, {}

    def step(self, action):
        # 1. Execute Action
        # 2. Receive Reward based on the state we JUST acted on
        reward = self._get_reward_from_interaction(action, self.last_obs)
        
        # 3. Update State: Generate a new scenario for the next step
        # This mocks the student moving to a new state or a new student entering
        new_obs = self._generate_scenario() 
        self.last_obs = new_obs 
        
        done = False 
        return new_obs, reward, done, False, {}

    def _generate_scenario(self):
        """Generates clear signals for training: Flow, Struggle, Fatigue, or Random."""
        choice = np.random.rand()
        vector = np.random.normal(0, 0.5, (128,)).astype(np.float32) # Reduced noise base
        
        # INDICES MUST MATCH StudentDNADecoder in inference_service.py
        # Mastery=0, Frustration=10, Attention=20
        
        if choice < 0.3:
            # SCENARIO 1: FLOW STATE (Protect)
            # High Attention (Idx 20 > 1.0), Low Frustration (Idx 10 < 0)
            vector[20] += 2.0 
            vector[10] -= 0.5 
        
        elif choice < 0.6:
            # SCENARIO 2: STRUGGLE (Chatbot)
            # High Frustration (Idx 10 > 1.0), Low Attention (Idx 20 < 0)
            vector[10] += 2.0
            vector[20] -= 1.0
            
        elif choice < 0.9:
            # SCENARIO 3: FATIGUE (Flashcards)
            # Low Attention (Idx 20 < -1.0), Low Frustration (Idx 10 < 0)
            vector[20] -= 2.0
            vector[10] -= 0.5
            
        # else: Random Noise (10%)
        return vector

    def _get_reward_from_interaction(self, action, state):
        """
        Contextual Bandit Reward Logic (Priority Stack):
        1. Flow State Protection (High Attention) -> DO NOTHING (Action 0)
        2. Frustration/Struggle -> CHATBOT (Action 2)
        3. Fatigue/Distraction -> FLASHCARDS (Action 3)
        4. Knowledge Gap -> VIDEO (Action 1)
        """
        # Calculate 'True' Latent Traits from Vector Slices
        # INDICES CORRECTION:
        mastery_val = state[0]
        frustration_val = state[10]
        attention_val = state[20]
        
        # --- PRIORITY 1: PROTECT FLOW STATE ---
        # If student is focused (> 62%), do NOT interrupt them.
        # DNA 62% ~= Latent +0.5
        if attention_val > 0.5:
            return 1.0 if action == 0 else -1.0 # Strong penalty for interruption

        # --- PRIORITY 2: ADDRESS STRUGGLE ---
        # If frustrated (> 47%), suggest AI help.
        # DNA 47% ~= Latent -0.1
        if frustration_val > -0.1:
            return 1.0 if action == 2 else -0.5

        # --- PRIORITY 3: RE-ENGAGE FATIGUE ---
        # If zoning out (< 45%), switch to passive revision.
        # DNA 45% ~= Latent -0.2
        if attention_val < -0.2:
            return 1.0 if action == 3 else -0.5

        # --- PRIORITY 4: PLUG KNOWLEDGE GAPS ---
        # DNA 38% ~= Latent -0.5
        if mastery_val < -0.5:
            return 1.0 if action == 1 else -0.5
            
        # Default: If nothing is wrong, doing nothing is safe
        return 0.5 if action == 0 else -0.1

# --- Training Script ---
def train_recommender():
    # 1. Instantiate SAINT (Used as a feature extractor)
    saint_model = SAINT(num_concepts=1000, num_interactions=20)
    
    # 2. Create Env
    env = StudentRSInv(saint_model)
    
    # 3. Train using PPO (Stable Baselines3)
    # Using MlpPolicy because our input is a flat 128-d vector
    model = PPO("MlpPolicy", env, verbose=1, learning_rate=3e-4)
    model.learn(total_timesteps=10000)
    
    # 4. Save the brain
    model.save("app/ml_assets/ppo_student_policy")
    print("RL Agent Trained and Saved.")