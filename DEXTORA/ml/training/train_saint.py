import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import asyncio
from app.db.postgres_client import postgres_client, StudentProfile
from app.models.saint_model import SAINT
import numpy as np

# 1. Dataset Loader: Converts Postgres rows into Tensors
class TelemetryDataset(Dataset):
    def __init__(self, data, seq_len=10):
        self.data = data
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # In a real app, you'd pull full history logs. 
        # Here we simulate sequences from the seeded vectors + random noise.
        context = torch.randint(0, 1000, (self.seq_len,))
        behavior = torch.randint(0, 20, (self.seq_len,))
        
        # Target: Predict the 'next' concept (Self-Supervised Learning)
        target = torch.randint(0, 1000, (1,)) 
        return context, behavior, target

async def train_saint():
    print("🧠 Starting SAINT Training...")
    
    # Load all students from Postgres
    # The helper manages its own session, so we just await it directly
    profiles = await postgres_client.get_all_profiles()
        
    dataset = TelemetryDataset(profiles)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    # Initialize Model, Loss, and Optimizer
    model = SAINT(num_concepts=1000, num_interactions=20)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # --- The Training Loop ---
    model.train()
    for epoch in range(5): # Start with 5 epochs for the 100 students
        total_loss = 0
        for context, behavior, target in loader:
            optimizer.zero_grad()
            
            # Forward Pass: Get the Personality Vector
            # For training, we add a 'Head' to predict the next concept
            output_vector = model(context, behavior)
            
            # Temporary classifier to ground the embeddings
            prediction = nn.Linear(128, 1000)(output_vector) 
            
            loss = criterion(prediction, target.squeeze())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1} | Loss: {total_loss/len(loader):.4f}")

    # Save the 'Weights' (The knowledge)
    torch.save(model.state_dict(), "app/ml_assets/saint_weights.pt")
    print("✅ SAINT Weights saved to app/ml_assets/saint_weights.pt")

if __name__ == "__main__":
    import sys
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(train_saint())