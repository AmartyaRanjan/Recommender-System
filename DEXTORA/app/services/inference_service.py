import torch
import numpy as np
from app.models.saint_model import SAINT
from app.models.rl_agent import StudentRSInv
from stable_baselines3 import PPO
from app.db.redis_client import redis_client

class StudentDNADecoder:
    def __init__(self):
        self.labels = [
            "Mastery", "Retention", "Fragility", "Velocity", "Error_Density",
            "Self_Correction", "Cognitive_Load", "Logic_Bias", "Memory_Proxy", "Processing_Speed",
            "Frustration", "Boredom", "Flow_State", "Fatigue", "Feedback_Sens",
            "Resilience", "Curiosity", "Confidence", "Elasticity", "Interest",
            "Attention_Span", "Persistence", "Procrastination", "Focus_Stability", "Switch_Propensity",
            "Repetition", "Hint_Dependency", "Consistency", "Nav_Style", "Device_Fluency",
            "Visual", "Auditory", "Read_Write", "Kinetic", "Detail_Orient",
            "Holistic", "Creative", "Goal_Orient", "Collab_Intent", "Roadmap_Adherence"
        ]

    def decode(self, vector: np.ndarray, telemetry_batch: list):
        """
        Calculates 40 psychological & cognitive metrics using latent vector 
        cross-referenced with real-time telemetry heuristics.
        """
        dna = {}
        # Calculate heuristics from raw telemetry
        # Intensity: How 'active' the student is (0.0 to 1.0)
        if telemetry_batch:
            avg_intensity = np.mean([
                np.mean(list(e.get('intensity').values())) if isinstance(e.get('intensity'), dict) 
                else e.get('intensity', 0.5) 
                for e in telemetry_batch
            ])
            total_duration = sum(e.get('duration_ms', 0) for e in telemetry_batch)
            total_switches = sum(e.get('tab_switches', 0) for e in telemetry_batch)
            batch_size = len(telemetry_batch)
        else:
            avg_intensity = 0.5
            total_duration = 0
            total_switches = 0
            batch_size = 0

        for i, label in enumerate(self.labels):
            latent_val = float(vector[i % 128])
            # Sigmoid Squash to 0-100 (Base Score)
            base_score = 1 / (1 + np.exp(-latent_val)) * 100
            
            # --- 🛠️ INDUSTRY-STANDARD HEURISTICS ---
            if label == "Attention_Span":
                # If only 1 packet, start at a 'Neutral' 65% instead of 0%
                if batch_size < 2:
                    score = 65.0 
                else:
                    # Hybrid: Reward high intensity, Penalize switches
                    # Formula: Base * (1 + Intensity) / (Switches + 1)
                    score = base_score * (1 + avg_intensity) / (total_switches + 1)
            
            elif label == "Frustration":
                # Formula: High Intensity + Low Progress = High Frustration
                score = (avg_intensity * 0.7 + latent_val * 0.3) * 100
                
            elif label == "Cognitive_Load":
                # Formula: (Interactions / Duration) * Complexity_Weight
                duration_min = (total_duration / 60000) if total_duration > 0 else 1
                score = (avg_intensity / duration_min) * 50
                
            elif label == "Boredom":
                # Boredom increases if Intensity is low but Duration is high
                score = base_score
                if avg_intensity < 0.3:
                    score += 15.0 
            
            else:
                score = base_score

            # Clean formatting: 2 decimals only, capped at 100
            dna[label] = round(max(0, min(100.0, float(score))), 2)
            
        return dna

class InferenceService:
    def __init__(self):
        # 1. Load SAINT (The Context/Behavior Processor)
        self.saint = SAINT(num_concepts=1000, num_interactions=20)
        self.saint.load_state_dict(torch.load("app/ml_assets/saint_weights.pt", map_location="cpu"))
        self.saint.eval()

        # 2. Load RL Policy (The Decision Maker)
        # We only need the policy for inference, not the whole environment
        self.rl_policy = PPO.load("app/ml_assets/ppo_student_policy", device="cpu")
        
        # 3. DNA Decoder
        self.decoder = StudentDNADecoder()
        
        # 4. Trend Memory
        self.prev_dna = {} # Stores the last state for each student

    async def get_detailed_trace(self, student_id: str, context_seq: list, behavior_seq: list, telemetry_batch: list):
        """Full trace for detailed diagnostics tool."""
        print(f"\n--- 🔍 DETAILED TRACE: {student_id} ---")

        # Convert to Tensors
        context_tensor = torch.LongTensor([context_seq])
        behavior_tensor = torch.LongTensor([behavior_seq])

        with torch.no_grad():
            # STEP 1: SAINT TRANSFORMER OUTPUT
            personality_vector = self.saint(context_tensor, behavior_tensor)
            personality_np = personality_vector.cpu().numpy().flatten()
            
            # STEP 1.5: DECODE DNA with Heuristics
            dna = self.decoder.decode(personality_np, telemetry_batch)
            
            # STEP 1.6: GROUNDING: Sync the Vector with the Heuristics
            # The RL Agent sees the raw vector. We must update the vector to match the 
            # heuristic reality so the agent acts on the "Real" state, not the random Transformer state.
            # Update Key Indices (must match rl_agent.py)
            personality_np[0] = self._to_latent(dna.get("Mastery", 50))
            personality_np[10] = self._to_latent(dna.get("Frustration", 50))
            personality_np[20] = self._to_latent(dna.get("Attention_Span", 50))

            # STEP 1.8: TREND ANALYSIS
            trends = self._calculate_trends(student_id, dna)
            self.prev_dna[student_id] = dna # Update memory

            # STEP 2: CACHE SYNC
            await redis_client.set_student_vector(student_id, personality_np)

            # STEP 3: RL AGENT CALCULATION
            action, _states = self.rl_policy.predict(personality_np, deterministic=True)
            
            if isinstance(action, np.ndarray):
                action = int(action.item())
            
            command = self._map_action_to_command(action, student_id)
            
            return {
                "dna": dna,
                "trends": trends,
                "action": command,
                "vector_snippet": personality_np[:5].tolist()
            }
    
    def _to_latent(self, score):
        """Helper to convert 0-100 score back to Logit (Latent Space)."""
        p = max(0.01, min(0.99, score / 100.0))
        return np.log(p / (1 - p))

    def _calculate_trends(self, student_id, current_dna):
        if student_id not in self.prev_dna:
            return "🆕 New Session: Establishing Baseline"
        
        prev = self.prev_dna[student_id]
        
        # We track 3 key educational trends
        att_diff = current_dna['Attention_Span'] - prev['Attention_Span']
        frus_diff = current_dna['Frustration'] - prev['Frustration']
        load_diff = current_dna['Cognitive_Load'] - prev['Cognitive_Load']
        
        report = []
        if att_diff < -5: report.append(f"⚠️ Attention dropping (↓{abs(att_diff):.2f}%)")
        if frus_diff > 5: report.append(f"🔥 Frustration rising (↑{frus_diff:.2f}%)")
        if load_diff > 10: report.append(f"🧠 Cognitive Load spiking")
        
        return " | ".join(report) if report else "✅ Steady Progress"

    async def get_intervention(self, student_id: str, context_seq: list, behavior_seq: list):
        print(f"\n--- 🔍 TRACING STUDENT: {student_id} ---")
        print(f"📥 INPUTS: Context Sequence: {context_seq} | Behavior Sequence: {behavior_seq}")

        # Convert to Tensors
        context_tensor = torch.LongTensor([context_seq])
        behavior_tensor = torch.LongTensor([behavior_seq])

        with torch.no_grad():
            # STEP 1: SAINT TRANSFORMER OUTPUT
            # This is the 'Latent Personality' representing the student's current state
            personality_vector = self.saint(context_tensor, behavior_tensor)
            personality_np = personality_vector.cpu().numpy().flatten()
            
            # Display the 'Vibe' of the vector (first 5 dimensions for clarity)
            print(f"🧠 TRANSFORMER OUTPUT (Personality Vector snippet): {personality_np[:5]}...")
            print(f"📊 VECTOR MAGNITUDE (Energy): {np.linalg.norm(personality_np):.4f}")
            
            # STEP 1.5: DECODE DNA (Simple inference pass, empty batch)
            dna = self.decoder.decode(personality_np, [])
            print(f"🧬 DECODED DNA (Top 3): {list(dna.items())[:3]}")

            # STEP 2: CACHE SYNC
            await redis_client.set_student_vector(student_id, personality_np)

            # STEP 3: RL AGENT CALCULATION
            # Here we look at the 'Policy' decision
            action, _states = self.rl_policy.predict(personality_np, deterministic=True)
            
            # action needs to be a standard python int, not numpy array
            if isinstance(action, np.ndarray):
                action = int(action.item())

            # In a real setup, we'd also look at 'action_probas' to see how 
            # confident the RL agent is between Video vs. Chatbot
            print(f"🤖 RL DECISION: Action Index {action}")
            
            if action == 0:
                print("🛑 RESULT: No intervention needed (Student is in flow).")
                return None
            
            command = self._map_action_to_command(action, student_id)
            print(f"🚀 FINAL OUTPUT: {command['action']} via {command['route']}")
            print("------------------------------------------\n")
            
            return command

    def _map_action_to_command(self, action: int, student_id: str):
        """Maps RL integer actions to Flutter deep-link commands."""
        commands = {
            1: {"type": "NUDGE", "action": "SWITCH_TO_VIDEO", "route": "/video_player"},
            2: {"type": "NUDGE", "action": "SWITCH_TO_CHATBOT", "route": "/ai_tutor"},
            3: {"type": "NUDGE", "action": "SWITCH_TO_FLASHCARDS", "route": "/revision"},
            4: {"type": "ROADMAP_UPDATE", "action": "ADJUST_SCHEDULE", "route": "/roadmap"}
        }
        return commands.get(action)

# Singleton instance for the FastAPI app
inference_service = InferenceService()