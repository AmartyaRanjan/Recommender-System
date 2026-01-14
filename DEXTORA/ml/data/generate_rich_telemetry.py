import json
import random

def generate_telemetry(num_packets=100):
    phases = []
    
    # Define state templates to ensure we hit all RL boundaries
    state_types = ["Flow", "Struggle", "Fatigue", "Neutral", "Gap"]
    
    current_student = "STU_1001" 
    
    for i in range(num_packets):
        # Pick a state type (weighted slightly towards Flow/Neutral to simulate real work)
        state = random.choices(state_types, weights=[0.3, 0.2, 0.2, 0.2, 0.1])[0]
        
        batch = []
        num_items = random.randint(1, 3) # Items per batch
        
        desc = f"Packet {i+1}: {state} Simulation"
        
        # Base values depending on state
        if state == "Flow":
            # High Intensity, Low Switching
            base_intensity = 0.85
            base_switches = 0
            base_duration = 60000
        elif state == "Struggle":
            # High Intensity, High Switching
            base_intensity = 0.9
            base_switches = 5
            base_duration = 30000
        elif state == "Fatigue":
            # Low Intensity
            base_intensity = 0.2
            base_switches = 1
            base_duration = 120000
        elif state == "Gap":
            # Low Mastery (hard to simulate with just telemetry, but we simulate 'giving up' behavior)
            base_intensity = 0.4
            base_switches = 8 # Searching for answers
            base_duration = 45000
        else: # Neutral
            base_intensity = 0.6
            base_switches = 2
            base_duration = 60000

        for _ in range(num_items):
            # Add some jitter
            intensity = min(0.99, max(0.01, base_intensity + random.uniform(-0.1, 0.1)))
            switches = max(0, int(base_switches + random.uniform(-2, 3)))
            duration = int(base_duration * random.uniform(0.8, 1.2))
            
            batch.append({
                "context_id": random.randint(100, 105),
                "behavior_id": random.randint(0, 5),
                "duration_ms": duration,
                "intensity": intensity,
                "tab_switches": switches
            })
            
        phases.append({
            "student_id": current_student,
            "desc": desc,
            "telemetry_batch": batch
        })

    with open("ml/data/samples/rich_telemetry.json", "w") as f:
        json.dump(phases, f, indent=4)
        print(f"✅ Generated {num_packets} telemetry packets in 'ml/data/samples/rich_telemetry.json'")

if __name__ == "__main__":
    generate_telemetry(100)
