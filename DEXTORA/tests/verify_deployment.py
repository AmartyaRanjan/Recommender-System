import asyncio
import websockets
import json

async def verify_flow():
    student_id = "STU_1001"
    # Matches the single endpoint: /ws/{student_id}
    uri = f"ws://localhost:8080/ws/{student_id}"

    print(f"🔌 Connecting to socket: {uri}")
    async with websockets.connect(uri) as ws:
        print("✅ Connected")
        
        # Prepare Payload
        payload = {
            "student_id": student_id,
            "telemetry_batch": [
                {
                    "context_id": 101,
                    "behavior_id": 2,
                    "timestamp": "2026-01-14T10:40:00Z",
                    "duration_ms": 45000,
                    "intensity": {"scroll_velocity": 0.4},
                    "engagement_metrics": {"tab_switches": 0},
                    "performance_data": {},
                    "metadata": {}
                }
            ]
        }
        
        print(f"📤 Sending Payload...")
        await ws.send(json.dumps(payload))
        
        # In Request-Response, we expect the result immediately on the same socket
        print("⏳ Waiting for Response...")
        response = await ws.recv()
        print(f"🎉 RECEIVED RESPONSE: {response}")

if __name__ == "__main__":
    asyncio.run(verify_flow())
