from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from app.services.inference_service import inference_service

router = APIRouter()

@router.websocket("/ws/{student_id}")
async def websocket_endpoint(websocket: WebSocket, student_id: str):
    # 1. AUTHENTICATION (Crucial for thousands of users)
    # Check if student_id is valid before accepting
    await websocket.accept()
    
    try:
        while True:
            # 2. RECEIVE JSON
            data = await websocket.receive_json()
            
            # 3. FAST INFERENCE
            # Ensure this is non-blocking so other users aren't delayed
            response = await inference_service.get_detailed_trace(
                student_id, 
                [e['context_id'] for e in data['telemetry_batch']],
                [e['behavior_id'] for e in data['telemetry_batch']],
                data['telemetry_batch']
            )
            
            # 4. SEND RESPONSE
            await websocket.send_json(response)
            
    except WebSocketDisconnect:
        # 5. CLEANUP
        print(f"🔌 Student {student_id} disconnected.")