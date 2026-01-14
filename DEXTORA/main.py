import uvicorn
from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.api.websocket import router as websocket_router
from app.db.redis_client import redis_client
from app.core.config import settings

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles startup and shutdown logic.
    Ensures Redis is connected before students start logging in.
    """
    # Startup: 0. Download Models from GCS (Global Brain Sync)
    from app.core.model_loader import download_models_from_gcs
    download_models_from_gcs()

    # Startup: Initialize Redis connection pool
    import asyncio
    from redis.exceptions import ConnectionError
    
    for i in range(5):
        try:
            print(f"Connecting to Redis at {settings.REDIS_HOST}:{settings.REDIS_PORT} (Attempt {i+1}/5)...")
            await redis_client.client.ping()
            print("Connected to Memorystore Redis successfully.")
            break
        except ConnectionError:
            if i == 4:
                print("❌ Could not connect to Redis after 5 attempts.")
                raise
            print("Redis not ready, waiting 2s...")
            await asyncio.sleep(2)
    
    yield
    
    # Shutdown: Clean up connections
    await redis_client.client.close()
    print("Redis connections closed.")

app = FastAPI(
    title="Personality-Driven Student RS",
    description="Real-time intervention system using SAINT and SB3",
    version="1.0.0",
    lifespan=lifespan
)

# Health Check for Google Cloud Run
@app.get("/health")
async def health_check():
    return {"status": "healthy", "users_active": "calculating..."}

# Include the WebSocket Gateway
app.include_router(websocket_router)

if __name__ == "__main__":
    # In production, Cloud Run provides the PORT environment variable
    import os
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)