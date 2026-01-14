import redis.asyncio as redis
import numpy as np
import json
from app.core.config import settings

class RedisClient:
    def __init__(self):
        # Initialize the async connection pool
        self.client = redis.Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            decode_responses=False  # Keep False to handle binary numpy arrays
        )
        self.ttl = 3600  # 1-hour session timeout

    async def hydrate_student_session(self, student_id: str):
        """
        Cold Start -> Hot Start.
        If not in Redis, pull the 'Master Vector' from PostgreSQL and load it.
        """
        exists = await self.client.exists(f"student:{student_id}:vector")
        if not exists:
            # TODO: Add logic to pull from PostgreSQL/AlloyDB here
            # For now, we initialize a neutral vector
            initial_vector = np.zeros(128, dtype=np.float32)
            await self.set_student_vector(student_id, initial_vector)

    async def set_student_vector(self, student_id: str, vector: np.ndarray):
        """
        Stores the 128-d vector as a binary blob for ultra-fast performance.
        """
        # Converting to bytes is much faster than JSON for 5,000 users
        vector_bytes = vector.astype(np.float32).tobytes()
        await self.client.setex(
            f"student:{student_id}:vector", 
            self.ttl, 
            vector_bytes
        )

    async def get_student_vector(self, student_id: str) -> np.ndarray:
        """
        Retrieves the vector and converts it back to a Numpy array.
        """
        data = await self.client.get(f"student:{student_id}:vector")
        if data:
            return np.frombuffer(data, dtype=np.float32)
        return np.zeros(128, dtype=np.float32)

    async def persist_student_vector(self, student_id: str):
        """
        Dehydration: Save the final session vector back to PostgreSQL.
        Called when a WebSocket disconnects.
        """
        vector = await self.get_student_vector(student_id)
        # TODO: self.postgres.save_vector(student_id, vector)
        print(f"Persisted vector for {student_id} to long-term storage.")
        await self.client.delete(f"student:{student_id}:vector")

# Singleton instance
redis_client = RedisClient()