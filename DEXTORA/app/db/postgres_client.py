from sqlalchemy import Column, String, Integer, DateTime, func
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from pgvector.sqlalchemy import Vector  # You must run: pip install pgvector
from app.core.config import settings
import numpy as np

Base = declarative_base()

class StudentProfile(Base):
    __tablename__ = "student_profiles"
    
    student_id = Column(String, primary_key=True, index=True)
    grade = Column(Integer)
    curriculum = Column(String)
    
    # The 128-d SAINT Personality Vector
    personality_vector = Column(Vector(128)) 
    
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

class PostgresClient:
    def __init__(self):
        self.engine = create_async_engine(settings.DATABASE_URL, echo=False)
        self.async_session = sessionmaker(
            self.engine, class_=AsyncSession, expire_on_commit=False
        )

    async def get_student_profile(self, student_id: str):
        async with self.async_session() as session:
            profile = await session.get(StudentProfile, student_id)
            return profile

    async def save_personality_vector(self, student_id: str, vector: np.ndarray):
        """
        Persists the final vector from Redis to PostgreSQL.
        """
        async with self.async_session() as session:
            async with session.begin():
                profile = await session.get(StudentProfile, student_id)
                if profile:
                    profile.personality_vector = vector.tolist()
                else:
                    # Create a new profile if they don't exist
                    new_profile = StudentProfile(
                        student_id=student_id, 
                        personality_vector=vector.tolist()
                    )
                    session.add(new_profile)
                await session.commit()

    async def get_all_profiles(self):
        """Fetches all student profiles for training."""
        from sqlalchemy import select
        async with self.async_session() as session:
            result = await session.execute(select(StudentProfile))
            return result.scalars().all()

postgres_client = PostgresClient()