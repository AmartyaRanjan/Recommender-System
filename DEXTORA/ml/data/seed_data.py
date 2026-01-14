import numpy as np
import random
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from app.db.postgres_client import Base, StudentProfile
from app.core.config import settings

# Use SYNC engine for seeding to avoid async driver issues
# Replace +psycopg or +asyncpg with +psycopg2 (or default)
SYNC_DATABASE_URL = settings.DATABASE_URL.replace("+psycopg", "").replace("+asyncpg", "")
if "postgresql://" not in SYNC_DATABASE_URL:
    SYNC_DATABASE_URL = SYNC_DATABASE_URL.replace("postgresql", "postgresql+psycopg2")

def seed_database():
    print("🚀 Starting DEXTORA Seed Process (SYNC)...")
    print(f"DEBUG: DATABASE_URL={SYNC_DATABASE_URL}")

    engine = create_engine(SYNC_DATABASE_URL, echo=False)

    # 1. Initialize Postgres Extension & Tables
    with engine.begin() as conn:
        print("🔧 Enabling pgvector extension...")
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        print("🏗️ Creating tables...")
        Base.metadata.create_all(conn)
    
    # 2. Define Personality Archetypes
    archetypes = {
        "Visual_Sprinter": [0.8, 0.1, 0.1], # High visual, low text/audio
        "Textual_DeepDiver": [0.1, 0.9, 0.0],
        "Kinetic_Explorer": [0.3, 0.2, 0.5]
    }

    # Use Raw DBAPI Connection to bypass SQLAlchemy completely
    raw_conn = engine.raw_connection()
    try:
        cursor = raw_conn.cursor()
        print(f"👥 Generating profiles for 100 students...")
        
        for i in range(100):
            s_id = f"STU_{1000 + i}"
            arch = random.choice(list(archetypes.keys()))
            
            # Generate a random 128-d vector centered around the archetype
            base = np.zeros(128)
            base[:3] = archetypes[arch] # First 3 dims represent learning mode
            noise = np.random.normal(0, 0.05, 128)
            personality_vector = (base + noise).astype(np.float32)
            vec_str = str(personality_vector.tolist())

            # Raw SQL with %s placeholders for psycopg2
            cursor.execute("""
                INSERT INTO student_profiles (student_id, grade, curriculum, personality_vector)
                VALUES (%s, %s, %s, %s)
            """, (s_id, random.randint(6, 12), random.choice(["K-12", "IGCSE", "IB"]), vec_str))
        
        raw_conn.commit()
        print("✅ Seeding Complete. 100 students created.")
    except Exception as e:
        raw_conn.rollback()
        print(f"❌ Error: {e}")
        raise e
    finally:
        raw_conn.close()

if __name__ == "__main__":
    seed_database()