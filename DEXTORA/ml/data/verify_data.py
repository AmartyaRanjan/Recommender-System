from sqlalchemy import create_engine, text
from app.core.config import settings

# Use SYNC engine
SYNC_DATABASE_URL = settings.DATABASE_URL.replace("+psycopg", "").replace("+asyncpg", "")
if "postgresql://" not in SYNC_DATABASE_URL:
    SYNC_DATABASE_URL = SYNC_DATABASE_URL.replace("postgresql", "postgresql+psycopg2")

def verify_data():
    engine = create_engine(SYNC_DATABASE_URL, echo=False)
    with engine.connect() as conn:
        result = conn.execute(text("SELECT COUNT(*) FROM student_profiles"))
        count = result.scalar()
        print(f"✅ Total Student Profiles: {count}")
        
        if count > 0:
            print("\nSample Data:")
            sample = conn.execute(text("SELECT student_id, grade, curriculum FROM student_profiles LIMIT 5"))
            for row in sample:
                print(row)

if __name__ == "__main__":
    verify_data()
