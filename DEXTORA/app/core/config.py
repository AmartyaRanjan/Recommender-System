from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache

class Settings(BaseSettings):
    # App Settings
    PROJECT_NAME: str = "DEXTORA AI"
    DEBUG: bool = False
    
    # Redis (Google Memorystore)
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    
    # Database (PostgreSQL / AlloyDB)
    DATABASE_URL: str = "postgresql+psycopg://user:pass@localhost:5432/dextora"
    
    # Model Paths
    SAINT_MODEL_PATH: str = "app/ml_assets/saint_weights.pt"
    RL_MODEL_PATH: str = "app/ml_assets/ppo_student_policy.zip"

    # Security
    SECRET_KEY: str = "SUPER_SECRET_KEY_CHANGE_IN_PROD"
    
    # Cloud Storage
    GCS_BUCKET_NAME: str | None = None
    GCS_MODEL_PREFIX: str = ""
    
    model_config = SettingsConfigDict(env_file=".env")

@lru_cache()
def get_settings():
    return Settings()

settings = get_settings()