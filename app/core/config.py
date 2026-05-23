import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    DJANGO_ALERTS_URL: str = "http://localhost:8000/api/alertas/"
    DJANGO_API_KEY: str = "dev-api-key"
    SIMULATION_MODE: bool = False
    ALERT_PROBABILITY: float = 0.35
    APP_NAME: str = "AI Service Simulator"
    APP_PORT: int = 9000
    CORS_ORIGINS: str = "http://localhost:5173,http://localhost:3000,http://127.0.0.1:5173"

    class Config:
        env_file = ".env"

settings = Settings()
