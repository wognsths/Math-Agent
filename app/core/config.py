import os
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # ---- Project ----
    PROJECT_NAME: str = "MathSolver A2A"
    VERSION: str = "0.1.0"

    # ---- LLM Models ----
    OPENAI_API_KEY: str = ""
    OPENAI_MODEL: str = "gpt-4o-mini"

    GOOGLE_API_KEY: str = ""
    GOOGLE_MODEL: str = "gemini-001-flash"

    # ---- OCR Models ----
    OCR_MODEL: str = "microsoft/trocr-base-handwritten"

    # ---- API ----
    API_HOST: str = os.getenv("HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("PORT", "8000"))

    # ---- Output ----
    OUTPUT_DIR: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs")

    # ---- Logging ----
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    # Pydantic‑settings v2 style
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=True)


settings = Settings()
os.makedirs(settings.OUTPUT_DIR, exist_ok=True)