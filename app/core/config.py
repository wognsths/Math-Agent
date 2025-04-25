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

    # ---- Paths ----
    # Base output directory
    OUTPUT_DIR: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs")
    
    # Sub-directories for specific types of data
    IMAGE_DIR: str = os.path.join(OUTPUT_DIR, "images")
    OCR_CORRECTIONS_DIR: str = os.path.join(OUTPUT_DIR, "ocr_corrections")
    LATEX_ERRORS_DIR: str = os.path.join(OUTPUT_DIR, "latex_errors")
    RENDERED_LATEX_DIR: str = os.path.join(OUTPUT_DIR, "rendered_latex")
    VERIFICATION_DIR: str = os.path.join(OUTPUT_DIR, "verifications")
    SOLUTIONS_DIR: str = os.path.join(OUTPUT_DIR, "solutions")
    
    # ---- API Endpoints ----
    # Base URL for internal API calls
    API_BASE_URL: str = os.getenv("API_BASE_URL", "http://localhost:8000")

    # ---- Logging ----
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_DIR: str = os.path.join(OUTPUT_DIR, "logs")

    # Pydantic‑settings v2 style
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=True)


settings = Settings()

# Create all required directories
os.makedirs(settings.OUTPUT_DIR, exist_ok=True)
os.makedirs(settings.IMAGE_DIR, exist_ok=True)
os.makedirs(settings.OCR_CORRECTIONS_DIR, exist_ok=True)
os.makedirs(settings.LATEX_ERRORS_DIR, exist_ok=True)
os.makedirs(settings.RENDERED_LATEX_DIR, exist_ok=True)
os.makedirs(settings.VERIFICATION_DIR, exist_ok=True)
os.makedirs(settings.SOLUTIONS_DIR, exist_ok=True)
os.makedirs(settings.LOG_DIR, exist_ok=True)