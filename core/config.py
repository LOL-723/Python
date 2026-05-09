from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    DEEPSEEK_API_KEY:str
    DEEPSEEK_BASE_URL:str
    LLM_MODEL:str
    LLM_TIMEOUT:float

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()