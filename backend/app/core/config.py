from pydantic_settings import BaseSettings
from typing import List

class Settings(BaseSettings):
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "SambioWage"
    VERSION: str = "1.0.0"
    
    # CORS 설정
    ALLOWED_HOSTS: List[str] = [
        "http://localhost:3000",
        "http://localhost:3001", 
        "http://127.0.0.1:3000",
        "http://127.0.0.1:3001",
        "*"  # 개발 환경에서 모든 도메인 허용
    ]
    
    # 파일 업로드 설정
    MAX_FILE_SIZE: int = 10 * 1024 * 1024  # 10MB
    UPLOAD_DIR: str = "uploads"
    ALLOWED_FILE_TYPES: List[str] = [".xlsx", ".xls", ".csv"]
    
    # ML 모델 설정
    MODEL_DIR: str = "models"
    
    class Config:
        env_file = ".env"

settings = Settings()