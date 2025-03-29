import os
from pathlib import Path
from pydantic_settings import BaseSettings

# Get the project root directory
ROOT_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    # API settings
    API_TITLE: str = "Ember Video Frame Search API"
    API_VERSION: str = "1.0.0"
    
    # Directories
    VIDEOS_DIR: Path = ROOT_DIR / "data" / "videos"
    FRAMES_DIR: Path = ROOT_DIR / "data" / "frames"
    CHROMA_PERSIST_DIR: Path = ROOT_DIR / "data" / "chromadb"
    
    # Frame extraction settings
    FRAMES_PER_SECOND: float = 1.0  # Extract 1 frame per second
    
    # Embedding model settings
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"  # Lightweight model that works well
    COLLECTION_NAME: str = "ember_frames"
    
    # CLIP model settings for generating image descriptions
    USE_CLIP_CAPTIONING: bool = True
    CLIP_MODEL: str = "ViT-B/32"

    class Config:
        env_file = ".env"
        case_sensitive = True


# Create settings instance
settings = Settings()

# Ensure directories exist
os.makedirs(settings.VIDEOS_DIR, exist_ok=True)
os.makedirs(settings.FRAMES_DIR, exist_ok=True)
os.makedirs(settings.CHROMA_PERSIST_DIR, exist_ok=True)