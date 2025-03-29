from pydantic import BaseModel, Field, HttpUrl
from typing import List, Optional

class FrameResponse(BaseModel):
    """Response model for a single frame"""
    image_url: str
    video_filename: str
    frame_number: int
    similarity_score: Optional[float] = None
    
class SearchResponse(BaseModel):
    """Response model for search results"""
    results: List[FrameResponse] = Field(default_factory=list)
    count: int = 0
    query: str