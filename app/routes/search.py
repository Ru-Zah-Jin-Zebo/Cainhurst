from fastapi import APIRouter, Query, HTTPException, Request
from typing import Optional
from app.models.schemas import SearchResponse
from app.services.search import SearchService

router = APIRouter(prefix="/api", tags=["search"])
search_service = SearchService()

@router.get("/search", response_model=SearchResponse)
async def search_frames(
    request: Request,
    query: str = Query(..., description="Search query string"),
    limit: int = Query(4, description="Maximum number of results to return", ge=1, le=20)
):
    """
    Search for relevant video frames based on a semantic query.
    Returns up to 'limit' frames (default 4) that match the query.
    """
    try:
        # Perform the semantic search
        results = search_service.search(query, limit)
        
        # Convert to absolute URLs
        base_url = str(request.base_url).rstrip('/')
        for result in results:
            result.image_url = f"{base_url}/frames/{result.image_url}"
        
        # Return formatted response
        return SearchResponse(
            results=results,
            count=len(results),
            query=query
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")