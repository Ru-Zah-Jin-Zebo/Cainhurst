from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os
from app.routes import search

app = FastAPI(
    title="Ember Video Frame Search API",
    description="API for semantic search of video frames",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static directory for serving frame images
frames_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "frames")
os.makedirs(frames_dir, exist_ok=True)
app.mount("/frames", StaticFiles(directory=frames_dir), name="frames")

# Include routers
app.include_router(search.router)

@app.get("/")
async def root():
    return {"message": "Welcome to Ember Video Frame Search API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)