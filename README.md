# Ember Video Frame Search API

A FastAPI application that enables semantic search on video frames from the Ember video library. The API allows users to search with natural language queries and returns the most relevant video frames.

## Features

- **Frame Extraction**: Automatically extracts frames from videos at a configurable rate
- **Semantic Search**: Uses sentence-transformers embeddings and ChromaDB for semantic similarity search
- **Smart Descriptions**: Optionally generates image descriptions using CLIP to improve search relevance
- **RESTful API**: Simple API for searching video frames based on natural language queries
- **Scalable**: Designed to handle a growing library of videos

## Frame Detection System

The system uses CLIP (Contrastive Language-Image Pre-training) for frame detection and description generation. It leverages a comprehensive set of labels based on the COCO (Common Objects in Context) dataset and additional custom labels specific to the Ember video library.

### Label Categories

1. **Ember-Specific Labels**
   - Character identification: "Ember character", "Ember close-up", "Ember from distance"
   - Actions: "Ember smiling", "Ember talking", "Ember walking", "Ember sitting", "Ember standing"
   - Phone interactions: "Ember holding phone", "Ember using phone"

2. **COCO Dataset Categories**
   - Person-related: "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat"
   - Indoor objects: "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "cell phone"
   - Outdoor objects: "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard"

3. **Scene and Environment Labels**
   - Location types: "indoor scene", "outdoor scene", "urban environment", "natural environment"
   - Time of day: "daytime scene", "nighttime scene", "sunset scene", "sunrise scene"
   - Crowd levels: "crowded scene", "empty scene", "busy environment", "quiet environment"

4. **Action and Pose Labels**
   - Basic actions: "person standing", "person sitting", "person walking", "person running"
   - Complex actions: "person jumping", "person dancing", "person exercising", "person working"
   - Device usage: "person using phone", "person using laptop", "person reading", "person writing"

5. **Phone-Specific Labels**
   - Actions: "person holding smartphone", "person using mobile phone", "person looking at phone screen"
   - Specific uses: "person texting on phone", "person taking selfie", "person recording video"
   - Context: "person scrolling phone", "person holding phone up", "person holding phone down"

6. **Shot and Composition Labels**
   - Shot types: "close-up shot", "wide shot", "medium shot", "group shot", "solo shot"
   - Scene types: "candid moment", "posed shot", "action shot", "portrait shot", "landscape shot"
   - Locations: "street scene", "park scene", "office scene", "home scene", "restaurant scene"

### How It Works

1. **Frame Extraction**: The system extracts frames from videos at a configurable rate (default: 1 frame per second)
2. **CLIP Analysis**: Each frame is analyzed using CLIP to generate a description based on the most relevant labels
3. **Description Generation**: The top 3 matching labels are combined into a natural language description
4. **Indexing**: Descriptions are indexed in ChromaDB for semantic search

This comprehensive label set ensures accurate detection of:
- Ember's presence and actions
- Phone usage and interactions
- Scene context and environment
- Shot composition and style

## Requirements

- Python 3.12.7 (required - other versions may have compatibility issues)
- FastAPI
- ChromaDB
- OpenCV
- Sentence Transformers
- CLIP (optional, for auto-captioning)

## Setup Instructions

### Option 1: Local Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Ru-Zah-Jin-Zebo/Cainhurst.git
   cd Cainhurst
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   # Using conda (recommended):
   conda create -n cainhurst python=3.12.7
   conda activate cainhurst

   # Or using venv:
   python -m venv venv
   source venv/bin/activate  # On Windows, use: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Place your video files in the `data/videos` directory.

5a. Fast/ Specific Option: Extract and index frames for ember video solution only:
   ```bash
   python scripts/process_videos.py --ember-only
   ```

5b. Slow/ General Option: Extract and index frames:
   ```bash
   python scripts/process_videos.py
   ```

6. Start the FastAPI server:
   ```bash
   uvicorn app.main:app --reload
   ```

7. The API will be available at http://localhost:8000

### Option 2: Docker Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Ru-Zah-Jin-Zebo/Cainhurst.git
   cd Cainhurst
   ```

2. Place your video files in the `data/videos` directory.

3. Build the Docker image:
   ```bash
   docker build -t cainhurst .
   ```

4. Run the container:
   ```bash
   # On Windows PowerShell:
   docker run -p 8000:8000 -v ${PWD}/data:/app/data cainhurst

   # On Linux/macOS:
   docker run -p 8000:8000 -v $(pwd)/data:/app/data cainhurst
   ```

5. The API will be available at http://localhost:8000

Note: The Docker setup uses a multi-stage build to optimize the image size and includes all necessary dependencies, including git for installing packages from GitHub repositories.

## API Documentation

### Search Endpoint

**Endpoint**: `/api/search`  
**Method**: GET  
**Query Parameters**:
- `query` (string, required): The search query string
- `limit` (integer, optional, default=4): Maximum number of results to return

**Example Request**:
```
GET /api/search?query=Person%20holding%20phone&limit=4
```

**Example Response**:
```json
{
  "results": [
    {
      "image_url": "http://localhost:8000/frames/video1_frame_00123.jpg",
      "video_filename": "video1.mp4",
      "frame_number": 123,
      "similarity_score": 0.89
    },
    {
      "image_url": "http://localhost:8000/frames/video2_frame_00045.jpg",
      "video_filename": "video2.mp4",
      "frame_number": 45,
      "similarity_score": 0.82
    }
  ],
  "count": 2,
  "query": "Person holding phone"
}
```

## Processing Video Files

The `process_videos.py` script extracts frames from videos and indexes them for semantic search:

```bash
# Basic usage (extract and index all videos)
python scripts/process_videos.py

# Recreate the index from scratch
python scripts/process_videos.py --recreate

# Skip extraction and only reindex existing frames
python scripts/process_videos.py --skip-extraction

# Run test queries after indexing
python scripts/process_videos.py --test
```

## Configuration

The application's configuration can be adjusted in `app/config.py` or via environment variables:

- `FRAMES_PER_SECOND`: Number of frames to extract per second (default: 1.0)
- `EMBEDDING_MODEL`: Sentence-transformer model to use (default: "all-MiniLM-L6-v2")
- `USE_CLIP_CAPTIONING`: Whether to use CLIP for auto-captioning (default: true)

## Design Decisions and Approach

### Architecture Overview

The system follows a layered architecture:

1. **Data Layer**: Videos are processed into frames and stored on disk
2. **Vector Store**: ChromaDB maintains embeddings and enables semantic search
3. **Service Layer**: Abstracts frame extraction and search operations
4. **API Layer**: FastAPI provides the RESTful interface

### Technology Choices

- **FastAPI**: Chosen for its high performance, automatic documentation, and type checking
- **ChromaDB**: A lightweight vector database perfect for semantic search use cases
- **Sentence Transformers**: Provides high-quality embeddings for text-based search
- **OpenCV**: Efficient video frame extraction
- **CLIP** (optional): Enhances searchability by auto-generating frame descriptions

### Video Processing Strategy

Rather than processing videos on-demand, the system pre-processes all videos to:
1. Extract frames at a configurable rate (default: 1 frame per second)
2. Optionally generate descriptions using CLIP
3. Create embeddings for each frame using descriptive text
4. Store frames as JPEG images for efficient retrieval

This approach provides fast search results at the cost of upfront processing time.

### Scalability Considerations

The current implementation works well for the sample dataset of 10 videos. For larger datasets:

- Increase batch size during indexing for efficiency
- Consider using a persistent ChromaDB instance instead of local storage
- Implement pagination for search results
- Add more filtering options (by video, time range, etc.)
- Use a CDN for serving frame images in production

### On-Demand Processing Alternative

The current implementation pre-extracts and indexes all video frames for immediate search capabilities. An alternative on-demand approach would work as follows:

#### How On-Demand Processing Would Work:

1. **Video Metadata Indexing**:
   - Instead of extracting all frames, only store video metadata (title, duration, etc.)
   - Create lightweight scene detection to identify key segments
   
2. **Query-Time Processing**:
   - When a user submits a search query, dynamically extract relevant frames from videos
   - Use the query to determine which videos and timestamps are most likely to contain relevant content
   - Extract frames only from those portions of the videos
   
3. **Progressive Enhancement**:
   - Return initial results quickly based on video metadata matches
   - Process videos in the background and update results as more relevant frames are found
   - Cache processed frames for similar future queries

4. **Implementation Approach**:
   - Use a job queue system (like Celery) for asynchronous video processing tasks
   - Implement a caching layer to store frequently accessed frames
   - Create a stateful API that can stream updates as frames are processed

#### Trade-offs vs. Pre-processing:

| Aspect | Pre-processing | On-Demand |
|--------|---------------|-----------|
| Initial setup time | Longer | Shorter |
| Storage requirements | Higher | Lower |
| Search response time | Faster | Slower initially, improves with caching |
| Processing resources | Front-loaded | Distributed over time |
| Freshness with new videos | Requires reprocessing | Immediate incorporation |

On-demand processing would be especially beneficial in scenarios where:
- The video library is very large
- Storage resources are limited
- Only a small subset of frames are typically searched for
- New videos are frequently added to the system

### Future Enhancements

- Add authentication and rate-limiting for API security
- Implement caching for frequent queries
- Add admin endpoints for video management
- Support for background video processing tasks
- Add more advanced filtering and sorting options
- Develop an intuitive web interface that allows users to visually browse search results, preview video frames, and manage the video library without technical knowledge
- Implement drag-and-drop functionality for video uploads and a dashboard to monitor processing status
- Add customization options for selecting which videos to include in searches and adjusting search relevance parameters
