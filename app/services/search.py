import os
import json
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from typing import List
import numpy as np

from app.models.schemas import FrameResponse
from app.config import settings

class SearchService:
    def __init__(self):
        # Initialize ChromaDB client with new configuration
        self.client = chromadb.PersistentClient(path=str(settings.CHROMA_PERSIST_DIR))
        
        # Use sentence-transformers embedding model for semantic search
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=settings.EMBEDDING_MODEL
        )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=settings.COLLECTION_NAME,
            embedding_function=self.embedding_function
        )
        
        # Load frame metadata
        self.frame_metadata = {}
        metadata_path = os.path.join(settings.FRAMES_DIR, "metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                self.frame_metadata = json.load(f)
    
    def search(self, query: str, limit: int = 4) -> List[FrameResponse]:
        """
        Perform semantic search on video frames
        
        Args:
            query: The search query string
            limit: Maximum number of results to return
            
        Returns:
            List of FrameResponse objects containing matching frames
        """
        # Query the collection
        results = self.collection.query(
            query_texts=[query],
            n_results=limit
        )
        
        # Process results
        frame_results = []
        if results and len(results['ids'][0]) > 0:
            for i, frame_id in enumerate(results['ids'][0]):
                # Get metadata for the frame
                if frame_id in self.frame_metadata:
                    metadata = self.frame_metadata[frame_id]
                    
                    # Calculate similarity score (normalized between 0-1)
                    similarity = float(results['distances'][0][i])
                    if similarity > 1.0:  # Convert distance to similarity if needed
                        similarity = 1.0 - min(similarity, 2.0) / 2.0
                    
                    frame_results.append(
                        FrameResponse(
                            image_url=metadata["filename"],
                            video_filename=metadata["video_filename"],
                            frame_number=metadata["frame_number"],
                            similarity_score=similarity
                        )
                    )
        
        return frame_results