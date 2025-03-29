#!/usr/bin/env python3
"""
Script to extract frames from videos and index them in ChromaDB
"""
import os
import sys
import json
import argparse
from pathlib import Path
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from tqdm import tqdm

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.services.extractor import FrameExtractor
from app.config import settings


def process_videos(args):
    """Extract frames from videos"""
    extractor = FrameExtractor()
    return extractor.process_all_videos()


def index_frames(metadata, args):
    """Index frames in ChromaDB"""
    print("Indexing frames in ChromaDB...")
    
    # Initialize ChromaDB client with new configuration
    client = chromadb.PersistentClient(path=str(settings.CHROMA_PERSIST_DIR))
    
    # Use sentence-transformers for embedding
    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=settings.EMBEDDING_MODEL
    )
    
    # Create or get collection
    if args.recreate and settings.COLLECTION_NAME in [c.name for c in client.list_collections()]:
        client.delete_collection(settings.COLLECTION_NAME)
        print(f"Deleted existing collection: {settings.COLLECTION_NAME}")
    
    collection = client.get_or_create_collection(
        name=settings.COLLECTION_NAME,
        embedding_function=embedding_function
    )
    
    # Prepare data for batch indexing
    ids = []
    documents = []
    metadatas = []
    
    for frame_id, frame_data in tqdm(metadata.items(), desc="Preparing embeddings"):
        # Create document to embed - combine all text fields
        doc_text = []
        
        # Add metadata fields that might be relevant
        if "description" in frame_data:
            doc_text.append(frame_data["description"])
        
        # Add the video filename (without extension) as it might contain relevant keywords
        video_name = os.path.splitext(frame_data["video_filename"])[0]
        video_name = video_name.replace("_", " ").replace("-", " ")
        doc_text.append(f"Video: {video_name}")
        
        # Add general context about the frame
        doc_text.append("Frame from video content")
        doc_text.append("Scene from video")
        
        # Join all text elements
        document = " ".join(doc_text)
        
        # Skip if document is empty
        if not document.strip():
            continue
        
        # Add to batch
        ids.append(frame_id)
        documents.append(document)
        
        # Add relevant metadata for filtering
        metadatas.append({
            "video_filename": frame_data["video_filename"],
            "frame_number": frame_data["frame_number"],
            "timestamp": frame_data["timestamp"]
        })
    
    # Check if we have data to index
    if not ids:
        print("No frames to index!")
        return
    
    # Add items to collection in batches
    batch_size = 100
    for i in range(0, len(ids), batch_size):
        batch_end = min(i + batch_size, len(ids))
        collection.add(
            ids=ids[i:batch_end],
            documents=documents[i:batch_end],
            metadatas=metadatas[i:batch_end]
        )
        print(f"Indexed batch {i//batch_size + 1}/{(len(ids)-1)//batch_size + 1}")
    
    print(f"Indexed {len(ids)} frames in ChromaDB")
    
    # Test a query
    if args.test:
        test_query(collection)


def test_query(collection):
    """Run a test query on the collection"""
    print("\nTesting the search functionality...")
    test_queries = [
        "Person holding a phone",
        "Outdoor scene in daylight",
        "Person smiling",
        "Urban environment"
    ]
    
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        results = collection.query(
            query_texts=[query],
            n_results=2
        )
        
        for i, (doc_id, distance) in enumerate(zip(results['ids'][0], results['distances'][0])):
            # Convert distance to similarity score (0-1)
            similarity = 1.0 - min(distance, 2.0) / 2.0
            print(f"  Result {i+1}: ID={doc_id}, Similarity={similarity:.2f}")
            print(f"  Document: {results['documents'][0][i]}")


def main():
    parser = argparse.ArgumentParser(description="Process videos and index frames for semantic search")
    parser.add_argument("--recreate", action="store_true", help="Recreate the ChromaDB collection")
    parser.add_argument("--test", action="store_true", help="Run test queries after indexing")
    parser.add_argument("--skip-extraction", action="store_true", help="Skip frame extraction")
    args = parser.parse_args()
    
    # Process videos to extract frames
    if not args.skip_extraction:
        print("Extracting frames from videos...")
        metadata = process_videos(args)
    else:
        # Load existing metadata
        metadata_path = os.path.join(settings.FRAMES_DIR, "metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            print(f"Loaded metadata for {len(metadata)} frames")
        else:
            print("No metadata found. Run without --skip-extraction first")
            return
    
    # Index frames in ChromaDB
    index_frames(metadata, args)


if __name__ == "__main__":
    main()