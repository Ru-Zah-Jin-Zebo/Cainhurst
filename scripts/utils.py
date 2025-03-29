"""Utility functions for video processing and search"""
import os
import shutil
import zipfile
from pathlib import Path


def extract_sample_data(zip_path, output_dir):
    """
    Extract the sample data zip file to the videos directory
    
    Args:
        zip_path: Path to the sample data zip file
        output_dir: Directory to extract videos to
    """
    # Check if zip file exists
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"Sample data zip file not found: {zip_path}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract zip file
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(output_dir)
    
    print(f"Extracted sample data to {output_dir}")
    
    # Move any videos from nested directories to the videos directory
    for root, dirs, files in os.walk(output_dir):
        for file in files:
            if file.endswith(('.mp4', '.avi', '.mov', '.mkv')):
                source_path = os.path.join(root, file)
                target_path = os.path.join(output_dir, file)
                
                # Only move if source and target are different
                if source_path != target_path:
                    shutil.move(source_path, target_path)
                    print(f"Moved {file} to {output_dir}")


def clean_data_directories(frames_dir=None, chroma_dir=None):
    """
    Clean data directories to start fresh
    
    Args:
        frames_dir: Directory containing extracted frames
        chroma_dir: Directory containing ChromaDB data
    """
    dirs_to_clean = []
    
    if frames_dir:
        dirs_to_clean.append(frames_dir)
    
    if chroma_dir:
        dirs_to_clean.append(chroma_dir)
    
    for directory in dirs_to_clean:
        if os.path.exists(directory):
            # Remove all files in directory
            for item in os.listdir(directory):
                item_path = os.path.join(directory, item)
                if os.path.isfile(item_path):
                    os.unlink(item_path)
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)
            
            print(f"Cleaned directory: {directory}")