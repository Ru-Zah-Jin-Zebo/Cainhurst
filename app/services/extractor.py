import os
import cv2
import json
import uuid
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from app.config import settings

try:
    # Try to import CLIP for automatic captioning
    import torch
    import clip
    from PIL import Image
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False


class FrameExtractor:
    def __init__(self, videos_dir: Path = None, frames_dir: Path = None, ember_only: bool = False):
        """
        Initialize the frame extractor service
        
        Args:
            videos_dir: Directory containing videos to process
            frames_dir: Directory to save extracted frames
            ember_only: If True, use only Ember-specific labels for faster processing
        """
        self.videos_dir = videos_dir or settings.VIDEOS_DIR
        self.frames_dir = frames_dir or settings.FRAMES_DIR
        self.ember_only = ember_only
        
        # Ensure directories exist
        os.makedirs(self.videos_dir, exist_ok=True)
        os.makedirs(self.frames_dir, exist_ok=True)
        
        # Set up CLIP model if available and enabled
        self.clip_model = None
        self.clip_preprocess = None
        
        if CLIP_AVAILABLE and settings.USE_CLIP_CAPTIONING:
            try:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
                self.clip_model, self.clip_preprocess = clip.load(settings.CLIP_MODEL, device=self.device)
                print(f"CLIP model loaded for automatic captioning (using {self.device})")
            except Exception as e:
                print(f"Failed to load CLIP model: {e}")
    
    def _get_candidate_labels(self) -> List[str]:
        """Get the appropriate set of candidate labels based on mode"""
        if self.ember_only:
            return [
                # Ember-specific labels
                "Ember character", "Ember holding phone", "Ember using phone",
                "Ember close-up", "Ember from distance", "Ember smiling",
                "Ember talking", "Ember walking", "Ember sitting", "Ember standing",
                
                # Basic scene context
                "indoor scene", "outdoor scene", "daytime scene", "nighttime scene",
                
                # Phone-specific actions
                "person holding smartphone", "person using mobile phone",
                "person looking at phone screen", "person texting on phone",
                "person taking selfie", "person recording video"
            ]
        else:
            return [
                # Ember-specific labels
                "Ember character", "Ember holding phone", "Ember using phone",
                "Ember close-up", "Ember from distance", "Ember smiling",
                "Ember talking", "Ember walking", "Ember sitting", "Ember standing",
                
                # COCO person-related categories
                "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
                "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
                
                # COCO indoor objects
                "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
                "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
                "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
                
                # COCO outdoor objects
                "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
                "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
                
                # Scene and environment labels
                "indoor scene", "outdoor scene", "urban environment", "natural environment",
                "daytime scene", "nighttime scene", "sunset scene", "sunrise scene",
                "crowded scene", "empty scene", "busy environment", "quiet environment",
                
                # Action and pose labels
                "person standing", "person sitting", "person walking", "person running",
                "person jumping", "person dancing", "person exercising", "person working",
                "person using phone", "person using laptop", "person reading", "person writing",
                "person talking", "person smiling", "person laughing", "person looking",
                
                # Phone-specific labels
                "person holding smartphone", "person using mobile phone", "person looking at phone screen",
                "person texting on phone", "person taking selfie", "person recording video",
                "person scrolling phone", "person holding phone up", "person holding phone down",
                "person using phone while walking", "person using phone while sitting",
                
                # Additional context labels
                "close-up shot", "wide shot", "medium shot", "group shot", "solo shot",
                "candid moment", "posed shot", "action shot", "portrait shot", "landscape shot",
                "street scene", "park scene", "office scene", "home scene", "restaurant scene"
            ]
    
    def extract_frames_from_video(
        self, 
        video_path: Path, 
        frames_per_second: float = None,
        max_frames: int = None
    ) -> List[Dict]:
        """
        Extract frames from a video file
        
        Args:
            video_path: Path to the video file
            frames_per_second: Number of frames to extract per second
            max_frames: Maximum number of frames to extract
            
        Returns:
            List of dictionaries with frame metadata
        """
        fps = frames_per_second or settings.FRAMES_PER_SECOND
        video_filename = os.path.basename(video_path)
        
        # Open video file
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        # Get video properties
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate frame interval
        frame_interval = int(video_fps / fps)
        if frame_interval < 1:
            frame_interval = 1
        
        # Limit total frames if specified
        if max_frames:
            frames_to_extract = min(total_frames, max_frames * frame_interval)
        else:
            frames_to_extract = total_frames
        
        # Extract frames
        frame_metadata = []
        frame_count = 0
        processed_count = 0
        
        while frame_count < frames_to_extract:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame at specified interval
            if frame_count % frame_interval == 0:
                # Generate unique ID for frame
                frame_id = str(uuid.uuid4())
                
                # Create frame filename
                frame_filename = f"{os.path.splitext(video_filename)[0]}_frame_{processed_count:05d}.jpg"
                frame_path = os.path.join(self.frames_dir, frame_filename)
                
                # Save frame
                cv2.imwrite(frame_path, frame)
                
                # Create metadata for frame
                metadata = {
                    "id": frame_id,
                    "filename": frame_filename,
                    "video_filename": video_filename,
                    "frame_number": frame_count,
                    "timestamp": frame_count / video_fps
                }
                
                # Generate description with CLIP if available
                if self.clip_model is not None:
                    description = self._generate_frame_description(frame_path)
                    if description:
                        metadata["description"] = description
                
                frame_metadata.append(metadata)
                processed_count += 1
                
                # Stop if we've reached max_frames
                if max_frames and processed_count >= max_frames:
                    break
            
            frame_count += 1
        
        # Release video capture
        cap.release()
        
        return frame_metadata
    
    def _generate_frame_description(self, frame_path: str) -> Optional[str]:
        """Generate a textual description of the frame using CLIP"""
        try:
            # Load and preprocess the image
            image = self.clip_preprocess(Image.open(frame_path)).unsqueeze(0).to(self.device)
            
            # Get appropriate candidate labels based on mode
            candidate_labels = self._get_candidate_labels()
            
            # Encode the candidate labels
            text = clip.tokenize(candidate_labels).to(self.device)
            
            # Compute similarity between image and text
            with torch.no_grad():
                image_features = self.clip_model.encode_image(image)
                text_features = self.clip_model.encode_text(text)
                
                # Normalize features
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                
                # Compute similarity
                similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            
            # Get top matching labels
            values, indices = similarity[0].topk(3)
            
            # Combine top labels into a description
            top_labels = [candidate_labels[idx] for idx in indices]
            description = ", ".join(top_labels)
            
            return description
        except Exception as e:
            print(f"Error generating frame description: {e}")
            return None
    
    def process_all_videos(self) -> Dict[str, Dict]:
        """
        Process all videos in the videos directory
        
        Returns:
            Dictionary mapping frame IDs to frame metadata
        """
        # Get list of video files
        video_files = []
        for ext in ['.mp4', '.avi', '.mov', '.mkv']:
            video_files.extend(list(self.videos_dir.glob(f'*{ext}')))
        
        if not video_files:
            raise ValueError(f"No video files found in {self.videos_dir}")
        
        print(f"Found {len(video_files)} video files to process")
        
        # Process each video
        all_metadata = {}
        for video_path in video_files:
            print(f"Processing video: {video_path}")
            frame_metadata = self.extract_frames_from_video(video_path)
            
            # Add to metadata dictionary
            for metadata in frame_metadata:
                all_metadata[metadata["id"]] = metadata
        
        # Save metadata to file
        metadata_path = os.path.join(self.frames_dir, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(all_metadata, f, indent=2)
        
        print(f"Processed {len(all_metadata)} frames from {len(video_files)} videos")
        return all_metadata