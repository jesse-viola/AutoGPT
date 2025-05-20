"""
BLIP Image Captioning Block for AutoGPT.

This block uses Salesforce's BLIP model to generate captions for images or video frames.
"""

import os
import uuid
import tempfile
import subprocess
from enum import Enum
from typing import List, Dict, Optional
from pathlib import Path

import cv2
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

from backend.data.block import Block, BlockCategory, BlockOutput, BlockSchema
from backend.data.model import SchemaField


class BlipModelSize(str, Enum):
    """Available BLIP model sizes."""
    BASE = "Salesforce/blip-image-captioning-base"
    LARGE = "Salesforce/blip-image-captioning-large"


class FrameExtractionMethod(str, Enum):
    """Methods for extracting frames from videos."""
    REGULAR_INTERVAL = "regular_interval"
    SCENE_DETECTION = "scene_detection"


class BlipImageCaptioningBlock(Block):
    """Block that generates captions for images using Salesforce's BLIP model."""

    class Input(BlockSchema):
        image_path: Optional[str] = SchemaField(
            description="Path to the image file to caption (either image_path or video_path must be provided)",
            default=None
        )
        video_path: Optional[str] = SchemaField(
            description="Path to the video file to extract frames from and caption (either image_path or video_path must be provided)",
            default=None
        )
        model_size: BlipModelSize = SchemaField(
            description="BLIP model size to use for captioning",
            default=BlipModelSize.BASE
        )
        frame_extraction_method: FrameExtractionMethod = SchemaField(
            description="Method to use for extracting frames from videos",
            default=FrameExtractionMethod.REGULAR_INTERVAL
        )
        frames_per_second: float = SchemaField(
            description="Number of frames to extract per second (for regular interval method)",
            default=0.5
        )
        scene_threshold: float = SchemaField(
            description="Threshold for scene detection (for scene detection method)",
            default=30.0
        )
        output_dir: str = SchemaField(
            description="Directory to save the extracted frames and captions",
            default="./captions"
        )

    class Output(BlockSchema):
        captions: List[Dict] = SchemaField(
            description="List of captions with timestamps and frame paths"
        )
        caption_file: str = SchemaField(
            description="Path to the JSON file containing all captions"
        )
        error: str = SchemaField(
            description="Error message if the captioning fails"
        )

    def __init__(self):
        super().__init__(
            id=str(uuid.uuid4()),
            description="Generates captions for images or video frames using Salesforce's BLIP model",
            categories={BlockCategory.IMAGE, BlockCategory.VIDEO},
            input_schema=BlipImageCaptioningBlock.Input,
            output_schema=BlipImageCaptioningBlock.Output,
            test_input={
                "image_path": "test_image.jpg",
                "model_size": BlipModelSize.BASE,
                "output_dir": "./test_captions"
            },
            test_output=[
                ("captions", [{"frame_path": "test_image.jpg", "caption": "a person walking on a beach"}]),
                ("caption_file", "./test_captions/captions.json")
            ],
        )
        self.processor = None
        self.model = None

    def _load_model(self, model_size: BlipModelSize):
        """Load the BLIP model."""
        self.processor = BlipProcessor.from_pretrained(model_size)
        self.model = BlipForConditionalGeneration.from_pretrained(model_size)
        
        # Move model to GPU if available
        if torch.cuda.is_available():
            self.model = self.model.to("cuda")

    def _caption_image(self, image_path: str) -> str:
        """Generate a caption for an image."""
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Process image
        inputs = self.processor(image, return_tensors="pt")
        
        # Move inputs to GPU if available
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        
        # Generate caption
        out = self.model.generate(**inputs)
        caption = self.processor.decode(out[0], skip_special_tokens=True)
        
        return caption

    def _extract_frames_regular_interval(self, video_path: str, output_dir: str, fps: float) -> List[Dict]:
        """Extract frames from a video at regular intervals."""
        # Open video
        video = cv2.VideoCapture(video_path)
        
        # Get video properties
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = video.get(cv2.CAP_PROP_FPS)
        duration = total_frames / video_fps
        
        # Calculate frame interval
        frame_interval = int(video_fps / fps)
        
        frames = []
        frame_count = 0
        
        while True:
            ret, frame = video.read()
            if not ret:
                break
                
            if frame_count % frame_interval == 0:
                # Calculate timestamp
                timestamp = frame_count / video_fps
                
                # Save frame
                frame_path = os.path.join(output_dir, f"frame_{frame_count:06d}.jpg")
                cv2.imwrite(frame_path, frame)
                
                frames.append({
                    "frame_path": frame_path,
                    "timestamp": timestamp,
                    "frame_number": frame_count
                })
                
            frame_count += 1
            
        video.release()
        return frames

    def _extract_frames_scene_detection(self, video_path: str, output_dir: str, threshold: float) -> List[Dict]:
        """Extract frames from a video using scene detection."""
        try:
            import PySceneDetect
            from scenedetect import VideoManager, SceneManager
            from scenedetect.detectors import ContentDetector
        except ImportError:
            raise ImportError("PySceneDetect is not installed. Install it with: pip install scenedetect")
        
        # Create video manager and scene manager
        video_manager = VideoManager([video_path])
        scene_manager = SceneManager()
        scene_manager.add_detector(ContentDetector(threshold=threshold))
        
        # Start video manager
        video_manager.start()
        
        # Detect scenes
        scene_manager.detect_scenes(frame_source=video_manager)
        
        # Get scene list
        scene_list = scene_manager.get_scene_list()
        
        # Extract first frame from each scene
        frames = []
        video = cv2.VideoCapture(video_path)
        video_fps = video.get(cv2.CAP_PROP_FPS)
        
        for i, scene in enumerate(scene_list):
            start_frame = scene[0].frame_num
            
            # Set video to start frame
            video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
            # Read frame
            ret, frame = video.read()
            if ret:
                # Calculate timestamp
                timestamp = start_frame / video_fps
                
                # Save frame
                frame_path = os.path.join(output_dir, f"scene_{i:03d}.jpg")
                cv2.imwrite(frame_path, frame)
                
                frames.append({
                    "frame_path": frame_path,
                    "timestamp": timestamp,
                    "frame_number": start_frame,
                    "scene_number": i
                })
        
        video.release()
        return frames

    def _check_ffmpeg_installed(self) -> bool:
        """Check if FFmpeg is installed."""
        try:
            subprocess.run(
                ["ffmpeg", "-version"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False
            )
            return True
        except FileNotFoundError:
            return False

    def run(self, input_data: Input, **kwargs) -> BlockOutput:
        """Run the BLIP image captioning block."""
        try:
            # Check if either image_path or video_path is provided
            if input_data.image_path is None and input_data.video_path is None:
                raise ValueError("Either image_path or video_path must be provided")
            
            # Check if FFmpeg is installed for video processing
            if input_data.video_path and not self._check_ffmpeg_installed():
                raise RuntimeError(
                    "FFmpeg is not installed. It's required for video processing."
                )
            
            # Create output directory if it doesn't exist
            os.makedirs(input_data.output_dir, exist_ok=True)
            
            # Load BLIP model
            self._load_model(input_data.model_size)
            
            captions = []
            
            # Process image
            if input_data.image_path:
                image_path = os.path.abspath(input_data.image_path)
                if not os.path.exists(image_path):
                    raise FileNotFoundError(f"Image not found: {image_path}")
                
                caption = self._caption_image(image_path)
                captions.append({
                    "frame_path": image_path,
                    "caption": caption,
                    "timestamp": 0.0
                })
            
            # Process video
            elif input_data.video_path:
                video_path = os.path.abspath(input_data.video_path)
                if not os.path.exists(video_path):
                    raise FileNotFoundError(f"Video not found: {video_path}")
                
                # Extract frames
                frames = []
                if input_data.frame_extraction_method == FrameExtractionMethod.REGULAR_INTERVAL:
                    frames = self._extract_frames_regular_interval(
                        video_path, 
                        input_data.output_dir, 
                        input_data.frames_per_second
                    )
                else:
                    frames = self._extract_frames_scene_detection(
                        video_path, 
                        input_data.output_dir, 
                        input_data.scene_threshold
                    )
                
                # Generate captions for each frame
                for frame in frames:
                    caption = self._caption_image(frame["frame_path"])
                    captions.append({
                        "frame_path": frame["frame_path"],
                        "caption": caption,
                        "timestamp": frame["timestamp"],
                        "frame_number": frame.get("frame_number"),
                        "scene_number": frame.get("scene_number")
                    })
            
            # Save captions to file
            import json
            caption_file = os.path.join(input_data.output_dir, "captions.json")
            with open(caption_file, "w", encoding="utf-8") as f:
                json.dump(captions, f, indent=2)
            
            yield "captions", captions
            yield "caption_file", caption_file
            
        except Exception as e:
            yield "error", f"Error generating captions: {str(e)}"
