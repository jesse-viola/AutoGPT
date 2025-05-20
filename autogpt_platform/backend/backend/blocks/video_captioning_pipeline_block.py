"""
Video Captioning and Voiceover Pipeline Block for AutoGPT.

This block combines Whisper, BLIP, and Bark to create a complete pipeline for
transcribing, captioning, and adding voiceovers to videos.
"""

import os
import uuid
import json
import tempfile
import subprocess
from enum import Enum
from typing import List, Dict, Optional, Union
from pathlib import Path

from backend.data.block import Block, BlockCategory, BlockOutput, BlockSchema
from backend.data.model import SchemaField

from backend.blocks.whisper_transcription_block import WhisperTranscriptionBlock, WhisperModelSize, OutputFormat
from backend.blocks.blip_captioning_block import BlipImageCaptioningBlock, BlipModelSize, FrameExtractionMethod
from backend.blocks.bark_tts_block import BarkTTSBlock, BarkVoicePreset


class PipelineMode(str, Enum):
    """Available modes for the video captioning pipeline."""
    TRANSCRIPTION_ONLY = "transcription_only"
    VISUAL_CAPTIONING_ONLY = "visual_captioning_only"
    VOICEOVER_ONLY = "voiceover_only"
    TRANSCRIPTION_AND_VOICEOVER = "transcription_and_voiceover"
    VISUAL_CAPTIONING_AND_VOICEOVER = "visual_captioning_and_voiceover"
    COMPLETE_PIPELINE = "complete_pipeline"


class VideoCaptioningPipelineBlock(Block):
    """Block that combines Whisper, BLIP, and Bark for video captioning and voiceover."""

    class Input(BlockSchema):
        video_path: str = SchemaField(
            description="Path to the video file to process"
        )
        pipeline_mode: PipelineMode = SchemaField(
            description="Mode of operation for the pipeline",
            default=PipelineMode.COMPLETE_PIPELINE
        )
        output_dir: str = SchemaField(
            description="Directory to save all output files",
            default="./pipeline_output"
        )
        whisper_model_size: WhisperModelSize = SchemaField(
            description="Whisper model size to use for transcription",
            default=WhisperModelSize.MEDIUM_EN
        )
        blip_model_size: BlipModelSize = SchemaField(
            description="BLIP model size to use for visual captioning",
            default=BlipModelSize.BASE
        )
        frame_extraction_method: FrameExtractionMethod = SchemaField(
            description="Method to use for extracting frames from videos",
            default=FrameExtractionMethod.SCENE_DETECTION
        )
        frames_per_second: float = SchemaField(
            description="Number of frames to extract per second (for regular interval method)",
            default=0.5
        )
        bark_voice_preset: Optional[BarkVoicePreset] = SchemaField(
            description="Voice preset to use for speech generation",
            default=BarkVoicePreset.FEMALE_1
        )
        include_visual_descriptions: bool = SchemaField(
            description="Whether to include visual descriptions in the voiceover",
            default=True
        )
        create_final_video: bool = SchemaField(
            description="Whether to create a final video with the generated voiceover",
            default=True
        )

    class Output(BlockSchema):
        transcription_file: str = SchemaField(
            description="Path to the transcription file"
        )
        caption_file: str = SchemaField(
            description="Path to the visual captions file"
        )
        voiceover_file: str = SchemaField(
            description="Path to the generated voiceover audio file"
        )
        final_video: str = SchemaField(
            description="Path to the final video with voiceover"
        )
        error: str = SchemaField(
            description="Error message if the pipeline fails"
        )

    def __init__(self):
        super().__init__(
            id=str(uuid.uuid4()),
            description="Combines Whisper, BLIP, and Bark for video captioning and voiceover",
            categories={BlockCategory.VIDEO, BlockCategory.AUDIO},
            input_schema=VideoCaptioningPipelineBlock.Input,
            output_schema=VideoCaptioningPipelineBlock.Output,
            test_input={
                "video_path": "test_video.mp4",
                "pipeline_mode": PipelineMode.COMPLETE_PIPELINE,
                "output_dir": "./test_pipeline_output",
                "whisper_model_size": WhisperModelSize.TINY_EN,
                "blip_model_size": BlipModelSize.BASE,
                "frame_extraction_method": FrameExtractionMethod.SCENE_DETECTION,
                "bark_voice_preset": BarkVoicePreset.FEMALE_1
            },
            test_output=[
                ("transcription_file", "./test_pipeline_output/transcription/test_video.srt"),
                ("caption_file", "./test_pipeline_output/captions/captions.json"),
                ("voiceover_file", "./test_pipeline_output/audio/voiceover.wav"),
                ("final_video", "./test_pipeline_output/final_video.mp4")
            ],
        )

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

    def _merge_transcription_and_captions(self, transcription_segments: List[Dict], caption_segments: List[Dict]) -> List[Dict]:
        """Merge transcription and visual caption segments."""
        # Create a timeline of all segments
        all_segments = []
        
        # Add transcription segments
        for segment in transcription_segments:
            all_segments.append({
                "start_time": segment["start"],
                "end_time": segment["end"],
                "text": segment["text"],
                "type": "transcription"
            })
        
        # Add caption segments
        for segment in caption_segments:
            # Find a gap in the transcription to insert the caption
            # or add it at the beginning/end of a nearby transcription segment
            timestamp = segment["timestamp"]
            
            # Find the closest transcription segment
            closest_segment = None
            min_distance = float('inf')
            
            for trans_segment in transcription_segments:
                start_time = self._time_to_seconds(trans_segment["start"])
                end_time = self._time_to_seconds(trans_segment["end"])
                
                # Check if timestamp falls within this segment
                if start_time <= timestamp <= end_time:
                    # Add to beginning or end of segment based on proximity
                    if timestamp - start_time < end_time - timestamp:
                        # Add to beginning
                        all_segments.append({
                            "start_time": self._seconds_to_time(start_time - 2),
                            "end_time": self._seconds_to_time(start_time),
                            "text": f"[{segment['caption']}]",
                            "type": "caption"
                        })
                    else:
                        # Add to end
                        all_segments.append({
                            "start_time": self._seconds_to_time(end_time),
                            "end_time": self._seconds_to_time(end_time + 2),
                            "text": f"[{segment['caption']}]",
                            "type": "caption"
                        })
                    break
                
                # Calculate distance to segment
                if timestamp < start_time:
                    distance = start_time - timestamp
                else:
                    distance = timestamp - end_time
                
                if distance < min_distance:
                    min_distance = distance
                    closest_segment = trans_segment
            
            # If no segment contains the timestamp, add it near the closest segment
            if closest_segment and min_distance < float('inf'):
                start_time = self._time_to_seconds(closest_segment["start"])
                end_time = self._time_to_seconds(closest_segment["end"])
                
                if timestamp < start_time:
                    # Add before the segment
                    all_segments.append({
                        "start_time": self._seconds_to_time(max(0, timestamp - 1)),
                        "end_time": self._seconds_to_time(timestamp + 1),
                        "text": f"[{segment['caption']}]",
                        "type": "caption"
                    })
                else:
                    # Add after the segment
                    all_segments.append({
                        "start_time": self._seconds_to_time(end_time),
                        "end_time": self._seconds_to_time(end_time + 2),
                        "text": f"[{segment['caption']}]",
                        "type": "caption"
                    })
        
        # Sort segments by start time
        all_segments.sort(key=lambda x: self._time_to_seconds(x["start_time"]))
        
        return all_segments

    def _time_to_seconds(self, time_str: str) -> float:
        """Convert time string to seconds."""
        if ',' in time_str:
            # SRT format: 00:00:00,000
            parts = time_str.split(':')
            hours = int(parts[0])
            minutes = int(parts[1])
            seconds_parts = parts[2].split(',')
            seconds = int(seconds_parts[0])
            milliseconds = int(seconds_parts[1])
        else:
            # VTT format: 00:00:00.000
            parts = time_str.split(':')
            hours = int(parts[0])
            minutes = int(parts[1])
            seconds_parts = parts[2].split('.')
            seconds = int(seconds_parts[0])
            milliseconds = int(seconds_parts[1])
        
        return hours * 3600 + minutes * 60 + seconds + milliseconds / 1000

    def _seconds_to_time(self, seconds: float) -> str:
        """Convert seconds to time string in SRT format."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = seconds % 60
        whole_seconds = int(seconds)
        milliseconds = int((seconds - whole_seconds) * 1000)
        
        return f"{hours:02d}:{minutes:02d}:{whole_seconds:02d},{milliseconds:03d}"

    def _create_final_video(self, video_path: str, audio_path: str, output_path: str) -> str:
        """Create a final video with the generated voiceover."""
        # Run FFmpeg to merge video and audio
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", video_path,
                "-i", audio_path,
                "-c:v", "copy",
                "-c:a", "aac",
                "-map", "0:v:0",
                "-map", "1:a:0",
                output_path
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )
        
        return output_path

    def run(self, input_data: Input, **kwargs) -> BlockOutput:
        """Run the video captioning pipeline."""
        try:
            # Check if FFmpeg is installed
            if not self._check_ffmpeg_installed():
                raise RuntimeError(
                    "FFmpeg is not installed. It's required for video processing."
                )
            
            # Create output directories
            os.makedirs(input_data.output_dir, exist_ok=True)
            transcription_dir = os.path.join(input_data.output_dir, "transcription")
            caption_dir = os.path.join(input_data.output_dir, "captions")
            audio_dir = os.path.join(input_data.output_dir, "audio")
            
            os.makedirs(transcription_dir, exist_ok=True)
            os.makedirs(caption_dir, exist_ok=True)
            os.makedirs(audio_dir, exist_ok=True)
            
            # Check if video exists
            video_path = os.path.abspath(input_data.video_path)
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"Video not found: {video_path}")
            
            # Get video filename without extension
            video_filename = os.path.splitext(os.path.basename(video_path))[0]
            
            transcription_file = None
            caption_file = None
            voiceover_file = None
            final_video = None
            
            # Step 1: Transcription with Whisper
            if input_data.pipeline_mode in [
                PipelineMode.TRANSCRIPTION_ONLY,
                PipelineMode.TRANSCRIPTION_AND_VOICEOVER,
                PipelineMode.COMPLETE_PIPELINE
            ]:
                whisper_block = WhisperTranscriptionBlock()
                whisper_input = WhisperTranscriptionBlock.Input(
                    file_path=video_path,
                    model_size=input_data.whisper_model_size,
                    output_format=OutputFormat.SRT,
                    language="en",
                    output_dir=transcription_dir
                )
                
                whisper_results = {}
                for key, value in whisper_block.run(whisper_input):
                    whisper_results[key] = value
                    if key == "transcription_file":
                        transcription_file = value
                
                if "error" in whisper_results:
                    raise RuntimeError(f"Whisper transcription failed: {whisper_results['error']}")
            
            # Step 2: Visual captioning with BLIP
            if input_data.pipeline_mode in [
                PipelineMode.VISUAL_CAPTIONING_ONLY,
                PipelineMode.VISUAL_CAPTIONING_AND_VOICEOVER,
                PipelineMode.COMPLETE_PIPELINE
            ]:
                blip_block = BlipImageCaptioningBlock()
                blip_input = BlipImageCaptioningBlock.Input(
                    video_path=video_path,
                    model_size=input_data.blip_model_size,
                    frame_extraction_method=input_data.frame_extraction_method,
                    frames_per_second=input_data.frames_per_second,
                    output_dir=caption_dir
                )
                
                blip_results = {}
                for key, value in blip_block.run(blip_input):
                    blip_results[key] = value
                    if key == "caption_file":
                        caption_file = value
                
                if "error" in blip_results:
                    raise RuntimeError(f"BLIP captioning failed: {blip_results['error']}")
            
            # Step 3: Generate voiceover with Bark
            if input_data.pipeline_mode in [
                PipelineMode.VOICEOVER_ONLY,
                PipelineMode.TRANSCRIPTION_AND_VOICEOVER,
                PipelineMode.VISUAL_CAPTIONING_AND_VOICEOVER,
                PipelineMode.COMPLETE_PIPELINE
            ]:
                # Prepare text for voiceover
                voiceover_text = ""
                
                if input_data.pipeline_mode == PipelineMode.COMPLETE_PIPELINE:
                    # Combine transcription and captions
                    with open(transcription_file, 'r', encoding='utf-8') as f:
                        transcription_content = f.read()
                    
                    # Parse SRT file
                    import re
                    transcription_segments = []
                    subtitle_blocks = re.split(r'\n\s*\n', transcription_content.strip())
                    
                    for block in subtitle_blocks:
                        lines = block.strip().split('\n')
                        if len(lines) >= 3:
                            # Parse timestamp line
                            timestamp_line = lines[1]
                            timestamp_match = re.match(r'(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})', timestamp_line)
                            
                            if timestamp_match:
                                start_time = timestamp_match.group(1)
                                end_time = timestamp_match.group(2)
                                
                                # Parse text (can be multiple lines)
                                text = ' '.join(lines[2:])
                                
                                transcription_segments.append({
                                    "start": start_time,
                                    "end": end_time,
                                    "text": text
                                })
                    
                    # Load captions
                    with open(caption_file, 'r', encoding='utf-8') as f:
                        caption_segments = json.load(f)
                    
                    # Merge transcription and captions
                    if input_data.include_visual_descriptions:
                        merged_segments = self._merge_transcription_and_captions(
                            transcription_segments, caption_segments
                        )
                        
                        # Create voiceover text
                        for segment in merged_segments:
                            voiceover_text += segment["text"] + " "
                    else:
                        # Use only transcription
                        for segment in transcription_segments:
                            voiceover_text += segment["text"] + " "
                
                elif input_data.pipeline_mode == PipelineMode.TRANSCRIPTION_AND_VOICEOVER:
                    # Use only transcription
                    with open(transcription_file, 'r', encoding='utf-8') as f:
                        transcription_content = f.read()
                    
                    # Parse SRT file to extract text
                    import re
                    text_only = re.sub(r'\d+\n\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}\n', '', transcription_content)
                    text_only = re.sub(r'\n\n', ' ', text_only)
                    
                    voiceover_text = text_only
                
                elif input_data.pipeline_mode == PipelineMode.VISUAL_CAPTIONING_AND_VOICEOVER:
                    # Use only captions
                    with open(caption_file, 'r', encoding='utf-8') as f:
                        caption_segments = json.load(f)
                    
                    # Sort by timestamp
                    caption_segments.sort(key=lambda x: x["timestamp"])
                    
                    # Create voiceover text
                    for segment in caption_segments:
                        voiceover_text += f"[{segment['caption']}] "
                
                # Generate voiceover
                bark_block = BarkTTSBlock()
                bark_input = BarkTTSBlock.Input(
                    text=voiceover_text,
                    voice_preset=input_data.bark_voice_preset,
                    output_dir=audio_dir,
                    output_filename="voiceover"
                )
                
                bark_results = {}
                for key, value in bark_block.run(bark_input):
                    bark_results[key] = value
                    if key == "audio_path":
                        voiceover_file = value
                
                if "error" in bark_results:
                    raise RuntimeError(f"Bark TTS failed: {bark_results['error']}")
            
            # Step 4: Create final video with voiceover
            if input_data.create_final_video and voiceover_file and input_data.pipeline_mode != PipelineMode.TRANSCRIPTION_ONLY:
                final_video_path = os.path.join(input_data.output_dir, f"{video_filename}_with_voiceover.mp4")
                final_video = self._create_final_video(video_path, voiceover_file, final_video_path)
            
            # Return results
            if transcription_file:
                yield "transcription_file", transcription_file
            
            if caption_file:
                yield "caption_file", caption_file
            
            if voiceover_file:
                yield "voiceover_file", voiceover_file
            
            if final_video:
                yield "final_video", final_video
            
        except Exception as e:
            yield "error", f"Pipeline failed: {str(e)}"
