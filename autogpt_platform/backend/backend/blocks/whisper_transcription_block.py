"""
Whisper Transcription Block for AutoGPT.

This block uses OpenAI's Whisper model to transcribe speech from video or audio files.
"""

import os
import uuid
import tempfile
import subprocess
from enum import Enum
from pathlib import Path
from typing import List, Optional

from backend.data.block import Block, BlockCategory, BlockOutput, BlockSchema
from backend.data.model import SchemaField


class WhisperModelSize(str, Enum):
    """Available Whisper model sizes."""
    TINY = "tiny"
    BASE = "base"
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"
    
    # Language-specific models
    TINY_EN = "tiny.en"
    BASE_EN = "base.en"
    SMALL_EN = "small.en"
    MEDIUM_EN = "medium.en"


class OutputFormat(str, Enum):
    """Available output formats for Whisper transcription."""
    TXT = "txt"
    VTT = "vtt"
    SRT = "srt"
    TSV = "tsv"
    JSON = "json"
    ALL = "all"


class WhisperTranscriptionBlock(Block):
    """Block that transcribes speech from video or audio files using OpenAI's Whisper model."""

    class Input(BlockSchema):
        file_path: str = SchemaField(
            description="Path to the video or audio file to transcribe"
        )
        model_size: WhisperModelSize = SchemaField(
            description="Whisper model size to use for transcription",
            default=WhisperModelSize.MEDIUM_EN
        )
        output_format: OutputFormat = SchemaField(
            description="Output format for the transcription",
            default=OutputFormat.SRT
        )
        language: Optional[str] = SchemaField(
            description="Language code (e.g., 'en' for English). If None, Whisper will auto-detect.",
            default=None
        )
        output_dir: str = SchemaField(
            description="Directory to save the transcription files",
            default="./transcriptions"
        )

    class Output(BlockSchema):
        transcription_file: str = SchemaField(
            description="Path to the transcription file"
        )
        text: str = SchemaField(
            description="The transcribed text (if available)"
        )
        error: str = SchemaField(
            description="Error message if the transcription fails"
        )

    def __init__(self):
        super().__init__(
            id=str(uuid.uuid4()),
            description="Transcribes speech from video or audio files using OpenAI's Whisper model",
            categories={BlockCategory.AUDIO, BlockCategory.VIDEO},
            input_schema=WhisperTranscriptionBlock.Input,
            output_schema=WhisperTranscriptionBlock.Output,
            test_input={
                "file_path": "test_video.mp4",
                "model_size": WhisperModelSize.TINY_EN,
                "output_format": OutputFormat.SRT,
                "language": "en",
                "output_dir": "./test_transcriptions"
            },
            test_output=[
                ("transcription_file", "./test_transcriptions/test_video.srt"),
                ("text", "This is a test transcription.")
            ],
        )

    def _check_whisper_installed(self) -> bool:
        """Check if Whisper is installed."""
        try:
            subprocess.run(
                ["whisper", "--help"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False
            )
            return True
        except FileNotFoundError:
            return False

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

    def _extract_audio(self, video_path: str) -> str:
        """Extract audio from video file."""
        temp_dir = tempfile.gettempdir()
        audio_path = os.path.join(temp_dir, f"{uuid.uuid4()}.wav")
        
        subprocess.run(
            ["ffmpeg", "-i", video_path, "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", audio_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )
        
        return audio_path

    def _read_text_file(self, file_path: str) -> str:
        """Read text from a file."""
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    def run(self, input_data: Input, **kwargs) -> BlockOutput:
        """Run the Whisper transcription block."""
        try:
            # Check if Whisper and FFmpeg are installed
            if not self._check_whisper_installed():
                raise RuntimeError(
                    "Whisper is not installed. Install it with: pip install openai-whisper"
                )
            
            if not self._check_ffmpeg_installed():
                raise RuntimeError(
                    "FFmpeg is not installed. It's required for audio processing."
                )
            
            # Create output directory if it doesn't exist
            os.makedirs(input_data.output_dir, exist_ok=True)
            
            # Get file information
            file_path = os.path.abspath(input_data.file_path)
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            
            file_name = os.path.basename(file_path)
            file_base = os.path.splitext(file_name)[0]
            
            # Build command
            cmd = ["whisper", file_path, "--model", input_data.model_size]
            
            # Add output format
            cmd.extend(["--output_format", input_data.output_format])
            
            # Add output directory
            cmd.extend(["--output_dir", input_data.output_dir])
            
            # Add language if specified
            if input_data.language:
                cmd.extend(["--language", input_data.language])
            
            # Run Whisper
            process = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True
            )
            
            # Get output file path
            output_file = os.path.join(
                input_data.output_dir, 
                f"{file_base}.{input_data.output_format}"
            )
            
            # Read text content if available
            text_content = ""
            if input_data.output_format in [OutputFormat.TXT, OutputFormat.SRT, OutputFormat.VTT]:
                text_content = self._read_text_file(output_file)
            
            yield "transcription_file", output_file
            yield "text", text_content
            
        except Exception as e:
            yield "error", f"Error transcribing file: {str(e)}"
