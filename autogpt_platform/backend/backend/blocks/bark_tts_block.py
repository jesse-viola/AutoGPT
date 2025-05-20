"""
Bark Text-to-Speech Block for AutoGPT.

This block uses Suno's Bark model to generate realistic speech from text.
"""

import os
import uuid
import json
import tempfile
from enum import Enum
from typing import List, Dict, Optional, Union
import numpy as np
from scipy.io.wavfile import write as write_wav

from backend.data.block import Block, BlockCategory, BlockOutput, BlockSchema
from backend.data.model import SchemaField


class BarkVoicePreset(str, Enum):
    """Available voice presets for Bark."""
    MALE_1 = "v2/en_speaker_0"
    MALE_2 = "v2/en_speaker_1"
    MALE_3 = "v2/en_speaker_2"
    MALE_4 = "v2/en_speaker_3"
    MALE_5 = "v2/en_speaker_4"
    FEMALE_1 = "v2/en_speaker_5"
    FEMALE_2 = "v2/en_speaker_6"
    FEMALE_3 = "v2/en_speaker_7"
    FEMALE_4 = "v2/en_speaker_8"
    FEMALE_5 = "v2/en_speaker_9"


class BarkTTSBlock(Block):
    """Block that generates speech from text using Suno's Bark model."""

    class Input(BlockSchema):
        text: str = SchemaField(
            description="Text to convert to speech"
        )
        voice_preset: Optional[BarkVoicePreset] = SchemaField(
            description="Voice preset to use for speech generation",
            default=None
        )
        output_dir: str = SchemaField(
            description="Directory to save the generated audio files",
            default="./audio"
        )
        output_filename: str = SchemaField(
            description="Filename for the output audio file (without extension)",
            default="bark_output"
        )
        text_segments: Optional[List[str]] = SchemaField(
            description="List of text segments to convert to speech (alternative to text field)",
            default=None
        )
        subtitle_file: Optional[str] = SchemaField(
            description="Path to a subtitle file (SRT or VTT) to convert to speech",
            default=None
        )
        sample_rate: int = SchemaField(
            description="Sample rate for the output audio",
            default=24000
        )

    class Output(BlockSchema):
        audio_path: str = SchemaField(
            description="Path to the generated audio file"
        )
        segment_paths: List[str] = SchemaField(
            description="Paths to the individual audio segment files (if multiple segments)"
        )
        error: str = SchemaField(
            description="Error message if the speech generation fails"
        )

    def __init__(self):
        super().__init__(
            id=str(uuid.uuid4()),
            description="Generates speech from text using Suno's Bark model",
            categories={BlockCategory.AUDIO, BlockCategory.TEXT},
            input_schema=BarkTTSBlock.Input,
            output_schema=BarkTTSBlock.Output,
            test_input={
                "text": "Hello, this is a test of the Bark text-to-speech system.",
                "voice_preset": BarkVoicePreset.FEMALE_1,
                "output_dir": "./test_audio",
                "output_filename": "test_output"
            },
            test_output=[
                ("audio_path", "./test_audio/test_output.wav"),
                ("segment_paths", ["./test_audio/test_output.wav"])
            ],
        )
        self.bark_model = None
        self.sample_rate = None

    def _load_bark_model(self):
        """Load the Bark model."""
        try:
            from bark import SAMPLE_RATE, generate_audio, preload_models
            
            # Preload models
            preload_models()
            
            self.bark_model = generate_audio
            self.sample_rate = SAMPLE_RATE
            
        except ImportError:
            raise ImportError("Bark is not installed. Install it with: pip install git+https://github.com/suno-ai/bark.git")

    def _generate_audio(self, text: str, voice_preset: Optional[str] = None) -> np.ndarray:
        """Generate audio from text using Bark."""
        if self.bark_model is None:
            self._load_bark_model()
        
        # Generate audio
        if voice_preset:
            from bark import generate_audio_with_preset
            audio_array = generate_audio_with_preset(text, voice_preset)
        else:
            audio_array = self.bark_model(text)
        
        return audio_array

    def _save_audio(self, audio_array: np.ndarray, output_path: str, sample_rate: int) -> str:
        """Save audio array to a WAV file."""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save audio
        write_wav(output_path, sample_rate, audio_array)
        
        return output_path

    def _parse_subtitle_file(self, subtitle_path: str) -> List[Dict]:
        """Parse a subtitle file (SRT or VTT) into a list of segments."""
        import re
        
        # Check file extension
        ext = os.path.splitext(subtitle_path)[1].lower()
        
        segments = []
        
        if ext == '.srt':
            # Parse SRT file
            with open(subtitle_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Split into subtitle blocks
            subtitle_blocks = re.split(r'\n\s*\n', content.strip())
            
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
                        
                        segments.append({
                            'start': start_time,
                            'end': end_time,
                            'text': text
                        })
        
        elif ext == '.vtt':
            # Parse VTT file
            with open(subtitle_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Skip header
            if content.startswith('WEBVTT'):
                content = content.split('\n\n', 1)[1]
            
            # Split into subtitle blocks
            subtitle_blocks = re.split(r'\n\s*\n', content.strip())
            
            for block in subtitle_blocks:
                lines = block.strip().split('\n')
                if len(lines) >= 2:
                    # Parse timestamp line
                    for i, line in enumerate(lines):
                        timestamp_match = re.match(r'(\d{2}:\d{2}:\d{2}\.\d{3}) --> (\d{2}:\d{2}:\d{2}\.\d{3})', line)
                        if timestamp_match:
                            start_time = timestamp_match.group(1)
                            end_time = timestamp_match.group(2)
                            
                            # Parse text (can be multiple lines)
                            text = ' '.join(lines[i+1:])
                            
                            segments.append({
                                'start': start_time,
                                'end': end_time,
                                'text': text
                            })
                            break
        
        return segments

    def _concatenate_audio(self, audio_paths: List[str], output_path: str) -> str:
        """Concatenate multiple audio files into a single file."""
        try:
            import pydub
            
            # Load audio segments
            audio_segments = [pydub.AudioSegment.from_wav(path) for path in audio_paths]
            
            # Concatenate segments
            combined = audio_segments[0]
            for segment in audio_segments[1:]:
                combined += segment
            
            # Export combined audio
            combined.export(output_path, format="wav")
            
            return output_path
            
        except ImportError:
            raise ImportError("pydub is not installed. Install it with: pip install pydub")

    def run(self, input_data: Input, **kwargs) -> BlockOutput:
        """Run the Bark TTS block."""
        try:
            # Create output directory if it doesn't exist
            os.makedirs(input_data.output_dir, exist_ok=True)
            
            # Determine text segments to process
            text_segments = []
            
            if input_data.subtitle_file:
                # Parse subtitle file
                subtitle_path = os.path.abspath(input_data.subtitle_file)
                if not os.path.exists(subtitle_path):
                    raise FileNotFoundError(f"Subtitle file not found: {subtitle_path}")
                
                subtitle_segments = self._parse_subtitle_file(subtitle_path)
                text_segments = [segment['text'] for segment in subtitle_segments]
                
            elif input_data.text_segments:
                # Use provided text segments
                text_segments = input_data.text_segments
                
            elif input_data.text:
                # Use single text input
                # Split into smaller chunks if longer than ~13 seconds of speech
                # (roughly 30 words per chunk)
                words = input_data.text.split()
                if len(words) > 30:
                    # Split into chunks of about 30 words
                    chunks = []
                    for i in range(0, len(words), 30):
                        chunk = ' '.join(words[i:i+30])
                        chunks.append(chunk)
                    text_segments = chunks
                else:
                    text_segments = [input_data.text]
            
            if not text_segments:
                raise ValueError("No text provided for speech generation")
            
            # Generate audio for each segment
            segment_paths = []
            
            for i, segment in enumerate(text_segments):
                # Generate audio
                audio_array = self._generate_audio(segment, input_data.voice_preset)
                
                # Save segment
                segment_filename = f"{input_data.output_filename}_segment_{i:03d}.wav"
                segment_path = os.path.join(input_data.output_dir, segment_filename)
                self._save_audio(audio_array, segment_path, input_data.sample_rate)
                
                segment_paths.append(segment_path)
            
            # If there's only one segment, use it as the final output
            if len(segment_paths) == 1:
                final_audio_path = segment_paths[0]
            else:
                # Concatenate all segments
                final_audio_path = os.path.join(input_data.output_dir, f"{input_data.output_filename}.wav")
                self._concatenate_audio(segment_paths, final_audio_path)
            
            yield "audio_path", final_audio_path
            yield "segment_paths", segment_paths
            
        except Exception as e:
            yield "error", f"Error generating speech: {str(e)}"
