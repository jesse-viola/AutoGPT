"""
TikTok Download Block for AutoGPT.

This block downloads TikTok videos without watermark using yt-dlp.
"""

import os
import uuid
from pathlib import Path

import yt_dlp

from backend.data.block import Block, BlockCategory, BlockOutput, BlockSchema
from backend.data.model import SchemaField


class TikTokDownloadBlock(Block):
    """Download a TikTok URL as watermark-free MP4 using yt-dlp."""

    class Input(BlockSchema):
        url: str = SchemaField(description="TikTok video URL")
        output_dir: str = SchemaField(
            description="Directory to save downloaded videos",
            default="./downloads"
        )

    class Output(BlockSchema):
        file_path: str = SchemaField(description="Local path of downloaded MP4")
        error: str = SchemaField(description="Error message if download fails")

    def __init__(self):
        super().__init__(
            id=str(uuid.uuid4()),
            description="Download TikTok video without watermark",
            categories={BlockCategory.SOCIAL},
            input_schema=TikTokDownloadBlock.Input,
            output_schema=TikTokDownloadBlock.Output,
            test_input={"url": "https://tiktok.com/@test/video/123456"},
            test_output=[("file_path", str)],
        )

    def run(self, input_data: Input, **_) -> BlockOutput:
        """Run the TikTok download block."""
        url = input_data.url
        outdir = os.path.abspath(input_data.output_dir)
        os.makedirs(outdir, exist_ok=True)
        outtmpl = os.path.join(outdir, "%(id)s.%(ext)s")

        try:
            # Configure yt-dlp options for best quality without watermark
            ydl_opts = {
                "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
                "outtmpl": outtmpl,
                "quiet": True,
                "no_warnings": True,
                "postprocessors": [{
                    "key": "FFmpegVideoConvertor",
                    "preferedformat": "mp4",
                }]
            }
            
            # Download the video
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                file_path = ydl.prepare_filename(info)
                
                # Ensure the file extension is correct
                if not file_path.endswith(".mp4"):
                    base_path = os.path.splitext(file_path)[0]
                    file_path = f"{base_path}.mp4"
                    
            yield "file_path", file_path
        except Exception as e:
            yield "error", f"Error downloading TikTok video: {str(e)}"
