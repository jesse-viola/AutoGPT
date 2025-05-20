from typing import Literal, List
import asyncio
import json
import os
import requests
import uuid
from pathlib import Path

import yt_dlp

from pydantic import SecretStr, Field

from backend.data.block import Block, BlockCategory, BlockOutput, BlockSchema
from backend.data.model import (
    APIKeyCredentials,
    CredentialsField,
    CredentialsMetaInput,
    SchemaField,
)
from backend.integrations.providers import ProviderName
from TikTokApi import TikTokApi


class TikTokCredentials(APIKeyCredentials):
    """TikTok API credentials with verify_fp for authentication."""

    verify_fp: SecretStr


TikTokCredentialsInput = CredentialsMetaInput[
    Literal[ProviderName.TIKTOK], Literal["api_key"]
]


def TikTokCredentialsField() -> TikTokCredentialsInput:
    """Creates a TikTok credentials input on a block."""
    return CredentialsField(
        description="TikTok API credentials with verify_fp for authentication",
    )


# Test credentials for unit testing
TEST_CREDENTIALS = TikTokCredentials(
    id="01234567-89ab-cdef-0123-456789abcdef",
    provider="tiktok",
    type="api_key",
    title="Mock TikTok API credentials",
    api_key=SecretStr("mock-tiktok-api-key"),
    verify_fp=SecretStr("mock-tiktok-verify-fp"),
    expires_at=None,
)

TEST_CREDENTIALS_INPUT = {
    "provider": TEST_CREDENTIALS.provider,
    "id": TEST_CREDENTIALS.id,
    "type": TEST_CREDENTIALS.type,
    "title": TEST_CREDENTIALS.title,
}


class TikTokSearchBlock(Block):
    """Block that searches for TikTok videos based on a query."""

    class Input(BlockSchema):
        credentials: TikTokCredentialsInput = TikTokCredentialsField()
        query: str = SchemaField(description="Search query for TikTok videos")
        max_results: int = SchemaField(
            description="Maximum number of results to return", default=20
        )
        min_likes: int | None = SchemaField(
            description="Minimum number of likes for videos to include", default=None
        )

    class Output(BlockSchema):
        url: str = SchemaField(description="URL of the TikTok video")
        title: str = SchemaField(description="Title/description of the TikTok video")
        likes: int = SchemaField(description="Number of likes on the TikTok video")
        error: str = SchemaField(description="Error message if the search fails")

    def __init__(self):
        super().__init__(
            id="a1b2c3d4-e5f6-7890-a1b2-c3d4e5f67890",
            description="This block searches for TikTok videos based on a query and returns matching videos.",
            categories={BlockCategory.SOCIAL},
            input_schema=TikTokSearchBlock.Input,
            output_schema=TikTokSearchBlock.Output,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "query": "test",
                "max_results": 5,
                "min_likes": 1000,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("url", "https://www.tiktok.com/@user/video/1234567890"),
                ("title", "Test video"),
                ("likes", 5000),
            ],
            test_mock={
                "search_tiktok": lambda *_args, **_kwargs: [
                    {
                        "share_url": "https://www.tiktok.com/@user/video/1234567890",
                        "desc": "Test video",
                        "stats": {"diggCount": 5000},
                    }
                ]
            },
        )

    @staticmethod
    def search_tiktok(credentials: TikTokCredentials, query: str, max_results: int):
        """Search TikTok for videos matching the query.

        This implementation uses a combination of TikTokApi for authentication
        and direct API requests for searching videos.
        """
        try:
            # Use TikTokApi to get a valid session
            async def _get_session_cookies():
                async with TikTokApi() as api:
                    ms_token = credentials.verify_fp.get_secret_value()
                    await api.create_sessions(ms_tokens=[ms_token], num_sessions=1)
                    # Get the session cookies
                    session = api.sessions[0]
                    return dict(
                        session.cookie_jar.filter_cookies("https://www.tiktok.com/")
                    )

            # Run the async function to get cookies
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            cookies = loop.run_until_complete(_get_session_cookies())
            loop.close()

            # Now use the cookies to make a direct search request
            headers = {
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
                "Referer": "https://www.tiktok.com/search?q=" + query,
            }

            # Use TikTok's web search API
            search_url = f"https://www.tiktok.com/api/search/item/full/?aid=1988&app_language=en&app_name=tiktok_web&battery_info=1&browser_language=en-US&browser_name=Mozilla&browser_online=true&browser_platform=MacIntel&browser_version=5.0%20%28Macintosh%3B%20Intel%20Mac%20OS%20X%2010_15_7%29%20AppleWebKit%2F537.36%20%28KHTML%2C%20like%20Gecko%29%20Chrome%2F91.0.4472.124%20Safari%2F537.36&channel=tiktok_web&cookie_enabled=true&device_id=7004716025302476290&device_platform=web_pc&focus_state=true&from_page=search&history_len=3&is_fullscreen=false&is_page_visible=true&keyword={query}&os=mac&priority_region=&referer=&region=US&screen_height=1080&screen_width=1920&tz_name=America%2FNew_York&webcast_language=en&msToken={credentials.verify_fp.get_secret_value()}"

            response = requests.get(search_url, headers=headers, cookies=cookies)
            response.raise_for_status()

            data = response.json()

            # Extract video data from the response
            results = []
            if "item_list" in data and isinstance(data["item_list"], list):
                for item in data["item_list"][:max_results]:
                    if "video" in item:
                        # Format the result to match the expected structure
                        video_dict = {
                            "share_url": f"https://www.tiktok.com/@{item.get('author', {}).get('uniqueId', 'user')}/video/{item.get('id', '')}",
                            "desc": item.get("desc", ""),
                            "stats": {
                                "diggCount": item.get("stats", {}).get("diggCount", 0)
                            },
                        }
                        results.append(video_dict)

            return results
        except Exception as e:
            raise RuntimeError(f"Error searching TikTok: {str(e)}")

    def run(
        self, input_data: Input, *, credentials: TikTokCredentials, **_
    ) -> BlockOutput:
        """Run the TikTok search block."""
        try:
            # Call the search_tiktok method
            results = self.search_tiktok(
                credentials=credentials,
                query=input_data.query,
                max_results=input_data.max_results,
            )

            if not results:
                yield "error", f"No results found for query: {input_data.query}"
                return

            for item in results:
                likes = item["stats"]["diggCount"]
                if input_data.min_likes and likes < input_data.min_likes:
                    continue
                yield "url", item["share_url"]
                yield "title", item["desc"]
                yield "likes", likes

        except Exception as e:
            yield "error", f"Error searching TikTok: {str(e)}"


class TikTokDownloaderBlock(Block):
    """Block that downloads TikTok videos from URLs using yt-dlp."""

    class Input(BlockSchema):
        urls: List[str] = SchemaField(
            description="List of TikTok video URLs to download"
        )
        output_dir: str = SchemaField(
            description="Directory to save downloaded videos", default="./downloads"
        )
        filename_prefix: str = SchemaField(
            description="Prefix for downloaded filenames", default="tiktok_"
        )
        format: str = SchemaField(description="Video format to download", default="mp4")
        no_watermark: bool = SchemaField(
            description="Whether to attempt to download without watermark", default=True
        )

    class Output(BlockSchema):
        file_path: str = SchemaField(description="Path to the downloaded video file")
        video_id: str = SchemaField(description="ID of the downloaded TikTok video")
        error: str = SchemaField(description="Error message if the download fails")

    def __init__(self):
        super().__init__(
            id="b2c3d4e5-f6a7-8901-b2c3-d4e5f6a78901",
            description="This block downloads TikTok videos from URLs without watermark using yt-dlp.",
            categories={BlockCategory.SOCIAL},
            input_schema=TikTokDownloaderBlock.Input,
            output_schema=TikTokDownloaderBlock.Output,
            test_input={
                "urls": ["https://www.tiktok.com/@user/video/1234567890"],
                "output_dir": "./test_downloads",
                "filename_prefix": "test_",
                "format": "mp4",
                "no_watermark": True,
            },
            test_output=[
                ("file_path", "./test_downloads/test_1234567890.mp4"),
                ("video_id", "1234567890"),
            ],
            test_mock={
                "download_tiktok_video": lambda *_args, **_kwargs: {
                    "file_path": "./test_downloads/test_1234567890.mp4",
                    "video_id": "1234567890",
                }
            },
        )

    @staticmethod
    def download_tiktok_video(
        url: str,
        output_dir: str,
        filename_prefix: str,
        format: str = "mp4",
        no_watermark: bool = True,
    ):
        """Download a TikTok video from the given URL using yt-dlp."""
        try:
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)

            # Extract video ID from URL
            video_id = url.split("/video/")[1].split("?")[0]

            # Set up output template
            output_template = os.path.join(
                output_dir, f"{filename_prefix}%(id)s.%(ext)s"
            )

            # Configure yt-dlp options
            ydl_opts = {
                "format": format,
                "outtmpl": output_template,
                "quiet": True,
                "no_warnings": True,
            }

            # If no_watermark is True, try to get the version without watermark
            if no_watermark:
                ydl_opts["format"] = (
                    "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best"
                )
                ydl_opts["postprocessors"] = [
                    {
                        "key": "FFmpegVideoConvertor",
                        "preferedformat": format,
                    }
                ]

            # Download the video
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                file_path = ydl.prepare_filename(info)

                # Ensure the file extension is correct
                if not file_path.endswith(f".{format}"):
                    base_path = os.path.splitext(file_path)[0]
                    file_path = f"{base_path}.{format}"

            return {"file_path": file_path, "video_id": video_id}
        except Exception as e:
            raise RuntimeError(f"Error downloading TikTok video: {str(e)}")

    def run(self, input_data: Input, **_) -> BlockOutput:
        """Run the TikTok downloader block."""
        for url in input_data.urls:
            try:
                # Download the video
                result = self.download_tiktok_video(
                    url=url,
                    output_dir=input_data.output_dir,
                    filename_prefix=input_data.filename_prefix,
                    format=input_data.format,
                    no_watermark=input_data.no_watermark,
                )

                yield "file_path", result["file_path"]
                yield "video_id", result["video_id"]

            except Exception as e:
                yield "error", f"Error downloading video {url}: {str(e)}"


def search_and_download_tiktok_videos(
    credentials: TikTokCredentials,
    query: str,
    max_results: int = 10,
    min_likes: int | None = None,
    output_dir: str = "./downloads",
    filename_prefix: str = "tiktok_",
):
    """
    Helper function to search for TikTok videos and download them.

    This function connects the TikTokSearchBlock and TikTokDownloaderBlock.

    Args:
        credentials: TikTok API credentials
        query: Search query for TikTok videos
        max_results: Maximum number of results to return
        min_likes: Minimum number of likes for videos to include
        output_dir: Directory to save downloaded videos
        filename_prefix: Prefix for downloaded filenames

    Returns:
        List of dictionaries containing file paths and video IDs
    """
    # Create the search block
    search_block = TikTokSearchBlock()

    # Create input data for the search block
    search_input = TikTokSearchBlock.Input(
        credentials=TikTokCredentialsInput(
            provider=ProviderName.TIKTOK,
            id=credentials.id,
            type="api_key",
            title=credentials.title,
        ),
        query=query,
        max_results=max_results,
        min_likes=min_likes,
    )

    # Run the search block
    search_results = {}
    for key, value in search_block.run(search_input, credentials=credentials):
        if key == "url":
            if "urls" not in search_results:
                search_results["urls"] = []
            search_results["urls"].append(value)
        elif key == "error":
            print(f"Search error: {value}")
            return []

    if not search_results.get("urls"):
        print("No videos found matching the search criteria")
        return []

    # Create the downloader block
    downloader_block = TikTokDownloaderBlock()

    # Create input data for the downloader block
    downloader_input = TikTokDownloaderBlock.Input(
        urls=search_results["urls"],
        output_dir=output_dir,
        filename_prefix=filename_prefix,
        format="mp4",
        no_watermark=True,
    )

    # Run the downloader block
    download_results = []
    current_result = {}

    for key, value in downloader_block.run(downloader_input):
        if key == "file_path":
            current_result["file_path"] = value
        elif key == "video_id":
            current_result["video_id"] = value
            # Add the completed result to the list and reset current_result
            download_results.append(current_result.copy())
            current_result = {}
        elif key == "error":
            print(f"Download error: {value}")

    return download_results


class TikTokDownloadBlock(Block):
    """Download a TikTok URL as watermark-free MP4 using yt-dlp."""

    class Input(BlockSchema):
        url: str = SchemaField(description="TikTok video URL")
        output_dir: str = SchemaField(
            description="Directory to save downloaded videos", default="./downloads"
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

    def run(self, input_data: Input, **kwargs) -> BlockOutput:
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
                "postprocessors": [
                    {
                        "key": "FFmpegVideoConvertor",
                        "preferedformat": "mp4",
                    }
                ],
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
