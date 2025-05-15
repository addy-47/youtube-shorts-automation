import os # for file operations
import time # for timing events and creating filenames like timestamps
import random # for randomizing elements
import textwrap # for wrapping text into lines but most cases being handled by textclip class in moviepy
import requests # for making HTTP requests
import numpy as np # for numerical operations here used for rounding off
import logging # for logging events
from PIL import Image, ImageFilter, ImageDraw, ImageFont# for image processing
from moviepy  import ( # for video editing
    VideoFileClip, VideoClip, TextClip, CompositeVideoClip,ImageClip,
    AudioFileClip, concatenate_videoclips, ColorClip, CompositeAudioClip, concatenate_audioclips
)
from moviepy.video.fx import *
# from moviepy.config import change_settings
# change_settings({"IMAGEMAGICK_BINARY": "magick"}) # for windows users
from gtts import gTTS
from dotenv import load_dotenv
import shutil # for file operations like moving and deleting files
import tempfile # for creating temporary files
from datetime import datetime # for more detailed time tracking
import concurrent.futures
from functools import wraps
import traceback  # Import traceback at the module level
from helper.minor_helper import measure_time, cleanup_temp_directories
from helper.fetch import fetch_videos_parallel
from helper.blur import custom_blur, custom_edge_blur
from helper.text import TextHelper
from helper.process import process_background_clips_parallel
from helper.audio import AudioHelper
from automation.parallel_tasks import ParallelTaskExecutor
from automation.renderer import render_video
import multiprocessing

# Configure logging for easier debugging
# Do NOT initialize basicConfig here - this will be handled by main.py
logger = logging.getLogger(__name__)

load_dotenv()  # Load environment variables from .env file

# Get temp directory from environment variable or use default
TEMP_DIR = os.getenv("TEMP_DIR", "D:\\youtube-shorts-automation\\temp")
# Ensure temp directory exists
os.makedirs(TEMP_DIR, exist_ok=True)

class YTShortsCreator_V:
    def __init__(self, fps=30):
        """
        Initialize the YouTube Shorts creator with necessary settings

        Args:
            fps (int): Frames per second for the output video
        """
        # Setup directories
        self.temp_dir = os.path.join(TEMP_DIR, f"shorts_v_{int(time.time())}")
        os.makedirs(self.temp_dir, exist_ok=True)

        # Initialize TextHelper
        self.text_helper = TextHelper()

        # Initialize AudioHelper
        self.audio_helper = AudioHelper(self.temp_dir)

        # Video settings
        self.resolution = (1080, 1920)  # Portrait mode for shorts (width, height)
        self.fps = fps
        self.audio_sync_offset = 0.0  # Remove audio delay to improve sync

        # Initialize TTS (Text-to-Speech)
        self.azure_tts = None
        self.google_tts = None

        # Initialize Google Cloud TTS
        if os.getenv("USE_GOOGLE_TTS", "true").lower() == "true":
            try:
                from automation.voiceover import GoogleVoiceover
                self.google_tts = GoogleVoiceover(
                    voice=os.getenv("GOOGLE_VOICE", "en-US-Neural2-D"),
                    output_dir=self.temp_dir
                )
                logger.info("Google Cloud TTS initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize Google Cloud TTS: {e}. Will use gTTS instead.")

        # Initialize Azure TTS as fallback (if configured)
        elif os.getenv("USE_AZURE_TTS", "false").lower() == "true":
            try:
                from automation.voiceover_azure import AzureVoiceover
                self.azure_tts = AzureVoiceover(
                    voice=os.getenv("AZURE_VOICE", "en-US-JennyNeural"),
                    output_dir=self.temp_dir
                )
                logger.info("Azure TTS initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize Azure TTS: {e}. Will use gTTS instead.")

    @measure_time
    def create_youtube_short(self, title, script_sections, background_query="abstract background",
                            output_filename=None, add_captions=False, style="video", voice_style=None, max_duration=25,
                            background_queries=None, blur_background=False, edge_blur=False, add_watermark_text=None):
        """
        Create a YouTube Short with the given script sections.

        Args:
            title (str): Title of the video
            script_sections (list): List of dict with text, duration, and voice_style
            background_query (str): Search term for background video
            output_filename (str): Output filename, if None one will be generated
            add_captions (bool): If True, add captions to the video
            style (str): Style of the video
            voice_style (str): Voice style from Azure TTS (excited, cheerful, etc)
            max_duration (int): Maximum video duration in seconds
            background_queries (list): Optional list of section-specific background queries
            blur_background (bool): Whether to apply blur effect to background videos
            edge_blur (bool): Whether to apply edge blur to background videos
            add_watermark_text (str): Text to use as watermark (None for no watermark)

        Returns:
            str: Output file path
        """
        try:
            if not output_filename:
                date_str = datetime.now().strftime('%Y%m%d_%H%M%S')
                output_filename = os.path.join(self.temp_dir, f"short_{date_str}.mp4")

            # Get total duration from script sections
            total_raw_duration = sum(section.get('duration', 5) for section in script_sections)
            duration_scaling_factor = min(1.0, max_duration / total_raw_duration) if total_raw_duration > max_duration else 1.0

            # Scale durations if needed to fit max time
            if duration_scaling_factor < 1.0:
                logger.info(f"Scaling durations by factor {duration_scaling_factor:.2f} to fit max duration of {max_duration}s")
                for section in script_sections:
                    section['duration'] = section['duration'] * duration_scaling_factor

            # Add unique IDs to sections if they don't have them
            for i, section in enumerate(script_sections):
                if 'id' not in section:
                    section['id'] = f"section_{i}_{int(time.time())}"

            # 1. Prepare background queries
            if not background_queries:
                background_queries = [background_query] * len(script_sections)
            elif len(background_queries) != len(script_sections):
                # Pad or truncate the queries list
                if len(background_queries) < len(script_sections):
                    background_queries.extend([background_query] * (len(script_sections) - len(background_queries)))
                else:
                    background_queries = background_queries[:len(script_sections)]

            # Create a parallel task executor to run major steps concurrently
            parallel_executor = ParallelTaskExecutor()

            # 2, 3, 4: Run background fetching, audio generation, and text clips generation in parallel
            logger.info("Starting parallel execution of major steps")

            # Define task functions
            def fetch_videos_task():
                logger.info("Fetching background videos in parallel")
                return fetch_videos_parallel(
                    queries=background_queries,
                    count_per_query=1,
                    min_duration=int(max(section.get('duration', 5) for section in script_sections)) + 2
                )

            def generate_audio_task():
                logger.info("Generating audio clips in parallel")
                return self.audio_helper.process_audio_for_script(
                    script_sections=script_sections,
                    voice_style=voice_style
                )

            def generate_text_clips_task():
                logger.info("Generating text clips in parallel")
                if add_captions:
                    return self.text_helper.generate_word_by_word_clips_parallel(
                        script_sections=script_sections
                    )
                else:
                    return self.text_helper.generate_text_clips_parallel(
                        script_sections=script_sections,
                        with_pill=True
                    )

            # Add tasks to executor
            parallel_executor.add_task("fetch_videos", fetch_videos_task)
            parallel_executor.add_task("generate_audio", generate_audio_task)
            parallel_executor.add_task("generate_text_clips", generate_text_clips_task)

            # Execute all tasks in parallel and wait for results
            results = parallel_executor.execute()

            # Extract results
            videos_by_query = results.get("fetch_videos", {})
            audio_data = results.get("generate_audio", [])
            text_clips = results.get("generate_text_clips", [])

            # Check if we have necessary components before continuing
            if not videos_by_query:
                logger.error("No background videos fetched")
                return None

            # 5. Process background videos
            logger.info("Processing background videos")
            video_paths = []
            
            # Make background clips from videos
            for i, section in enumerate(script_sections):
                query = background_queries[i]
                target_duration = section.get('duration', 5)
                
                # Find the video for this section
                if query in videos_by_query and videos_by_query[query]:
                    video_paths.append({
                        'path': videos_by_query[query][0],
                        'section_idx': i,
                        'duration': target_duration,
                        'query': query
                    })

            # Process videos in parallel
            background_clips = process_background_clips_parallel(
                video_info=video_paths,
                blur=blur_background,
                edge_blur=edge_blur,
                output_resolution=self.resolution
            )
            
            # Make sure we have all the components
            if not background_clips:
                logger.error("No background clips generated")
                return None

            if not audio_data:
                logger.error("No audio generated")
                return None

            if not text_clips:
                logger.warning("No text clips generated")

            # Create section clips (background + audio + text)
            section_clips = []
            section_info = {}  # For better logging in parallel renderer

            for i, (bg_clip, audio, text_clip) in enumerate(zip(background_clips, audio_data, text_clips)):
                # Add text clip to background if available
                if text_clip:
                    composite = CompositeVideoClip([bg_clip, text_clip])
                else:
                    composite = bg_clip

                # Add audio
                if audio:
                    composite = composite.with_audio(AudioFileClip(audio['path']))

                # Add debugging info to the clip
                section_text = script_sections[i].get('text', '')[:30] + '...' if len(script_sections[i].get('text', '')) > 30 else script_sections[i].get('text', '')
                composite._debug_info = f"Section {i}: {section_text}"
                composite._section_idx = i

                # Store section information
                section_info[i] = {
                    'section_idx': i,
                    'section_text': section_text,
                    'duration': script_sections[i].get('duration', 5)
                }

                section_clips.append(composite)

            # Use our unified renderer
            logger.info("Rendering final video using optimized renderer")
            
            # Ensure rendering temp directory exists
            render_temp_dir = os.path.join(self.temp_dir, "render")
            os.makedirs(render_temp_dir, exist_ok=True)
            
            # Use the unified rendering interface
            output_path = render_video(
                clips=section_clips,
                output_file=output_filename,
                fps=self.fps,
                temp_dir=render_temp_dir,
                preset="ultrafast",
                parallel=True,
                memory_per_worker_gb=2.0,
                options={
                    'clean_temp': True,
                    'section_info': section_info
                }
            )
            
            logger.info(f"Successfully rendered video to {output_path}")

            # Add watermark if requested
            if add_watermark_text and os.path.exists(output_path):
                logger.info("Adding watermark to final video")
                try:
                    # Load the rendered video
                    final_video = VideoFileClip(output_path)

                    # Add watermark
                    final_with_watermark = self.text_helper.add_watermark(final_video, watermark_text=add_watermark_text)

                    # Determine watermarked output filename
                    watermarked_output = output_path.replace('.mp4', '_watermarked.mp4')

                    # Write the watermarked video
                    final_with_watermark.write_videofile(
                        watermarked_output,
                        fps=self.fps,
                        codec="libx264",
                        audio_codec="aac",
                        preset="ultrafast"
                    )

                    # Replace original with watermarked version
                    os.replace(watermarked_output, output_path)

                    # Clean up
                    final_video.close()
                    final_with_watermark.close()
                except Exception as watermark_error:
                    logger.error(f"Error adding watermark: {watermark_error}")
                    logger.error(f"Detailed watermark error: {traceback.format_exc()}")

            return output_path

        except Exception as e:
            logger.error(f"Error creating video: {e}")
            logger.error(f"Detailed error trace: {traceback.format_exc()}")
            # If we encounter an error, try to clean up temp files
            cleanup_temp_directories([self.temp_dir])
            return None

    def cleanup(self):
        """Clean up temporary files"""
        try:
            cleanup_temp_directories([self.temp_dir])
            logger.info(f"Cleaned up temporary files in {self.temp_dir}")
        except Exception as e:
            logger.error(f"Error cleaning up temporary files: {e}")



