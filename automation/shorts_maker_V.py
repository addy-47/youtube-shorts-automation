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
from helper.minor_helper import measure_time, cleanup_temp_directories
from helper.fetch import fetch_videos_parallel
from helper.blur import custom_blur, custom_edge_blur
from helper.text import TextHelper
from helper.process import process_background_clips_parallel
from helper.audio import AudioHelper
from automation.parallel_tasks import ParallelTaskExecutor
from automation.parallel_renderer import render_clips_in_parallel
import dill

# Configure logging for easier debugging
# Do NOT initialize basicConfig here - this will be handled by main.py
logger = logging.getLogger(__name__)

load_dotenv()  # Load environment variables from .env file

# Get temp directory from environment variable or use default
TEMP_DIR = os.getenv("TEMP_DIR", "D:\\youtube-shorts-automation\\temp")
# Ensure temp directory exists
os.makedirs(TEMP_DIR, exist_ok=True)

# Check if dill supports advanced serialization
try:
    import dill
    dill_version = dill.__version__
    logger.info(f"Enhanced parallel rendering available with dill {dill_version}")

    # Configure dill for multiprocessing - correct approach
    try:
        import multiprocessing
        # Replace the default pickle methods with dill methods
        multiprocessing.reduction.dump = dill.dump
        multiprocessing.reduction.dumps = dill.dumps
        multiprocessing.reduction.load = dill.load
        multiprocessing.reduction.loads = dill.loads
        logger.info("Successfully configured dill for multiprocessing")
    except (ImportError, AttributeError) as e:
        logger.warning(f"Could not fully configure dill as the default serializer: {e}")
except ImportError:
    logger.warning("Advanced serialization unavailable - dill not installed")

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

        # Check for enhanced rendering capability
        self.has_enhanced_rendering = False
        try:
            import dill
            self.has_enhanced_rendering = True
            logger.info(f"Enhanced parallel rendering available with dill {dill.__version__}")
        except ImportError:
            logger.info("Basic rendering capability only (install dill for enhanced parallel rendering)")

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

            if not audio_data:
                logger.error("No audio generated")
                return None

            # Print what we got for debugging
            logger.info(f"Fetched videos for {len(videos_by_query)} queries")
            logger.info(f"Generated {len(audio_data)} audio clips")
            logger.info(f"Generated {len(text_clips)} text clips")

            # 5. Process background videos
            logger.info("Processing background videos")
            clip_info_list = []
            for i, section in enumerate(script_sections):
                if i >= len(background_queries):
                    logger.warning(f"Missing background query for section {i}")
                    continue

                query = background_queries[i]
                target_duration = section.get('duration', 5)

                # Find the video for this section
                if query in videos_by_query and videos_by_query[query]:
                    video_path = videos_by_query[query][0]
                    try:
                        clip = VideoFileClip(video_path)
                        clip_info_list.append({
                            'clip': clip,
                            'target_duration': target_duration
                        })
                    except Exception as e:
                        logger.error(f"Error loading video {video_path}: {e}")

            if not clip_info_list:
                logger.error("No videos could be loaded for processing")
                return None

            # Process backgrounds in parallel
            background_clips = process_background_clips_parallel(
                clip_info_list=clip_info_list,
                blur_background=blur_background,
                edge_blur=edge_blur
            )

            # 6. Combine everything into the final video
            logger.info("Assembling final video")

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

                # Add debugging info to the clip for parallel renderer
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

            # Instead of using concatenate_videoclips and write_videofile, use parallel rendering
            logger.info(f"Rendering final video using parallel renderer")

            preset = "ultrafast"
            # Use render_clips_in_parallel instead of concatenate and write_videofile
            try:
                # Ensure parallel rendering temp directory exists
                parallel_render_dir = os.path.join(self.temp_dir, "parallel_render")
                os.makedirs(parallel_render_dir, exist_ok=True)

                render_clips_in_parallel(
                    section_clips,
                    output_filename,
                    fps=self.fps,
                    preset=preset,
                    codec="libx264",
                    audio_codec="aac",
                    temp_dir=parallel_render_dir,
                    section_info=section_info
                )
                logger.info(f"Successfully rendered video to {output_filename}")
            except Exception as e:
                logger.error(f"Error in parallel rendering: {e}")

                # Fallback to traditional rendering if parallel rendering fails
                logger.info("Falling back to traditional rendering method")
                try:
                    # Concatenate all section clips
                    final_video = concatenate_videoclips(section_clips)

                    # Add watermark if requested
                    if add_watermark_text:
                        final_video = self.text_helper.add_watermark(final_video, watermark_text=add_watermark_text)

                    # Write the final video with standard method
                    final_video.write_videofile(
                        output_filename,
                        fps=self.fps,
                        codec="libx264",
                        audio_codec="aac",
                        preset=preset
                    )
                    final_video.close()
                except Exception as fallback_error:
                    logger.error(f"Fallback rendering also failed: {fallback_error}")
                    return None

            return output_filename

        except Exception as e:
            logger.error(f"Error creating video: {e}")
            # If we encounter an error, try to clean up temp files
            cleanup_temp_directories([self.temp_dir])

    def cleanup(self):
        """Clean up temporary files"""
        try:
            cleanup_temp_directories([self.temp_dir])
            logger.info(f"Cleaned up temporary files in {self.temp_dir}")
        except Exception as e:
            logger.error(f"Error cleaning up temporary files: {e}")



