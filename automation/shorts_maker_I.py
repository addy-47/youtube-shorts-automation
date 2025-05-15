# for shorts created using gen ai images

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
from gtts import gTTS
from dotenv import load_dotenv
import shutil # for file operations like moving and deleting files
import tempfile # for creating temporary files
from datetime import datetime # for more detailed time tracking
import concurrent.futures
from functools import wraps
import traceback  # Import traceback at the module level
from helper.minor_helper import measure_time, cleanup_temp_directories
from helper.image import generate_images_parallel, create_image_clips_parallel
from helper.blur import custom_blur, custom_edge_blur
from helper.text import TextHelper
from helper.audio import AudioHelper
from automation.shorts_maker_V import YTShortsCreator_V
from automation.parallel_renderer import render_clips_in_parallel, configure_multiprocessing, SERIALIZER
from automation.sequential_renderer import render_clips_with_threads
import multiprocessing

# from moviepy.config import change_settings
# change_settings({"IMAGEMAGICK_BINARY": "magick"}) # for windows users

# Configure logging for easier debugging
# Do NOT initialize basicConfig here - this will be handled by main.py
logger = logging.getLogger(__name__)

load_dotenv()  # Load environment variables from .env file

# Get temp directory from environment variable or use default
TEMP_DIR = os.getenv("TEMP_DIR", "D:\\youtube-shorts-automation\\temp")
# Ensure temp directory exists
os.makedirs(TEMP_DIR, exist_ok=True)

# We don't need to configure multiprocessing here anymore
# Import and use the configuration from parallel_renderer.py instead
logger.info(f"Using serialization method: {SERIALIZER}")

class YTShortsCreator_I:
    def __init__(self, fps=30):
        """
        Initialize the YouTube Shorts creator with necessary settings

        Args:
            fps (int): Frames per second for the output video
        """
        # Setup directories
        self.temp_dir = os.path.join(TEMP_DIR, f"shorts_i_{int(time.time())}")
        os.makedirs(self.temp_dir, exist_ok=True)

        # Initialize TextHelper
        self.text_helper = TextHelper()

        # Initialize AudioHelper
        self.audio_helper = AudioHelper(self.temp_dir)

        # Check for enhanced rendering capability
        self.has_enhanced_rendering = SERIALIZER != "pickle"
        if self.has_enhanced_rendering:
            logger.info(f"Enhanced parallel rendering available with {SERIALIZER}")
        else:
            logger.info("Basic rendering capability only (using standard pickle)")

        # Video settings
        self.resolution = (1080, 1920)  # Portrait mode for shorts (width, height)
        self.fps = fps
        self.audio_sync_offset = 0.0  # Remove audio delay to improve sync

        # Font settings
        self.fonts_dir = os.path.join(os.path.dirname(__file__), 'fonts')
        os.makedirs(self.fonts_dir, exist_ok=True)
        self.title_font_path = r"D:\youtube-shorts-automation\packages\fonts\default_font.ttf"
        self.body_font_path = r"D:\youtube-shorts-automation\packages\fonts\default_font.ttf"

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

        # Create an instance of YTShortsCreator_V to use its text functions
        self.v_creator = YTShortsCreator_V(fps=fps)

        # Define transition effects with named functions instead of lambdas
        def fade_transition(clip, duration):
            return clip.fadein(duration).fadeout(duration)

        def slide_left_transition(clip, duration):
            def position_func(t):
                return ((t/duration) * self.resolution[0] - clip.w if t < duration else 0, 'center')
            return clip.with_position(position_func)

        def zoom_in_transition(clip, duration):
            def size_func(t):
                return max(1, 1 + 0.5 * min(t/duration, 1))
            return clip.resized(size_func)

        # Define video transition effects between background segments
        def crossfade_transition(clip1, clip2, duration):
            return concatenate_videoclips([
                clip1.with_end(clip1.duration),
                clip2.with_start(0).cross_fadein(duration)
            ], padding=-duration, method="compose")

        def fade_black_transition(clip1, clip2, duration):
            return concatenate_videoclips([
                clip1.fadeout(duration),
                clip2.fadein(duration)
            ])

        # Replace lambda functions with named functions
        self.transitions = {
            "fade": fade_transition,
            "slide_left": slide_left_transition,
            "zoom_in": zoom_in_transition
        }

        # Define video transition effects between background segments
        self.video_transitions = {
            "crossfade": crossfade_transition,
            "fade_black": fade_black_transition
        }

        # Load Pexels API ke for background videos
        self.pexels_api_key = os.getenv("PEXELS_API_KEY")  # for fallback images
        self.huggingface_api_key = os.getenv("HUGGINGFACE_API_KEY")
        self.hf_model = os.getenv("HF_MODEL", "stabilityai/stable-diffusion-2-1")
        self.hf_api_url = f"https://api-inference.huggingface.co/models/{self.hf_model}"
        self.hf_headers = {"Authorization": f"Bearer {self.huggingface_api_key}"}

    @measure_time
    def create_youtube_short(self, title, script_sections, background_query="abstract background",
                            output_filename=None, add_captions=False, style="image", voice_style=None, max_duration=25,
                            background_queries=None, blur_background=False, edge_blur=False, add_watermark_text=None):
        """
        Create a YouTube Short with the given script sections.

        Args:
            title (str): Title of the video
            script_sections (list): List of dict with text, duration, and voice_style
            background_query (str): Search term for background image
            output_filename (str): Output filename, if None one will be generated
            add_captions (bool): If True, add captions to the video
            style (str): Style of the video
            voice_style (str): Voice style from Azure TTS (excited, cheerful, etc)
            max_duration (int): Maximum video duration in seconds
            background_queries (list): Optional list of section-specific background queries
            blur_background (bool): Whether to apply blur effect to background images
            edge_blur (bool): Whether to apply edge blur to background images
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
            from automation.parallel_tasks import ParallelTaskExecutor
            parallel_executor = ParallelTaskExecutor()

            # 2, 3, 4: Run background image generation, audio generation, and text clips generation in parallel
            logger.info("Starting parallel execution of major steps")

            # Define task functions
            def generate_images_task():
                logger.info("Generating images in parallel")
                # Generate image prompts by enhancing background queries with the style
                image_prompts = [f"{query}, {style}" for query in background_queries]
                image_paths = generate_images_parallel(prompts=image_prompts, style=style)

                # Create a dictionary mapping queries to image paths
                images_by_query = {}
                for i, query in enumerate(background_queries):
                    if i < len(image_paths) and image_paths[i]:
                        if query not in images_by_query:
                            images_by_query[query] = []
                        images_by_query[query].append(image_paths[i])

                return images_by_query

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
            parallel_executor.add_task("generate_images", generate_images_task)
            parallel_executor.add_task("generate_audio", generate_audio_task)
            parallel_executor.add_task("generate_text_clips", generate_text_clips_task)

            # Execute all tasks in parallel and wait for results
            results = parallel_executor.execute()

            # Extract results
            images_by_query = results.get("generate_images")
            audio_data = results.get("generate_audio")
            text_clips = results.get("generate_text_clips")

            # 5. Process background images
            logger.info("Processing background images")
            image_paths = []
            durations = []

            for i, section in enumerate(script_sections):
                query = background_queries[i]
                target_duration = section.get('duration', 5)

                # Find the image for this section
                if query in images_by_query and images_by_query[query]:
                    image_path = images_by_query[query][0]
                    image_paths.append(image_path)
                    durations.append(target_duration)

            # Create image clips with zoom effect in parallel
            background_clips = create_image_clips_parallel(
                image_paths=image_paths,
                durations=durations,
                with_zoom=True
            )

            # Apply blur effects if requested
            if blur_background or edge_blur:
                for i, clip in enumerate(background_clips):
                    if blur_background:
                        background_clips[i] = custom_blur(clip, intensity=2)
                    elif edge_blur:
                        background_clips[i] = custom_edge_blur(clip, edge_size=80)

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

            # Use parallel rendering for final video
            logger.info(f"Rendering final video using parallel renderer (serializer: {SERIALIZER})")

            preset = "ultrafast"
            
            # Attempt multi-step rendering approach with fallbacks
            output_success = False
            
            # Step 1: Try process-based parallel rendering (fastest but most complex)
            if not output_success:
                try:
                    # Ensure parallel rendering temp directory exists
                    parallel_render_dir = os.path.join(self.temp_dir, "parallel_render")
                    os.makedirs(parallel_render_dir, exist_ok=True)

                    # Use fewer processes for Windows to avoid handle exhaustion
                    num_processes = max(1, min(multiprocessing.cpu_count() - 1, 3))
                    
                    logger.info(f"Starting parallel rendering with {num_processes} processes")
                    
                    # Pre-render all clips to avoid serialization issues
                    prerender_all = True
                    
                    render_clips_in_parallel(
                        section_clips,
                        output_filename,
                        fps=self.fps,
                        preset=preset,
                        codec="libx264",
                        audio_codec="aac",
                        temp_dir=parallel_render_dir,
                        section_info=section_info,
                        num_processes=num_processes,
                        prerender_all=prerender_all
                    )
                    logger.info(f"Successfully rendered video to {output_filename}")
                    output_success = True
                except Exception as e:
                    logger.error(f"Error in process-based parallel rendering: {e}")
                    logger.error(f"Detailed error: {traceback.format_exc()}")
            
            # Step 2: Try thread-based rendering (slower but more robust)
            if not output_success:
                try:
                    logger.info("Attempting thread-based rendering as fallback...")
                    thread_render_dir = os.path.join(self.temp_dir, "thread_render")
                    os.makedirs(thread_render_dir, exist_ok=True)
                    
                    # Use threads instead of processes (avoids serialization issues)
                    num_threads = min(len(section_clips), 4)  # Limit to 4 threads
                    
                    render_clips_with_threads(
                        section_clips,
                        output_filename,
                        fps=self.fps,
                        preset=preset,
                        codec="libx264",
                        audio_codec="aac",
                        temp_dir=thread_render_dir,
                        section_info=section_info,
                        num_threads=num_threads
                    )
                    logger.info(f"Successfully rendered video with thread-based approach to {output_filename}")
                    output_success = True
                except Exception as e:
                    logger.error(f"Error in thread-based rendering: {e}")
                    logger.error(f"Detailed error: {traceback.format_exc()}")

            # Step 3: Fallback to traditional sequential rendering
            if not output_success:
                logger.info("Falling back to traditional sequential rendering method")
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
                    output_success = True
                    logger.info(f"Successfully rendered video with sequential method to {output_filename}")
                except Exception as fallback_error:
                    logger.error(f"All rendering methods failed. Final error: {fallback_error}")
                    logger.error(f"Detailed fallback error: {traceback.format_exc()}")
                    return None

            # Add watermark as a post-process if requested and not already added in fallback
            if output_success and add_watermark_text and os.path.exists(output_filename):
                logger.info("Adding watermark to final video")
                try:
                    # Load the rendered video
                    final_video = VideoFileClip(output_filename)

                    # Add watermark
                    final_with_watermark = self.text_helper.add_watermark(final_video, watermark_text=add_watermark_text)

                    # Determine watermarked output filename
                    watermarked_output = output_filename.replace('.mp4', '_watermarked.mp4')

                    # Write the watermarked video
                    final_with_watermark.write_videofile(
                        watermarked_output,
                        fps=self.fps,
                        codec="libx264",
                        audio_codec="aac",
                        preset="ultrafast"
                    )

                    # Replace original with watermarked version
                    os.replace(watermarked_output, output_filename)

                    # Clean up
                    final_video.close()
                    final_with_watermark.close()
                except Exception as watermark_error:
                    logger.error(f"Error adding watermark: {watermark_error}")
                    logger.error(f"Detailed watermark error: {traceback.format_exc()}")
            
            return output_filename

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



