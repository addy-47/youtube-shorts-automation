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
from automation.renderer import render_video
import multiprocessing

# from moviepy.config import change_settings
# change_settings({"IMAGEMAGICK_BINARY": "magick"}) # for windows users

# Configure logging for easier debugging
# Do NOT initialize basicConfig here - this will be handled by main.py
logger = logging.getLogger(__name__)

load_dotenv()  # Load environment variables from .env file

# Get temp directory from environment variable or use default
TEMP_DIR = os.getenv("TEMP_DIR", os.path.join(os.path.dirname(os.path.dirname(__file__)), "temp"))
# Ensure temp directory exists
os.makedirs(TEMP_DIR, exist_ok=True)

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

        # Video settings
        self.resolution = (1080, 1920)  # Portrait mode for shorts (width, height)
        self.fps = fps
        self.audio_sync_offset = 0.0  # Remove audio delay to improve sync

        # Font settings
        self.fonts_dir = os.path.join(os.path.dirname(__file__), 'fonts')
        os.makedirs(self.fonts_dir, exist_ok=True)
        self.title_font_path = r"/home/addy/projects/youtube-shorts-automation/packages/fonts/default_font.ttf"
        self.body_font_path = r"/home/addy/projects/youtube-shorts-automation/packages/fonts/default_font.ttf"

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

            # 2, 3: Run background image generation and audio generation in parallel
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

            # Add tasks to executor
            parallel_executor.add_task("generate_images", generate_images_task)
            parallel_executor.add_task("generate_audio", generate_audio_task)

            # Execute all tasks in parallel and wait for results
            results = parallel_executor.execute()

            # Extract results
            images_by_query = results.get("generate_images")
            audio_data = results.get("generate_audio")

            # Check if we have enough images or need to fallback to video mode
            should_fallback = False
            
            # Case 1: No images generated at all
            if not images_by_query or all(len(images) == 0 for images in images_by_query.values()):
                logger.warning("⚠️ FALLBACK: All image generation methods failed")
                should_fallback = True
            
            # Case 2: Count how many sections have corresponding images
            else:
                # Count how many sections will have a valid image
                sections_with_images = 0
                for i, query in enumerate(background_queries):
                    if query in images_by_query and images_by_query[query]:
                        sections_with_images += 1
                
                # If we have less than 70% of required images, fallback to video mode
                if sections_with_images < 0.7 * len(script_sections):
                    logger.warning(f"⚠️ FALLBACK: Only {sections_with_images}/{len(script_sections)} sections have images ({int(sections_with_images/len(script_sections)*100)}%)")
                    should_fallback = True
            
            # If fallback is triggered, switch to video mode
            if should_fallback:
                logger.warning("⚠️ SWITCHING to shorts_maker_V to create video with stock videos instead of AI images")
                
                # We've already generated audio, so pass it to shorts_maker_V to avoid regenerating
                logger.info("Creating video using shorts_maker_V with the same script sections and pre-generated audio")
                
                # Create a video version using the pre-generated audio
                video_creator = self.v_creator
                
                # Only difference is we use video style instead of image style
                return video_creator.create_youtube_short(
                    title=title,
                    script_sections=script_sections,
                    background_query=background_query,
                    output_filename=output_filename,
                    add_captions=add_captions,
                    style="video",  # Switch to video style
                    voice_style=voice_style,
                    max_duration=max_duration,
                    background_queries=background_queries,
                    blur_background=blur_background,
                    edge_blur=edge_blur,
                    add_watermark_text=add_watermark_text,
                    existing_audio_data=audio_data  # Pass the audio we already generated
                )

            # First, check audio durations to use them as source of truth
            audio_durations = {}
            for i, audio_section in enumerate(audio_data):
                expected_duration = script_sections[i].get('duration', 5)
                if audio_section and 'path' in audio_section:
                    try:
                        audio_clip = AudioFileClip(audio_section['path'])
                        actual_duration = audio_clip.duration
                        audio_clip.close()
                        
                        # Store actual audio duration for this section
                        audio_durations[i] = actual_duration
                        
                        # Update the script section duration to match audio
                        script_sections[i]['duration'] = actual_duration
                        
                        # Log mismatches for debugging
                        if abs(actual_duration - expected_duration) > 0.5:
                            logger.info(f"Section {i} audio duration ({actual_duration:.2f}s) will be used instead of script duration ({expected_duration:.2f}s)")
                    except Exception as e:
                        logger.error(f"Error checking audio duration for section {i}: {e}")
                        audio_durations[i] = expected_duration
                else:
                    audio_durations[i] = expected_duration
            
            # Now generate all text clips with the correct audio durations
            logger.info("Generating all text clips using accurate audio durations")
            
            # Separate script sections for different text rendering styles
            intro_sections = [script_sections[0]] if len(script_sections) > 0 else []
            middle_sections = script_sections[1:-1] if len(script_sections) > 2 else []
            outro_sections = [script_sections[-1]] if len(script_sections) > 1 else []
            
            # Generate standard text clips for intro with correct duration
            intro_clips = []
            if intro_sections:
                logger.info(f"Generating intro text clip with duration: {intro_sections[0]['duration']:.2f}s")
                intro_clips = self.text_helper.generate_text_clips_parallel(
                    script_sections=intro_sections,
                    with_pill=True,
                    font_size=65,  # Slightly larger font for intro/outro
                    animation="fade"
                )
                # Add section indices to intro clips for proper ordering
                for i, clip in enumerate(intro_clips):
                    clip._section_idx = 0  # Intro is always first section
                    clip._debug_info = f"Intro clip {i}"
                    # Verify duration
                    logger.info(f"Created intro clip with duration: {clip.duration:.2f}s (target: {intro_sections[0]['duration']:.2f}s)")
            
            # Generate word-by-word clips for middle sections with correct durations
            middle_clips = []
            if middle_sections:
                logger.info(f"Generating {len(middle_sections)} word-by-word middle clips with accurate durations")
                middle_clips = self.text_helper.generate_word_by_word_clips_parallel(
                    script_sections=middle_sections,
                    font_size=60
                )
                for i, clip in enumerate(middle_clips):
                    section_idx = clip._section_idx if hasattr(clip, '_section_idx') else i + 1
                    logger.info(f"Created middle clip {i} for section {section_idx} with duration: {clip.duration:.2f}s (target: {middle_sections[i]['duration']:.2f}s)")
            
            # Generate standard text clips for outro with correct duration
            outro_clips = []
            if outro_sections:
                logger.info(f"Generating outro text clip with duration: {outro_sections[0]['duration']:.2f}s")
                outro_clips = self.text_helper.generate_text_clips_parallel(
                    script_sections=outro_sections,
                    with_pill=True,
                    font_size=65,
                    animation="fade_out"  # Fade out for outro
                )
                # Add section indices to outro clips
                outro_idx = len(script_sections) - 1 if script_sections else 0
                for i, clip in enumerate(outro_clips):
                    clip._section_idx = outro_idx
                    clip._debug_info = f"Outro clip {i}"
                    # Verify duration
                    logger.info(f"Created outro clip with duration: {clip.duration:.2f}s (target: {outro_sections[0]['duration']:.2f}s)")
            
            # Combine all clips in the correct order
            text_clips = intro_clips + middle_clips + outro_clips
            
            # Double-check we have the right number of clips and with correct indices
            if len(text_clips) != len(script_sections):
                logger.warning(f"Text clip count mismatch: got {len(text_clips)}, expected {len(script_sections)}")
                # Fill in missing clips with None
                if len(text_clips) < len(script_sections):
                    text_clips.extend([None] * (len(script_sections) - len(text_clips)))
                else:
                    text_clips = text_clips[:len(script_sections)]
            
            # Final verification of clip durations against audio durations
            for i, clip in enumerate(text_clips):
                if clip:
                    target_duration = audio_durations.get(i, script_sections[i].get('duration', 5))
                    if abs(clip.duration - target_duration) > 0.5:
                        logger.warning(f"Text clip {i} duration mismatch: {clip.duration:.2f}s vs audio {target_duration:.2f}s - fixing...")
                        text_clips[i] = clip.with_duration(target_duration)
            
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
                # Fallback to video mode if no background clips were generated
                logger.warning("⚠️ FALLBACK: No background clips were generated")
                return self.v_creator.create_youtube_short(
                    title=title,
                    script_sections=script_sections,
                    background_query=background_query,
                    output_filename=output_filename,
                    add_captions=add_captions,
                    style="video",
                    voice_style=voice_style,
                    max_duration=max_duration,
                    background_queries=background_queries,
                    blur_background=blur_background,
                    edge_blur=edge_blur,
                    add_watermark_text=add_watermark_text,
                    existing_audio_data=audio_data
                )
            
            # Also fallback if we have too few background clips compared to script sections
            if len(background_clips) < len(script_sections) * 0.7:
                logger.error(f"Too few background clips generated: {len(background_clips)}/{len(script_sections)}")
                logger.warning(f"⚠️ FALLBACK: Only {len(background_clips)}/{len(script_sections)} background clips were generated")
                return self.v_creator.create_youtube_short(
                    title=title,
                    script_sections=script_sections,
                    background_query=background_query,
                    output_filename=output_filename,
                    add_captions=add_captions,
                    style="video",
                    voice_style=voice_style,
                    max_duration=max_duration,
                    background_queries=background_queries,
                    blur_background=blur_background,
                    edge_blur=edge_blur,
                    add_watermark_text=add_watermark_text,
                    existing_audio_data=audio_data
                )

            if not audio_data:
                logger.error("No audio generated")
                return None

            if not text_clips:
                logger.warning("No text clips generated")

            # Create section clips (background + audio + text)
            section_clips = []
            section_info = {}  # For better logging in parallel renderer

            # Now process each section using audio duration as the source of truth
            for i, (bg_clip, audio, text_clip) in enumerate(zip(background_clips, audio_data, text_clips)):
                # Use the audio duration as the source of truth for section duration
                section_duration = audio_durations.get(i, script_sections[i].get('duration', 5))
                
                # Adjust background to match audio duration
                bg_clip = bg_clip.with_duration(section_duration)
                
                # Add text clip to background if available
                if text_clip:
                    # Make text clip match audio duration
                    text_clip = text_clip.with_duration(section_duration)
                    composite = CompositeVideoClip([bg_clip, text_clip])
                    # Set duration after creation
                    composite = composite.with_duration(section_duration)
                else:
                    composite = bg_clip
                
                # Add audio without duration adjustment
                if audio and 'path' in audio:
                    try:
                        # Load audio without modifying its duration
                        audio_clip = AudioFileClip(audio['path'])
                        # Apply audio as is, retaining its original length
                        composite = composite.with_audio(audio_clip)
                    except Exception as e:
                        logger.error(f"Error adding audio to section {i}: {e}")
                        # Create silent audio as fallback
                        try:
                            silent_audio = AudioFileClip.__new__(AudioFileClip)
                            silent_audio.duration = section_duration
                            composite = composite.with_audio(silent_audio)
                        except:
                            logger.error(f"Could not create silent audio for section {i}")
                else:
                    # No audio provided, create silent audio
                    logger.warning(f"No audio data for section {i}, creating silent audio")
                    try:
                        silent_audio = AudioFileClip.__new__(AudioFileClip)
                        silent_audio.duration = section_duration
                        composite = composite.with_audio(silent_audio)
                    except Exception as e:
                        logger.error(f"Could not create silent audio: {e}")

                # Add debugging info to the clip
                section_text = script_sections[i].get('text', '')[:30] + '...' if len(script_sections[i].get('text', '')) > 30 else script_sections[i].get('text', '')
                composite._debug_info = f"Section {i}: {section_text}"
                composite._section_idx = i

                # Store section information
                section_info[i] = {
                    'section_idx': i,
                    'section_text': section_text,
                    'duration': section_duration
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
                memory_per_worker_gb=1.0,
                options={
                    'clean_temp': True,
                    'section_info': section_info
                }
            )

            # Validate output path exists
            if not output_path or not os.path.exists(output_path):
                logger.error(f"Failed to render final video - output file not found: {output_path}")
                if output_filename and os.path.exists(output_filename):
                    # Use original output filename if render_video fails
                    logger.info(f"Using fallback output file: {output_filename}")
                    output_path = output_filename
                else:
                    logger.error("No valid output file available")
                    return None

            logger.info(f"Successfully rendered video to {output_path}")

            # Add watermark as a post-process if requested
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



