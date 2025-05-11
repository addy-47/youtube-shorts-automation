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
from helper.fetch import _fetch_videos
from helper.blur import custom_blur, custom_edge_blur
from helper.text import TextHelper
from helper.process import _process_background_clip

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
            output_dir (str): Directory to save the output videos
            fps (int): Frames per second for the output video
        """
        # Setup directories
        self.temp_dir = os.path.join(TEMP_DIR, f"shorts_v_{int(time.time())}")
        os.makedirs(self.temp_dir, exist_ok=True)

        # Initialize TextHelper
        self.text_helper = TextHelper()

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

    @measure_time
    def create_youtube_short(self,title,script_sections, background_query="abstract background",
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
                output_filename = os.path.join(self.output_dir, f"short_{date_str}.mp4")

            # Get total duration from script sections
            total_raw_duration = sum(section.get('duration', 5) for section in script_sections)
            duration_scaling_factor = min(1.0, max_duration / total_raw_duration) if total_raw_duration > max_duration else 1.0

            # Scale durations if needed to fit max time
            if duration_scaling_factor < 1.0:
                logger.info(f"Scaling durations by factor {duration_scaling_factor:.2f} to fit max duration of {max_duration}s")
                for section in script_sections:
                    section['duration'] = section['duration'] * duration_scaling_factor

            total_duration = sum(section.get('duration', 5) for section in script_sections)
            logger.info(f"Total video duration: {total_duration}s")

            # Calculate number of background segments needed (usually one per middle section, excluding first and last)
            middle_section_count = max(1, len(script_sections) - 2)
            logger.info(f"Creating video with {middle_section_count} background segments for middle sections")

            # Background videos - section specific if provided
            if background_queries and len(background_queries) == len(script_sections):
                # Use section-specific queries
                section_backgrounds = []

                # Start fetching all backgrounds
                logger.info("Starting background video fetch")
                start_time = time.time()

                # Process middle sections (excluding first and last sections)
                for i, query in enumerate(background_queries[1:-1], 1):
                    # Get multiple videos for variety
                    section_videos = _fetch_videos(query, count=1, min_duration=int(script_sections[i]['duration']) + 2)
                    if section_videos:
                        section_backgrounds.append(section_videos[0])
                    else:
                        # If no video found for section, try the generic background query
                        fallback_videos = _fetch_videos(background_query, count=1, min_duration=int(script_sections[i]['duration']) + 2)
                        if fallback_videos:
                            section_backgrounds.append(fallback_videos[0])
                        else:
                            # Create a black background if no video can be found
                            logger.warning(f"Could not find any videos for section {i}. Creating black background.")
                            section_duration = script_sections[i].get('duration', 5)
                            # We'll add a placeholder path here and create the actual black ColorClip when processing backgrounds
                            section_backgrounds.append("BLACK_BACKGROUND_PLACEHOLDER")

                # Add backgrounds for first and last sections
                first_section_videos = _fetch_videos(background_queries[0], count=1, min_duration=int(script_sections[0]['duration']) + 2)
                if first_section_videos:
                    section_backgrounds.insert(0, first_section_videos[0])
                else:
                    # If no specific video found, add a generic one
                    first_generic = _fetch_videos(background_query, count=1, min_duration=int(script_sections[0]['duration']) + 2)
                    if first_generic:
                        section_backgrounds.insert(0, first_generic[0])
                    else:
                        # Create a black background if no video can be found
                        logger.warning("Could not find any videos for intro section. Creating black background.")
                        section_backgrounds.insert(0, "BLACK_BACKGROUND_PLACEHOLDER")

                last_section_videos = _fetch_videos(background_queries[-1], count=1, min_duration=int(script_sections[-1]['duration']) + 2)
                if last_section_videos:
                    section_backgrounds.append(last_section_videos[0])
                else:
                    # If no specific video found, add a generic one
                    last_generic = _fetch_videos(background_query, count=1, min_duration=int(script_sections[-1]['duration']) + 2)
                    if last_generic:
                        section_backgrounds.append(last_generic[0])
                    else:
                        # Create a black background if no video can be found
                        logger.warning("Could not find any videos for outro section. Creating black background.")
                        section_backgrounds.append("BLACK_BACKGROUND_PLACEHOLDER")

                # Ensure we have enough backgrounds
                while len(section_backgrounds) < len(script_sections):
                    # Add generic backgrounds if needed
                    generic_videos = _fetch_videos(background_query, count=1, min_duration=5)
                    if generic_videos:
                        section_backgrounds.append(generic_videos[0])

                end_time = time.time()
                logger.info(f"Completed background video fetch in {end_time - start_time:.2f} seconds")
                background_videos = section_backgrounds
            else:
                # Use a single query for all backgrounds
                logger.info("Starting background video fetch")
                start_time = time.time()
                background_videos = _fetch_videos(background_query, count=len(script_sections), min_duration=5)
                end_time = time.time()
                logger.info(f"Completed background video fetch in {end_time - start_time:.2f} seconds")

            # Make sure we have enough background videos
            if len(background_videos) < len(script_sections):
                # Fetch more background videos if needed
                logger.info(f"Fetching {len(script_sections) - len(background_videos)} more background videos")
                more_videos = _fetch_videos(
                    background_query,
                    count=len(script_sections) - len(background_videos),
                    min_duration=5
                )

                # If we couldn't fetch any more videos, add placeholders for black backgrounds
                if not more_videos:
                    logger.warning("Could not fetch additional background videos. Using black backgrounds.")
                    for _ in range(len(script_sections) - len(background_videos)):
                        background_videos.append("BLACK_BACKGROUND_PLACEHOLDER")
                else:
                    background_videos.extend(more_videos)

            # Generate TTS for each section
            logger.info("Starting TTS audio generation")
            start_time = time.time()

            audio_files = []
            audio_durations = []

            for i, section in enumerate(script_sections):
                text = section['text']
                section_voice_style = section.get('voice_style', voice_style)

                # Generate temporary filename
                section_audio_file = os.path.join(self.temp_dir, f"section_{i}.mp3")

                # Azure TTS or gTTS
                if self.google_tts:
                    try:
                        audio_path = self.google_tts.generate_speech(
                            text,
                            output_filename=section_audio_file,
                            voice_style=section_voice_style
                        )
                    except Exception as e:
                        logger.error(f"Google Cloud TTS error: {e}. Falling back to gTTS.")
                        tts = gTTS(text=text, lang='en', slow=False)
                        tts.save(section_audio_file)
                        audio_path = section_audio_file
                elif self.azure_tts:
                    try:
                        audio_path = self.azure_tts.generate_speech(
                            text,
                            output_filename=section_audio_file,
                            voice_style=section_voice_style
                        )
                    except Exception as e:
                        logger.error(f"Azure TTS error: {e}. Falling back to gTTS.")
                        tts = gTTS(text=text, lang='en', slow=False)
                        tts.save(section_audio_file)
                        audio_path = section_audio_file
                else:
                    tts = gTTS(text=text, lang='en', slow=False)
                    tts.save(section_audio_file)
                    audio_path = section_audio_file

                audio_files.append(audio_path)

                # Check actual audio duration and update section duration if needed
                try:
                    temp_audio = AudioFileClip(audio_path)
                    actual_duration = temp_audio.duration
                    section['actual_audio_duration'] = actual_duration  # Store actual duration for safety checks
                    audio_durations.append(actual_duration)
                    temp_audio.close()
                except Exception as e:
                    logger.error(f"Error checking audio duration for section {i}: {e}")
                    section['actual_audio_duration'] = section['duration']  # Fallback to planned duration
                    audio_durations.append(section['duration'])

            end_time = time.time()
            logger.info(f"Completed TTS audio generation in {end_time - start_time:.2f} seconds")

            # Update total duration based on actual audio durations
            total_actual_duration = sum(audio_durations)
            logger.info(f"Updated total duration: {total_actual_duration}s")

            # Process background videos
            logger.info("Starting background processing")
            start_time = time.time()

            background_clips = []
            for i, (video_path, section) in enumerate(zip(background_videos, script_sections)):
                try:
                    # Get the actual audio duration instead of planned duration
                    section_duration = section.get('actual_audio_duration', section.get('duration', 5))

                    # Handle the placeholder value we added for cases when no video could be found
                    if video_path == "BLACK_BACKGROUND_PLACEHOLDER":
                        logger.info(f"Creating black background for section {i} as requested")
                        black_bg = ColorClip(size=self.resolution, color=(0, 0, 0), duration=section_duration)
                        background_clips.append(black_bg)
                        continue

                    if os.path.exists(video_path):
                        video_clip = VideoFileClip(video_path)

                        # Apply processing to fit duration and style
                        processed_clip = _process_background_clip(
                            video_clip,
                            section_duration,
                            blur_background=blur_background,
                            edge_blur=edge_blur
                        )

                        # Store processed clip
                        background_clips.append(processed_clip)
                    else:
                        logger.warning(f"Background video {i} not found: {video_path}")
                        # Create a black background as fallback
                        black_bg = ColorClip(size=self.resolution, color=(0, 0, 0), duration=section_duration)
                        background_clips.append(black_bg)
                except Exception as e:
                    logger.error(f"Error processing background clip {i}: {e}")
                    # Create a black background as fallback for this section
                    section_duration = section.get('actual_audio_duration', section.get('duration', 5))
                    black_bg = ColorClip(size=self.resolution, color=(0, 0, 0), duration=section_duration)
                    background_clips.append(black_bg)

            end_time = time.time()
            logger.info(f"Completed background processing in {end_time - start_time:.2f} seconds")

            # Create audio and video for each section
            section_clips = []

            for i, (section, audio_path, bg_clip) in enumerate(zip(script_sections, audio_files, background_clips)):
                try:
                    # Load audio with extra safety check
                    audio_clip = AudioFileClip(audio_path)
                    section_duration = audio_clip.duration  # Use actual audio duration

                    # Ensure background clip is long enough
                    if bg_clip.duration < section_duration:
                        logger.warning(f"Section duration ({section_duration:.2f}s) exceeds available background ({bg_clip.duration:.2f}s), looping")
                        # Instead of using vfx.loop which causes serialization issues, manually create a looped clip
                        loops_needed = int(np.ceil(section_duration / bg_clip.duration))
                        looped_clips = []

                        for _ in range(loops_needed):
                            looped_clips.append(bg_clip.copy())

                        # Concatenate the loops
                        bg_clip = concatenate_videoclips(looped_clips)
                        # Trim to exact duration needed
                        bg_clip = bg_clip.subclipped(0, section_duration)

                    # Set audio to background
                    bg_with_audio = bg_clip.with_duration(section_duration).with_audio(audio_clip)

                    # Add text captions if requested
                    if add_captions:
                        # Use different text approaches based on section position
                        if i == 0 or i == len(script_sections) - 1:  # First section (intro) or last section (outro)
                            # Use regular text clip for intro and outro
                            text_clip = self.text_helper._create_text_clip(
                                section['text'],
                                duration=section_duration,
                                animation="fade",
                                with_pill=True,
                                font_size=70,  # Slightly larger font for intro/outro
                                position=('center', 'center')
                            )
                        else:  # Middle sections
                            # Use word-by-word animation for middle sections
                            text_clip = self.text_helper._create_word_by_word_clip(
                                text=section['text'],
                                duration=section_duration,
                                font_size=60,
                                position=('center', 'center'),
                                text_color=(255, 255, 255, 255),
                                pill_color=(0, 0, 0, 160)
                            )

                        # Composite the text over the background
                        section_clip = CompositeVideoClip([bg_with_audio, text_clip])
                    else:
                        # Always add text overlay regardless of add_captions setting
                        # But still respect the intro/middle/outro distinction
                        if i == 0 or i == len(script_sections) - 1:  # First section (intro) or last section (outro)
                            # Use regular text clip for intro and outro
                            text_clip = self.text_helper._create_text_clip(
                                section['text'],
                                duration=section_duration,
                                animation="fade",
                                with_pill=True,
                                font_size=70,  # Larger font size for better visibility
                                position=('center', 'center')
                            )
                        else:  # Middle sections
                            # Use word-by-word animation for middle sections
                            text_clip = self.text_helper._create_word_by_word_clip(
                                text=section['text'],
                                duration=section_duration,
                                font_size=60,
                                position=('center', 'center'),
                                text_color=(255, 255, 255, 255),
                                pill_color=(0, 0, 0, 160)
                            )

                        # Composite the text over the background
                        section_clip = CompositeVideoClip([bg_with_audio, text_clip])

                    section_clips.append(section_clip)

                except Exception as e:
                    logger.error(f"Error creating section {i}: {e}")
                    # Create a black clip with text as fallback
                    fallback_duration = section.get('actual_audio_duration', section.get('duration', 5))
                    black_bg = ColorClip(size=self.resolution, color=(0, 0, 0), duration=fallback_duration)

                    try:
                        # Try to add audio if possible
                        audio_clip = AudioFileClip(audio_path)
                        black_bg = black_bg.with_audio(audio_clip)
                    except Exception as audio_err:
                        logger.error(f"Error adding audio to fallback clip: {audio_err}")

                    # Add text to explain the error
                    error_text = TextClip(
                        "Error loading section",
                        color='white',
                        size=self.resolution,
                        font_size=60,
                        font=self.text_helper.body_font_path,
                        method='caption'
                    ).with_duration(fallback_duration)

                    section_clip = CompositeVideoClip([black_bg, error_text])
                    section_clips.append(section_clip)

            # Process and validate section clips
            validated_section_clips = []
            for i, clip in enumerate(section_clips):
                try:
                    # Ensure audio duration is valid for this section
                    if clip.audio is not None:
                        # Get actual duration of the audio and clip
                        audio_duration = clip.audio.duration
                        clip_duration = clip.duration

                        # If audio is too short, loop or extend it
                        if audio_duration < clip_duration:
                            logger.warning(f"Audio for section {i} is shorter than clip ({audio_duration}s vs {clip_duration}s), extending")
                            # Create a new audio that exactly matches the clip duration
                            from moviepy.audio.AudioClip import CompositeAudioClip, AudioClip
                            extended_audio = clip.audio.with_duration(clip_duration)
                            clip = clip.with_audio(extended_audio)

                        # If audio is longer, trim it
                        elif audio_duration > clip_duration:
                            logger.warning(f"Audio for section {i} is longer than clip ({audio_duration}s vs {clip_duration}s), trimming")
                            trimmed_audio = clip.audio.subclipped(0, clip_duration)
                            clip = clip.with_audio(trimmed_audio)

                    # Add the section index as a custom attribute for tracking
                    clip._section_idx = i
                    clip._section_text = script_sections[i]['text'][:30] + "..." if len(script_sections[i]['text']) > 30 else script_sections[i]['text']
                    logger.info(f"Adding validated clip {i}: '{clip._section_text}'")

                    validated_section_clips.append(clip)
                except Exception as e:
                    logger.error(f"Error validating section clip {i}: {e}")
                    # Use clip as-is if validation fails
                    clip._section_idx = i
                    clip._section_text = f"Section {i} (validation failed)"
                    logger.info(f"Adding fallback clip {i}")
                    validated_section_clips.append(clip)

            # Log the validated clip order before rendering
            logger.info("=== CLIP ORDER BEFORE RENDERING ===")
            for i, clip in enumerate(validated_section_clips):
                section_idx = getattr(clip, '_section_idx', 'Unknown')
                section_text = getattr(clip, '_section_text', 'Unknown text')
                logger.info(f"Position {i}: Section {section_idx} - '{section_text}'")
            logger.info("=== END CLIP ORDER LOG ===")

            # Use parallel renderer to improve performance
            try:
                from automation.parallel_renderer import render_clips_in_parallel
                logger.info("Using parallel renderer for improved performance")

                # Check for dill for improved serialization
                try:
                    import dill
                    version = dill.__version__
                    if version >= "0.3.9":
                        logger.info(f"Found dill {version} for improved serialization")
                except ImportError:
                    logger.debug("Dill not found, using standard serialization")

                # Make sure clips are in the correct order by sorting them based on their index
                # Sort validated_section_clips by index if they were added out of order
                logger.info("Sorting clips by their section index before rendering")
                section_indices = list(range(len(validated_section_clips)))
                sorted_clips_with_indices = list(zip(section_indices, validated_section_clips))
                sorted_clips = [clip for _, clip in sorted(sorted_clips_with_indices, key=lambda x: x[0])]

                # Log the sorted clip order
                logger.info("=== CLIP ORDER AFTER SORTING ===")
                for i, clip in enumerate(sorted_clips):
                    section_idx = getattr(clip, '_section_idx', 'Unknown')
                    section_text = getattr(clip, '_section_text', 'Unknown text')
                    logger.info(f"Position {i}: Section {section_idx} - '{section_text}'")
                logger.info("=== END SORTED CLIP ORDER LOG ===")

                validated_section_clips = sorted_clips

                # Ensure all clips are properly named with their index before rendering
                for i, clip in enumerate(validated_section_clips):
                    # If clip has a '_idx' attribute, set it to ensure proper ordering
                    if not hasattr(clip, '_idx'):
                        clip._idx = i
                    else:
                        clip._idx = i  # Override any existing index to ensure sequential order

                    # Set a debug attribute with section information to trace through rendering
                    clip._debug_info = f"Section {getattr(clip, '_section_idx', i)}: {getattr(clip, '_section_text', '')}"
                    logger.info(f"Setting clip {i} debug info: {clip._debug_info}")

                # Pass source section info to parallel_renderer for better debugging
                section_info = {}
                for i, clip in enumerate(validated_section_clips):
                    section_info[i] = {
                        'section_idx': getattr(clip, '_section_idx', i),
                        'section_text': getattr(clip, '_section_text', f'Section {i}')
                    }

                # Render all clips in parallel
                output_filename = render_clips_in_parallel(
                    validated_section_clips,
                    output_filename,
                    fps=self.fps,
                    logger=logger,
                    temp_dir=self.temp_dir,
                    section_info=section_info  # Pass section info for better debugging
                )
            except Exception as parallel_error:
                logger.warning(f"Parallel renderer failed: {parallel_error}. Using standard rendering.")

                # Use standard rendering as fallback
                logger.info("Starting standard video rendering")
                try:
                    # Ensure correct order of clips before concatenation
                    section_indices = list(range(len(validated_section_clips)))
                    sorted_clips_with_indices = list(zip(section_indices, validated_section_clips))
                    sorted_clips = [clip for _, clip in sorted(sorted_clips_with_indices, key=lambda x: x[0])]

                    # Concatenate all section clips in correct order
                    final_clip = concatenate_videoclips(sorted_clips)

                    # Add watermark if requested
                    if add_watermark_text:
                        final_clip = self.text_helper.add_watermark(final_clip, watermark_text=add_watermark_text)

                    # Write final video
                    logger.info(f"Rendering final video to {output_filename}")
                    render_start = time.time()

                    final_clip.write_videofile(
                        output_filename,
                        fps=self.fps,
                        codec="libx264",
                        audio_codec="aac",
                        threads=2,
                        preset="veryfast",
                        ffmpeg_params=[
                            "-pix_fmt", "yuv420p",  # For compatibility with all players
                            "-profile:v", "main",   # Better compatibility with mobile devices
                            "-crf", "22",           # Better quality-to-size ratio
                            "-maxrate", "3M",       # Maximum bitrate for streaming
                            "-bufsize", "6M"        # Buffer size for rate control
                        ]
                    )
                finally:
                    # Clean up all clips
                    for clip in validated_section_clips:
                        try:
                            clip.close()
                        except:
                            pass

            return output_filename

        except Exception as e:
            logger.error(f"Error in create_youtube_short: {e}")



