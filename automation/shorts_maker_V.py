import os # for file operations
import time # for timing events and creating filenames like timestamps
import random # for randomizing elements
import textwrap # for wrapping text into lines but most cases being handled by textclip class in moviepy
import requests # for making HTTP requests
import numpy as np # for numerical operations here used for rounding off
import logging # for logging events
from PIL import Image, ImageFilter, ImageDraw, ImageFont# for image processing
from moviepy.editor import ( # for video editing
    VideoFileClip, VideoClip, TextClip, CompositeVideoClip,ImageClip,
    AudioFileClip, concatenate_videoclips, ColorClip, CompositeAudioClip, concatenate_audioclips
)
from moviepy.video.fx import all as vfx
from moviepy.config import change_settings
change_settings({"IMAGEMAGICK_BINARY": "magick"}) # for windows users
from gtts import gTTS
from dotenv import load_dotenv
import shutil # for file operations like moving and deleting files
import tempfile # for creating temporary files
from datetime import datetime # for more detailed time tracking
import concurrent.futures
from functools import wraps
from helper.minor_helper import measure_time
from helper.fetch import _fetch_videos
from helper.blur import custom_blur, custom_edge_blur
from helper.text import TextHelper
from helper.process import _process_background_clip
from helper.video_encoder import VideoEncoder
from helper.keyframe_animation import KeyframeTrack, convert_callable_to_keyframes

# Configure logging for easier debugging
# Do NOT initialize basicConfig here - this will be handled by main.py
logger = logging.getLogger(__name__)

load_dotenv()  # Load environment variables from .env file

class YTShortsCreator_V:
    def __init__(self, fps=30):
        """
        Initialize the YouTube Shorts creator with necessary settings

        Args:
            output_dir (str): Directory to save the output videos
            fps (int): Frames per second for the output video
        """
        # Setup directories
        self.temp_dir = tempfile.mkdtemp()  # Create temp directory for intermediate files
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
            return clip.set_position(position_func)

        def zoom_in_transition(clip, duration):
            def size_func(t):
                return max(1, 1 + 0.5 * min(t/duration, 1))
            return clip.resize(size_func)

        # Define video transition effects between background segments
        def crossfade_transition(clip1, clip2, duration):
            return concatenate_videoclips([
                clip1.set_end(clip1.duration),
                clip2.set_start(0).crossfadein(duration)
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
    def create_youtube_short(self, title, script_sections, background_query, output_filename="yt_short.mp4", 
                            style="photorealistic", voice_style="neutral", max_duration=30, 
                            background_queries=None, blur_background=False, edge_blur=False, parallel_results=None):
        """
        Create a YouTube Short with video background.

        Args:
            title: Title of the short
            script_sections: List of dictionaries with text and duration for each section
            background_query: Query for fetching background video
            output_filename: Output file name
            style: Visual style (not used for video-based shorts)
            voice_style: Voice style for TTS
            max_duration: Maximum duration in seconds
            background_queries: List of queries for each section
            blur_background: Whether to apply blur effect to background
            edge_blur: Whether to apply edge blur effect
            parallel_results: Pre-processed results from parallel tasks

        Returns:
            Path to the generated video
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

            # Replace background fetching with parallel results if available
            if parallel_results and 'backgrounds' in parallel_results:
                self.logger.info("Using pre-fetched backgrounds from parallel processing")
                
                # Check if we have pre-fetched backgrounds for all sections
                all_backgrounds_available = True
                for i, section in enumerate(script_sections):
                    if i not in parallel_results['backgrounds']:
                        all_backgrounds_available = False
                        self.logger.warning(f"Missing background for section {i}, will need to fetch it")
                        
                if all_backgrounds_available:
                    # Use pre-fetched backgrounds
                    background_clips = []
                    for i, section in enumerate(script_sections):
                        background_file = parallel_results['backgrounds'].get(i)
                        if background_file:
                            # Load the background file
                            try:
                                target_duration = min(section['duration'], max_clip_duration)
                                background_clip = self.load_background_clip(background_file, target_duration)
                                background_clips.append(background_clip)
                            except Exception as e:
                                self.logger.error(f"Error loading pre-fetched background {i}: {e}")
                                # Fall back to fetching
                                background_query = background_queries[i] if background_queries and i < len(background_queries) else background_query
                                background_clips.append(self.fetch_and_prepare_background(background_query, section, max_clip_duration))
                        else:
                            # Fall back to fetching
                            background_query = background_queries[i] if background_queries and i < len(background_queries) else background_query
                            background_clips.append(self.fetch_and_prepare_background(background_query, section, max_clip_duration))
                else:
                    # Fall back to original fetching
                    background_clips = self.fetch_background_clips_for_sections(script_sections, background_query, background_queries, max_clip_duration)
            else:
                # No parallel results, use original fetching
                background_clips = self.fetch_background_clips_for_sections(script_sections, background_query, background_queries, max_clip_duration)

            # Add for audio handling with parallel results
            # Replace audio generation with parallel results if available
            if parallel_results and 'audio' in parallel_results:
                self.logger.info("Using pre-generated audio from parallel processing")
                
                # Check if we have pre-generated audio for all sections
                all_audio_available = True
                for i, section in enumerate(script_sections):
                    if i not in parallel_results['audio']:
                        all_audio_available = False
                        self.logger.warning(f"Missing audio for section {i}, will need to generate it")
                        
                if all_audio_available:
                    # Use pre-generated audio
                    section_audio_clips = []
                    for i, section in enumerate(script_sections):
                        audio_file = parallel_results['audio'].get(i)
                        if audio_file:
                            # Load the audio file
                            try:
                                audio_clip = AudioFileClip(audio_file)
                                section_audio_clips.append(audio_clip)
                            except Exception as e:
                                self.logger.error(f"Error loading pre-generated audio {i}: {e}")
                                # Fall back to generation
                                section_audio_clips.append(self.generate_section_audio(section))
                        else:
                            # Fall back to generation
                            section_audio_clips.append(self.generate_section_audio(section))
                else:
                    # Fall back to original generation
                    section_audio_clips = [self.generate_section_audio(section) for section in script_sections]
            else:
                # No parallel results, use original generation
                section_audio_clips = [self.generate_section_audio(section) for section in script_sections]

            # Process background videos
            logger.info("Starting background processing")
            start_time = time.time()

            background_clips = []
            for i, (video_path, section) in enumerate(zip(background_videos, script_sections)):
                try:
                    # Get the actual audio duration instead of planned duration
                    section_duration = section.get('actual_audio_duration', section.get('duration', 5))

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

            for i, (section, audio_path, bg_clip) in enumerate(zip(script_sections, section_audio_clips, background_clips)):
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
                        bg_clip = bg_clip.subclip(0, section_duration)

                    # Set audio to background
                    bg_with_audio = bg_clip.set_duration(section_duration).set_audio(audio_clip)

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
                        black_bg = black_bg.set_audio(audio_clip)
                    except Exception as audio_err:
                        logger.error(f"Error adding audio to fallback clip: {audio_err}")

                    # Add text to explain the error
                    error_text = TextClip(
                        "Error loading section",
                        color='white',
                        size=self.resolution,
                        fontsize=60,
                        method='caption',
                        align='center'
                    ).set_duration(fallback_duration)

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
                            extended_audio = clip.audio.set_duration(clip_duration)
                            clip = clip.set_audio(extended_audio)

                        # If audio is longer, trim it
                        elif audio_duration > clip_duration:
                            logger.warning(f"Audio for section {i} is longer than clip ({audio_duration}s vs {clip_duration}s), trimming")
                            trimmed_audio = clip.audio.subclip(0, clip_duration)
                            clip = clip.set_audio(trimmed_audio)

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
                    # Use optimized encoder for final output
                    VideoEncoder.write_clip(
                        final_clip, 
                        output_filename, 
                        fps=30, 
                        is_final=True, 
                        show_progress=True
                    )
                finally:
                    # Clean up all clips
                    for clip in validated_section_clips:
                        try:
                            clip.close()
                        except:
                            pass

            # Final cleanup
            self._cleanup()

            return output_filename

        except Exception as e:
            logger.error(f"Error in create_youtube_short: {e}")
            # If there's a temp directory, clean it up
            if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
                try:
                    shutil.rmtree(self.temp_dir)
                except Exception as cleanup_error:
                    logger.error(f"Error cleaning up temp directory: {cleanup_error}")
            raise

    def _cleanup(self):
        """Clean up temporary files"""
        try:
            shutil.rmtree(self.temp_dir)
            logger.info("Temporary files cleaned up successfully.")
        except Exception as e:
            logger.warning(f"Error cleaning up temporary files: {str(e)}")

    def set_temp_dir(self, temp_dir):
        """
        Set the temporary directory for this creator.
        
        Args:
            temp_dir: Path to temporary directory
        """
        self.temp_dir = temp_dir
        self.logger.info(f"Set temporary directory to: {temp_dir}")
        
        # Also set tempfile.tempdir for any modules that use it directly
        import tempfile
        tempfile.tempdir = temp_dir


