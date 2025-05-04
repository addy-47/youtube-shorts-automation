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

        # Define max clip duration (maximum length for a single video clip)
        self.max_clip_duration = 30  # 30 seconds is a reasonable default

        # Font settings
        self.font = "Arial"  # Default font
        self.font_size = 60  # Default font size
        self.font_color = 'white'  # Default font color

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
        
    def add_watermark(self, clip, watermark_text="LazyCreator"):
        """
        Add a watermark to a video clip
        
        Args:
            clip: The video clip to add watermark to
            watermark_text: Text to use as watermark
            
        Returns:
            Clip with watermark added
        """
        if not watermark_text:
            return clip
            
        # Create watermark text
        watermark = TextClip(
            watermark_text,
            fontsize=40,
            color='white',
            font=self.font,
            align='East'
        ).set_opacity(0.5).set_duration(clip.duration)
        
        # Position in bottom right corner with padding
        watermark = watermark.set_position(('right', 'bottom'))
        
        # Composite with original clip
        return CompositeVideoClip([clip, watermark])

    def set_temp_dir(self, temp_dir):
        """
        Set the temporary directory for this creator.
        
        Args:
            temp_dir: Path to temporary directory
        """
        self.temp_dir = temp_dir
        logger.info(f"Set temporary directory to: {temp_dir}")
        
        # Also set tempfile.tempdir for any modules that use it directly
        import tempfile
        tempfile.tempdir = temp_dir

    @measure_time
    def fetch_background_video(self, query):
        """
        Fetch a background video for a section
        
        Args:
            query: Search query for the video
            
        Returns:
            Path to downloaded video file
        """
        logger.info(f"Fetching background video with query: {query}")
        # _fetch_videos doesn't accept output_dir parameter, use default temp_dir
        video_files = _fetch_videos(query, count=1)
        if video_files and len(video_files) > 0:
            return video_files[0]
        else:
            return None
            
    @measure_time
    def generate_section_audio(self, section):
        """
        Generate audio for a section
        
        Args:
            section: Script section with text and voice_style
            
        Returns:
            Path to the generated audio file
        """
        text = section['text']
        voice_style = section.get('voice_style', 'neutral')
        
        # Generate unique filename
        output_filename = os.path.join(self.temp_dir, f"tts_{int(time.time()*1000)}_{hash(text)%10000}.mp3")
        
        # Try Google TTS first
        if self.google_tts:
            try:
                logger.info(f"Generating audio with Google TTS: {text[:30]}...")
                return self.google_tts.generate_speech(text, output_filename=output_filename, voice_style=voice_style)
            except Exception as e:
                logger.error(f"Google TTS failed: {e}")
                
        # Try Azure TTS if Google fails
        if self.azure_tts:
            try:
                logger.info(f"Generating audio with Azure TTS: {text[:30]}...")
                return self.azure_tts.generate_speech(text, output_filename=output_filename)
            except Exception as e:
                logger.error(f"Azure TTS failed: {e}")
                
        # Fall back to gTTS
        try:
            logger.info(f"Generating audio with gTTS: {text[:30]}...")
            tts = gTTS(text=text, lang='en', slow=False)
            tts.save(output_filename)
            return output_filename
        except Exception as e:
            logger.error(f"gTTS failed: {e}")
            
            # Create silent audio as last resort
            try:
                from moviepy.audio.AudioClip import AudioClip
                import numpy as np
                
                # Calculate duration based on text length
                words = text.split()
                duration = max(3, len(words) / 2.5)  # Minimum 3 seconds
                
                def make_frame(t):
                    return np.zeros(2)  # Stereo silence
                    
                silent_clip = AudioClip(make_frame=make_frame, duration=duration)
                silent_clip.write_audiofile(output_filename, fps=44100, nbytes=2, codec='libmp3lame')
                
                return output_filename
            except Exception as silent_err:
                logger.error(f"Failed to create silent audio: {silent_err}")
                return None

    @measure_time
    def generate_single_voiceover(self, text, voice_style='neutral'):
        """
        Generate a single voiceover audio file - for parallel processing support
        
        Args:
            text: Text content to convert to speech
            voice_style: Voice style to use
            
        Returns:
            Path to the generated audio file
        """
        # Create a mock section dict to pass to generate_section_audio
        section = {
            'text': text,
            'voice_style': voice_style
        }
        return self.generate_section_audio(section)
        
    @measure_time
    def create_text_clip(self, text, idx_or_duration=None, font_size=None, font_name=None, font_color=None):
        """
        Create a text clip for parallel processing
        
        Args:
            text: Text content
            idx_or_duration: Either the section index or the duration in seconds
            font_size: Font size (optional)
            font_name: Font name (optional)
            font_color: Font color (optional)
            
        Returns:
            Path to rendered text clip or TextClip object
        """
        # Use provided values or defaults
        font_size = font_size or self.font_size
        font_name = font_name or self.font
        font_color = font_color or self.font_color
        
        # Determine duration from script sections if needed
        # In parallel processing, this will be called with the index
        duration = 5  # Default duration
        if isinstance(idx_or_duration, (int, float)):
            if idx_or_duration > 1000:  # Assume it's a duration if it's large
                duration = idx_or_duration
            else:
                # It's probably an index, so just use default duration
                # The actual duration will be adjusted later
                duration = 5
        
        # Create the text clip
        if len(text) > 100:  # Use word-by-word for longer texts
            return self.text_helper._create_word_by_word_clip(
                text=text,
                duration=duration,
                font_size=font_size,
                position=('center', 'center'),
                text_color=font_color,
                pill_color=(0, 0, 0, 160)
            )
        else:  # Use regular text clip for shorter texts
            return self.text_helper._create_text_clip(
                text,
                duration=duration,
                animation="fade",
                with_pill=True,
                font_size=font_size,
                position=('center', 'center')
            )

    @measure_time
    def create_youtube_short(self, title, script_sections, background_query, output_filename="yt_short.mp4", 
                            style="photorealistic", voice_style="neutral", max_duration=30, 
                            background_queries=None, blur_background=False, edge_blur=False, parallel_results=None,
                            add_captions=False, add_watermark_text=None):
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
            add_captions: Whether to add captions to the video
            add_watermark_text: Text to add as watermark

        Returns:
            Path to the generated video
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

            total_duration = sum(section.get('duration', 5) for section in script_sections)
            logger.info(f"Total video duration: {total_duration}s")

            # Calculate number of background segments needed (usually one per middle section, excluding first and last)
            middle_section_count = max(1, len(script_sections) - 2)
            logger.info(f"Creating video with {middle_section_count} background segments for middle sections")

            # Fetch background videos based on parallel results or fetch new ones
            background_videos = []
            
            # If we have pre-fetched backgrounds, use them
            if parallel_results and 'backgrounds' in parallel_results and len(parallel_results['backgrounds']) > 0:
                logger.info("Using pre-fetched backgrounds from parallel processing")
                background_videos = [parallel_results['backgrounds'].get(i) for i in range(len(script_sections))]
                # Filter out any None values
                background_videos = [bg for bg in background_videos if bg is not None]
            
            # If we don't have enough backgrounds, fetch them
            if not background_videos or len(background_videos) < len(script_sections):
                logger.info("Fetching background videos")
                
                # If we have specific queries for each section, use them
                if background_queries and len(background_queries) >= len(script_sections):
                    # Fetch a video for each section with its specific query
                    for i, section in enumerate(script_sections):
                        section_query = background_queries[i]
                        logger.info(f"Fetching background for section {i} with query: {section_query}")
                        section_video = self.fetch_background_video(section_query)
                        if section_video:
                            background_videos.append(section_video)
                else:
                    # Use the general background query for all sections
                    logger.info(f"Fetching backgrounds with general query: {background_query}")
                    for i in range(len(script_sections)):
                        section_video = self.fetch_background_video(background_query)
                        if section_video:
                            background_videos.append(section_video)
            
            # Ensure we have enough background videos
            if len(background_videos) < len(script_sections):
                logger.warning(f"Not enough background videos found. Using available ones multiple times if needed.")
                # Duplicate existing videos if needed
                while len(background_videos) < len(script_sections):
                    background_videos.append(background_videos[len(background_videos) % len(background_videos)])

            # Add for audio handling with parallel results
            # Replace audio generation with parallel results if available
            if parallel_results and 'audio' in parallel_results:
                logger.info("Using pre-generated audio from parallel processing")
                
                # Check if we have pre-generated audio for all sections
                all_audio_available = True
                for i, section in enumerate(script_sections):
                    if i not in parallel_results['audio']:
                        all_audio_available = False
                        logger.warning(f"Missing audio for section {i}, will need to generate it")
                        
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
                                logger.error(f"Error loading pre-generated audio {i}: {e}")
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
                        processed_clip = self._process_background_clip(
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
            
            # Process audio files to get AudioFileClip objects
            audio_clips = []
            for audio_path in section_audio_clips:
                try:
                    # If it's already an AudioFileClip, use it directly
                    if isinstance(audio_path, AudioFileClip):
                        audio_clips.append(audio_path)
                    # Otherwise, load it from the file path
                    elif audio_path and os.path.exists(audio_path):
                        audio_clip = AudioFileClip(audio_path)
                        audio_clips.append(audio_clip)
                    else:
                        # Create silent audio as fallback
                        from moviepy.audio.AudioClip import AudioClip
                        import numpy as np
                        def make_frame(t):
                            return np.zeros(2)  # Stereo silence
                        silent_clip = AudioClip(make_frame=make_frame, duration=5)
                        audio_clips.append(silent_clip)
                except Exception as e:
                    logger.error(f"Error loading audio: {e}")
                    # Create silent audio as fallback
                    from moviepy.audio.AudioClip import AudioClip
                    import numpy as np
                    def make_frame(t):
                        return np.zeros(2)  # Stereo silence
                    silent_clip = AudioClip(make_frame=make_frame, duration=5)
                    audio_clips.append(silent_clip)

            for i, (section, audio_clip, bg_clip) in enumerate(zip(script_sections, audio_clips, background_clips)):
                try:
                    # Use actual audio duration
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
                        final_clip = self.add_watermark(final_clip, watermark_text=add_watermark_text)

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

    @measure_time
    def load_background_clip(self, background_file, target_duration):
        """
        Load a background video clip and process it for use
        
        Args:
            background_file: Path to the background video file
            target_duration: Target duration for the clip
            
        Returns:
            Processed background clip
        """
        if not os.path.exists(background_file):
            raise FileNotFoundError(f"Background file not found: {background_file}")
            
        # Load the video clip
        video_clip = VideoFileClip(background_file)
        
        # Process the clip
        processed_clip = self._process_background_clip(video_clip, target_duration)
        
        return processed_clip
        
    def _process_background_clip(self, clip, target_duration, blur_background=False, edge_blur=False):
        """
        Process a background clip to match requirements
        
        Args:
            clip: Input video clip
            target_duration: Target duration
            blur_background: Whether to apply blur effect
            edge_blur: Whether to apply edge blur effect
            
        Returns:
            Processed clip
        """
        # Use helper._process_background_clip if available, otherwise process here
        try:
            return _process_background_clip(
                clip,
                target_duration,
                blur_background=blur_background,
                edge_blur=edge_blur
            )
        except Exception as e:
            logger.warning(f"Error using helper._process_background_clip: {e}. Processing locally.")
            
        # Resize to match the resolution if needed
        if clip.size != self.resolution:
            # Calculate resize factor to ensure the clip fills the screen
            width_factor = self.resolution[0] / clip.w
            height_factor = self.resolution[1] / clip.h
            resize_factor = max(width_factor, height_factor)
            
            # Resize the clip
            resized_clip = clip.resize(newsize=(int(clip.w * resize_factor), int(clip.h * resize_factor)))
            
            # Crop to our exact resolution
            x_center = resized_clip.w / 2
            y_center = resized_clip.h / 2
            x1 = max(0, int(x_center - self.resolution[0] / 2))
            y1 = max(0, int(y_center - self.resolution[1] / 2))
            
            cropped_clip = resized_clip.crop(
                x1=x1,
                y1=y1,
                width=self.resolution[0],
                height=self.resolution[1]
            )
            clip = cropped_clip
        
        # Loop the clip if it's shorter than the target duration
        if clip.duration < target_duration:
            from moviepy.video.fx.loop import loop
            clip = loop(clip, duration=target_duration)
        
        # Trim if longer than target duration
        if clip.duration > target_duration:
            clip = clip.subclip(0, target_duration)
        
        # Apply blur if requested
        if blur_background:
            try:
                clip = custom_blur(clip)
            except Exception as e:
                logger.warning(f"Error applying blur effect: {e}")
                
        # Apply edge blur if requested
        if edge_blur:
            try:
                clip = custom_edge_blur(clip)
            except Exception as e:
                logger.warning(f"Error applying edge blur effect: {e}")
                
        return clip

    @measure_time
    def fetch_and_prepare_background(self, query, section, max_clip_duration):
        """
        Fetch and prepare a single background clip for a section
        
        Args:
            query: Search query for the background
            section: Script section
            max_clip_duration: Maximum clip duration
            
        Returns:
            Processed background clip
        """
        # Get target duration
        section_duration = min(section.get('duration', 5), max_clip_duration)
        
        try:
            # Fetch video
            video_path = self.fetch_background_video(query)
            
            if video_path and os.path.exists(video_path):
                # Load and process clip
                video_clip = VideoFileClip(video_path)
                return self._process_background_clip(
                    video_clip,
                    section_duration,
                    blur_background=False,
                    edge_blur=False
                )
            else:
                # Create black background as fallback
                logger.warning(f"Video not found for query '{query}', creating fallback")
                return ColorClip(size=self.resolution, color=(0, 0, 0), duration=section_duration)
        except Exception as e:
            # Create fallback on error
            logger.error(f"Error fetching/processing background for query '{query}': {e}")
            return ColorClip(size=self.resolution, color=(0, 0, 0), duration=section_duration)

    @measure_time
    def fetch_background_clips_for_sections(self, script_sections, general_query, section_queries=None, max_clip_duration=30):
        """
        Fetch and prepare background clips for each script section
        
        Args:
            script_sections: List of script sections
            general_query: General search query to use for all sections
            section_queries: List of specific queries for each section
            max_clip_duration: Maximum clip duration
            
        Returns:
            List of processed background clips
        """
        background_clips = []
        
        # Fetch videos for each section
        for i, section in enumerate(script_sections):
            # Use section-specific query if available
            query = general_query
            if section_queries and i < len(section_queries) and section_queries[i]:
                query = section_queries[i]
                
            logger.info(f"Fetching background for section {i} with query: {query}")
            
            # Get the target duration for this section
            section_duration = min(section.get('duration', 5), max_clip_duration)
            
            # Fetch and prepare clip for this section
            try:
                # Get video path
                video_path = self.fetch_background_video(query)
                
                if video_path and os.path.exists(video_path):
                    # Load and process clip
                    video_clip = VideoFileClip(video_path)
                    processed_clip = self._process_background_clip(
                        video_clip,
                        section_duration,
                        blur_background=False,
                        edge_blur=False
                    )
                    background_clips.append(processed_clip)
                else:
                    # Create fallback if video not found
                    logger.warning(f"Video not found for section {i}, creating fallback")
                    black_bg = ColorClip(size=self.resolution, color=(0, 0, 0), duration=section_duration)
                    background_clips.append(black_bg)
            except Exception as e:
                # Create fallback on error
                logger.error(f"Error processing background for section {i}: {e}")
                black_bg = ColorClip(size=self.resolution, color=(0, 0, 0), duration=section_duration)
                background_clips.append(black_bg)
                
        return background_clips
