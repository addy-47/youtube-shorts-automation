# for shorts created using gen ai images

import os # for file operations
import time # for timing events and creating filenames like timestamps
import random # for randomizing elements
import textwrap # for wrapping text but is being handled by textclip class in moviepy
import requests # for making HTTP requests
import numpy as np # for numerical operations here used for rounding off
import logging # for logging events
from PIL import Image, ImageFilter, ImageDraw, ImageFont# for image processing
from moviepy.editor import ( # for video editing
    VideoFileClip, VideoClip, TextClip, CompositeVideoClip,ImageClip,
    AudioFileClip, concatenate_videoclips, ColorClip, CompositeAudioClip
)
from moviepy.config import change_settings
change_settings({"IMAGEMAGICK_BINARY": "magick"}) # for windows users
from gtts import gTTS
from dotenv import load_dotenv
import shutil # for file operations like moving and deleting files
import tempfile # for creating temporary files
# Import text clip functions from shorts_maker_V
from shorts_maker_V import YTShortsCreator_V
from datetime import datetime # for more detailed time tracking
import concurrent.futures # for multithreading

# Configure logging for easier debugging
# Do NOT initialize basicConfig here - this will be handled by main.py
logger = logging.getLogger(__name__)

# Timer function for performance monitoring
def measure_time(func):
    """Decorator to measure the execution time of functions"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        start_datetime = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        logger.info(f"STARTING {func.__name__} at {start_datetime}")
        result = func(*args, **kwargs)
        end_time = time.time()
        duration = end_time - start_time
        logger.info(f"COMPLETED {func.__name__} in {duration:.2f} seconds")
        return result
    return wrapper

class YTShortsCreator_I:
    def __init__(self, output_dir="output", fps=30):
        """
        Initialize the YouTube Shorts creator with necessary settings

        Args:
            output_dir (str): Directory to save the output videos
            fps (int): Frames per second for the output video
        """
        # Setup directories
        self.output_dir = output_dir
        self.temp_dir = tempfile.mkdtemp()  # Create temp directory for intermediate files
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(self.temp_dir, exist_ok=True)

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
        self.audio_sync_offset = 0.25  # Delay audio slightly to sync with visuals

        # Font settings
        self.fonts_dir = os.path.join(os.path.dirname(__file__), 'fonts')
        os.makedirs(self.fonts_dir, exist_ok=True)
        self.title_font_path = r"D:\youtube-shorts-automation\fonts\default_font.ttf"
        self.body_font_path = r"D:\youtube-shorts-automation\fonts\default_font.ttf"

        # Create an instance of YTShortsCreator_V to use its text functions
        self.v_creator = YTShortsCreator_V(output_dir=output_dir, fps=fps)

        # Initialize TTS (Text-to-Speech)
        self.azure_tts = None
        self.google_tts = None

        # Initialize Google Cloud TTS
        if os.getenv("USE_GOOGLE_TTS", "true").lower() == "true":
            try:
                from voiceover import GoogleVoiceover
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
                from voiceover import AzureVoiceover
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

        # Load Pexels API ke for background videos
        load_dotenv()
        self.pexels_api_key = os.getenv("PEXELS_API_KEY")  # for fallback images
        self.huggingface_api_key = os.getenv("HUGGINGFACE_API_KEY")
        self.hf_model = os.getenv("HF_MODEL", "stabilityai/stable-diffusion-2-1")
        self.hf_api_url = f"https://api-inference.huggingface.co/models/{self.hf_model}"
        self.hf_headers = {"Authorization": f"Bearer {self.huggingface_api_key}"}

    @measure_time
    def _generate_image_from_prompt(self, prompt, style="photorealistic", file_path=None):
        """
        Generate an image using Hugging Face Diffusion API based on prompt

        Args:
            prompt (str): Image generation prompt
            style (str): Style to apply to the image (e.g., "digital art", "realistic", "photorealistic")
            file_path (str): Path to save the image, if None a path will be generated

        Returns:
            str: Path to the generated image or None if failed
        """
        if not file_path:
            file_path = os.path.join(self.temp_dir, f"gen_img_{int(time.time())}_{random.randint(1000, 9999)}.png")

        # Remove any existing style descriptors from the prompt
        style_keywords = ["digital art", "photorealistic", "oil painting", "realistic", "anime",
                         "concept art", "cinematic", "cartoon", "3d render", "watercolor",
                         "sketch", "illustration", "painting"]

        # First, clean the prompt of any existing style descriptors
        clean_prompt = prompt
        for keyword in style_keywords:
            clean_prompt = clean_prompt.replace(f", {keyword}", "")
            clean_prompt = clean_prompt.replace(f" {keyword}", "")
            clean_prompt = clean_prompt.replace(f"{keyword} ", "")
            clean_prompt = clean_prompt.replace(f"{keyword},", "")

        # Clean up any double commas or spaces that might have been created
        while ",," in clean_prompt:
            clean_prompt = clean_prompt.replace(",,", ",")
        while "  " in clean_prompt:
            clean_prompt = clean_prompt.replace("  ", " ")
        clean_prompt = clean_prompt.strip(" ,")

        # Now add the desired style and quality enhancements
        enhanced_prompt = f"{clean_prompt}, {style}, highly detailed, crisp focus, 4K, high resolution"

        logger.info(f"Original prompt: {prompt[:50]}...")
        logger.info(f"Using style: {style}")
        logger.info(f"Enhanced prompt: {enhanced_prompt[:50]}...")

        retry_count = 0
        max_retries = 3
        success = False
        initial_wait_time = 20  # Starting wait time in seconds

        # Check if Hugging Face API key is available
        if not self.huggingface_api_key:
            logger.error("No Hugging Face API key provided. Will fall back to shorts_maker_V.")
            return None

        while not success and retry_count < max_retries:
            try:
                # Make request to Hugging Face API
                response = requests.post(
                    self.hf_api_url,
                    headers=self.hf_headers,
                    json={"inputs": enhanced_prompt},
                    timeout=30  # Add timeout to prevent hanging indefinitely
                )

                if response.status_code == 200:
                    # Save the image
                    with open(file_path, "wb") as f:
                        f.write(response.content)
                    logger.info(f"Image saved to {file_path}")
                    success = True
                else:
                    # If model is loading, wait and retry
                    try:
                        if "application/json" in response.headers.get("Content-Type", ""):
                            response_json = response.json()
                            if response.status_code == 503 and "estimated_time" in response_json:
                                wait_time = response_json.get("estimated_time", initial_wait_time)
                                logger.info(f"Model is loading. Waiting {wait_time} seconds...")
                                time.sleep(wait_time)
                            else:
                                # Other error
                                logger.error(f"Error generating image: {response.status_code} - {response.text}")
                                time.sleep(initial_wait_time)  # Wait before retrying
                        else:
                            # Non-JSON response (HTML error page)
                            logger.error(f"Non-JSON error response: {response.status_code}")
                            # For 503 errors, wait longer before retry
                            if response.status_code == 503:
                                wait_time = initial_wait_time * (retry_count + 1)  # Gradually increase wait time
                                logger.info(f"Service unavailable (503). Waiting {wait_time} seconds before retry...")
                                time.sleep(wait_time)
                            else:
                                time.sleep(initial_wait_time)  # Wait before retrying
                    except ValueError:
                        # Non-JSON response
                        logger.error(f"Could not parse response: {response.status_code}")
                        time.sleep(initial_wait_time)  # Wait before retrying

                    # Check if we should fall back before trying more retries
                    if response.status_code == 503 and retry_count >= 1:
                        logger.warning("Multiple 503 errors from Hugging Face API. Falling back to shorts_maker_V.")
                        return None

                    retry_count += 1
            except requests.exceptions.RequestException as e:
                logger.error(f"Network error during image generation: {e}")
                retry_count += 1
                time.sleep(initial_wait_time)
            except Exception as e:
                logger.error(f"Unexpected exception during image generation: {e}")
                retry_count += 1
                time.sleep(initial_wait_time)

        # If all retries failed, return None to signal fallback to shorts_maker_V
        if not success:
            logger.error("Failed to generate image with Hugging Face API after multiple attempts")
            return None

        return file_path

    @measure_time
    def _fetch_stock_image(self, query):
        """
        This method is intentionally disabled. Fallback now uses shorts_maker_V instead.
        """
        logger.warning("Stock image fetch called but is disabled. Will fall back to shorts_maker_V.")
        return None

    @measure_time
    def _create_text_based_image(self, text, file_path):
        """
        This method is intentionally disabled. Fallback now uses shorts_maker_V instead.
        """
        logger.warning("Text-based image creation called but is disabled. Will fall back to shorts_maker_V.")
        return None

    @measure_time
    def _create_still_image_clip(self, image_path, duration, text=None, text_position=('center','center'),
                               font_size=60, with_zoom=True, zoom_factor=0.05):
        """
        Create a still image clip with optional text and zoom effect

        Args:
            image_path (str): Path to the image
            duration (float): Duration of the clip in seconds
            text (str): Optional text overlay
            text_position (str): Position of text ('top', 'center', ('center','center'))
            font_size (int): Font size for text
            with_zoom (bool): Whether to add a subtle zoom effect
            zoom_factor (float): Rate of zoom (higher = faster zoom)

        Returns:
            VideoClip: MoviePy clip containing the image and effects
        """
        # Load image
        image = ImageClip(image_path)

        # Resize to fill screen while maintaining aspect ratio
        img_ratio = image.size[0] / image.size[1]
        target_ratio = self.resolution[0] / self.resolution[1]

        if img_ratio > target_ratio:  # Image is wider
            new_height = self.resolution[1]
            new_width = int(new_height * img_ratio)
        else:  # Image is taller
            new_width = self.resolution[0]
            new_height = int(new_width / img_ratio)

        image = image.resize(newsize=(new_width, new_height))

        # Center crop if needed
        if new_width > self.resolution[0] or new_height > self.resolution[1]:
            x_center = new_width // 2
            y_center = new_height // 2
            x1 = max(0, x_center - self.resolution[0] // 2)
            y1 = max(0, y_center - self.resolution[1] // 2)
            image = image.crop(x1=x1, y1=y1, width=self.resolution[0], height=self.resolution[1])

        # Add zoom effect if requested
        if with_zoom:
            def zoom(t):
                # Start at 1.0 zoom and gradually increase
                zoom_level = 1 + (t / duration) * zoom_factor
                return zoom_level

            # Replace lambda with named function
            def zoom_func(t):
                return zoom(t)

            image = image.resize(zoom_func)

        # Set the duration
        image = image.set_duration(duration)

        # Add text if provided
        if text:
            try:
                # Try using the text clip function from YTShortsCreator_V
                txt_clip = self.v_creator._create_text_clip(
                    text,
                    duration=duration,
                    font_size=font_size,
                    position=text_position,
                    with_pill=True
                )
                # Combine image and text
                return CompositeVideoClip([image, txt_clip], size=self.resolution)
            except Exception as e:
                logger.error(f"Error creating text clip using V creator: {e}")
                # Fallback to a simple text implementation if the V creator fails
                try:
                    # Use the simpler built-in MoviePy TextClip without fancy effects
                    simple_txt_clip = TextClip(
                        txt=text,
                        fontsize=font_size,
                        color='white',
                        align='center',
                        method='caption',
                        size=(int(self.resolution[0] * 0.9), None)
                    ).set_position(('center', int(self.resolution[1] * 0.85))).set_duration(duration)

                    # Create a semi-transparent background for better readability
                    txt_w, txt_h = simple_txt_clip.size
                    bg_width = txt_w + 40
                    bg_height = txt_h + 40
                    bg_clip = ColorClip(size=(bg_width, bg_height), color=(0, 0, 0, 128))
                    bg_clip = bg_clip.set_position(('center', int(self.resolution[1] * 0.85) - 20)).set_duration(duration).set_opacity(0.7)

                    # Combine all elements
                    return CompositeVideoClip([image, bg_clip, simple_txt_clip], size=self.resolution)
                except Exception as e2:
                    logger.error(f"Fallback text clip also failed: {e2}")
                    # If all text methods fail, just return the image without text
                    logger.warning("Returning image without text overlay due to text rendering failures")
                    return image
        return image

    @measure_time
    def _create_text_clip(self, text, duration=5, font_size=60, font_path=None, color='white',
                          position='center', animation="fade", animation_duration=1.0, shadow=True,
                          outline=True, with_pill=False, pill_color=(0, 0, 0, 160), pill_radius=30):
        """
        Create a text clip with various effects and animations.
        Using YTShortsCreator_V's implementation for better visibility.
        """
        return self.v_creator._create_text_clip(
            text=text,
            duration=duration,
            font_size=font_size,
            font_path=font_path,
            color=color,
            position=position,
            animation=animation,
            animation_duration=animation_duration,
            shadow=shadow,
            outline=outline,
            with_pill=with_pill,
            pill_color=pill_color,
            pill_radius=pill_radius
        )

    @measure_time
    def _create_word_by_word_clip(self, text, duration, font_size=60, font_path=None,
                             text_color=(255, 255, 255, 255),
                             pill_color=(0, 0, 0, 160),
                             position=('center', 'center')):
        """
        Create a clip where words appear one by one with timing.
        Using YTShortsCreator_V's implementation for better visibility.
        """
        return self.v_creator._create_word_by_word_clip(
            text=text,
            duration=duration,
            font_size=font_size,
            font_path=font_path,
            text_color=text_color,
            pill_color=pill_color,
            position=position
        )

    def _create_pill_image(self, size, color=(0, 0, 0, 160), radius=30):
        """
        Create a pill-shaped background image with rounded corners.
        Using YTShortsCreator_V's implementation.
        """
        return self.v_creator._create_pill_image(size, color, radius)

    @measure_time
    def _create_tts_audio(self, text, filename=None, voice_style="none"):
        """
        Create TTS audio file with robust error handling

        Args:
            text (str): Text to convert to speech
            filename (str): Output filename
            voice_style (str): Style of voice ('excited', 'calm', etc.)

        Returns:
            str: Path to the audio file or None if all methods fail
        """
        if not filename:
            filename = os.path.join(self.temp_dir, f"tts_{int(time.time())}.mp3")

        # Make sure text is not empty and has minimum length
        if not text or len(text.strip()) == 0:
            text = "No text provided"
        elif len(text.strip()) < 5:
            # For very short texts like "Check it out!", expand it slightly to ensure TTS works well
            text = text.strip() + "."  # Add period if missing

        # Try to use Google Cloud TTS if available
        if self.google_tts:
            try:
                voice = os.getenv("GOOGLE_VOICE", "en-US-Neural2-D")
                # Map voice styles for Google Cloud TTS
                google_styles = {
                    "excited": "excited",
                    "calm": "calm",
                    "serious": "serious",
                    "sad": "sad",
                    "none": None
                }
                style = google_styles.get(voice_style, None)

                return self.google_tts.generate_speech(text, output_filename=filename, voice_style=style)
            except Exception as e:
                logger.error(f"Google Cloud TTS failed: {e}, falling back to Azure TTS or gTTS")

        # Try to use Azure TTS if available
        if self.azure_tts:
            try:
                voice = os.getenv("AZURE_VOICE", "en-US-JennyNeural")
                # Map voice styles for Azure
                azure_styles = {
                    "excited": "cheerful",
                    "calm": "gentle",
                    "serious": "serious",
                    "sad": "sad",
                    "none": None
                }
                style = azure_styles.get(voice_style, None)

                return self.azure_tts.generate_speech(text, output_filename=filename)
            except Exception as e:
                logger.error(f"Azure TTS failed: {e}, falling back to gTTS")

        # Fall back to gTTS with multiple retries
        retry_count = 0
        max_retries = 3

        while retry_count < max_retries:
            try:
                tts = gTTS(text=text, lang='en', slow=False)
                tts.save(filename)
                logger.info(f"Successfully created TTS audio: {filename}")
                return filename
            except requests.exceptions.RequestException as e:
                logger.error(f"Network error in gTTS (attempt {retry_count+1}/{max_retries}): {e}")
                time.sleep(2)
                retry_count += 1
            except Exception as e:
                logger.error(f"gTTS error (attempt {retry_count+1}/{max_retries}): {e}")
                time.sleep(2)
                retry_count += 1

        # If all TTS methods fail, create a silent audio clip as a last resort
        try:
            logger.warning("All TTS methods failed. Creating silent audio clip.")
            # Calculate duration based on text length (approx. speaking time)
            words = text.split()
            # Average speaking rate is about 150 words per minute or 2.5 words per second
            duration = max(3, len(words) / 2.5)  # Minimum 3 seconds

            # Create a silent audio clip
            from moviepy.audio.AudioClip import AudioClip
            import numpy as np

            def make_frame(t):
                return np.zeros(2)  # Stereo silence

            silent_clip = AudioClip(make_frame=make_frame, duration=duration)
            silent_clip.write_audiofile(filename, fps=44100, nbytes=2, codec='libmp3lame')

            logger.info(f"Created silent audio clip as fallback: {filename}")
            return filename
        except Exception as e:
            logger.error(f"Failed to create even silent audio: {e}")
            return None

    @measure_time
    def add_watermark(self, clip, watermark_text="Lazycreator", position=("right", "top"), opacity=0.7, font_size=30):
        """
        Add a watermark to a video clip

        Args:
            clip (VideoClip): Video clip to add watermark to
            watermark_text (str): Text to display as watermark
            position (tuple): Position of watermark ('left'/'right', 'top'/'bottom')
            opacity (float): Opacity of watermark (0-1)
            font_size (int): Font size for watermark

        Returns:
            VideoClip: Clip with watermark added
        """
        # Create text clip for watermark
        watermark = TextClip(
            txt=watermark_text,
            fontsize=font_size,
            color='white',
            align='center'
        ).set_duration(clip.duration).set_opacity(opacity)

        # Calculate position
        if position[0] == "right":
            x_pos = clip.w - watermark.w - 20
        else:
            x_pos = 20

        if position[1] == "bottom":
            y_pos = clip.h - watermark.h - 20
        else:
            y_pos = 20

        watermark = watermark.set_position((x_pos, y_pos))

        # Add watermark to video
        return CompositeVideoClip([clip, watermark], size=self.resolution)

    @measure_time
    def create_youtube_short(self, title, script_sections, background_query="abstract background",
                        output_filename=None, add_captions=False, style="ghibli art", voice_style=None, max_duration=25,
                        background_queries=None, blur_background=False, edge_blur=False, add_watermark_text=None):
        """
        Create a YouTube Short using AI-generated images for each script section
        Falls back to shorts_maker_V (video-based) if image generation fails

        Args:
            title (str): Title of the short
            script_sections (list): List of dictionaries with text and duration for each section
            background_query (str): Fallback query for image generation
            output_filename (str): Output file path
            add_captions (bool): Whether to add captions to the video
            style (str): Style of images to generate (e.g., "digital art", "cinematic", "ghibli art")
            voice_style (str): Style of TTS voice
            max_duration (int): Maximum duration in seconds
            background_queries (list): List of queries for each section's background
            blur_background (bool): Whether to apply blur effect to backgrounds
            edge_blur (bool): Whether to apply edge blur to backgrounds
            add_watermark_text (str): Text to use as watermark (None for no watermark)

        Returns:
            str: Path to the created video
        """
        try:
            # Helper variable to track if we should fall back to video mode
            should_fallback_to_video = False

            if not output_filename:
                timestamp = int(time.time())
                output_filename = os.path.join(self.output_dir, f"youtube_short_{timestamp}.mp4")

            # Start timing the overall process
            overall_start_time = time.time()
            logger.info(f"Creating YouTube Short: {title}")

            # Ensure the output directory exists
            os.makedirs(os.path.dirname(output_filename), exist_ok=True)

            # Try to generate AI images for sections
            hugging_face_failed = False

            # Test first image generation to see if Hugging Face API works
            test_query = background_queries[0] if background_queries and len(background_queries) > 0 else background_query
            logger.info(f"Testing Hugging Face API with query: {test_query}")
            test_image = self._generate_image_from_prompt(test_query, style=style)

            if test_image is None:
                # Hugging Face API failed, fallback to shorts_maker_V
                logger.warning("⚠️ FALLBACK: Hugging Face API failed for image generation")
                logger.warning("⚠️ SWITCHING to shorts_maker_V to create video with stock videos instead of AI images")
                hugging_face_failed = True
                should_fallback_to_video = True

            # If initial test failed, use video mode
            if should_fallback_to_video:
                # Create video using shorts_maker_V
                logger.info("Creating video using shorts_maker_V with the same script sections")
                return self.v_creator.create_youtube_short(
                    title=title,
                    script_sections=script_sections,
                    background_query=background_query,
                    output_filename=output_filename,
                    add_captions=add_captions,
                    style="video",  # Force video style since we're using shorts_maker_V
                    voice_style=voice_style,
                    max_duration=max_duration,
                    background_queries=background_queries,
                    blur_background=blur_background,
                    edge_blur=edge_blur,
                    add_watermark_text=add_watermark_text
                )

            # If we get here, Hugging Face is working, proceed with image-based short
            # Identify intro and outro sections
            intro_section = script_sections[0] if script_sections else None
            outro_section = script_sections[-1] if len(script_sections) > 1 else None

            # Middle sections (excluding intro and outro)
            middle_sections = script_sections[1:-1] if len(script_sections) > 2 else []

            # Generate audio clips with TTS for each section
            tts_start_time = time.time()
            logger.info(f"Starting TTS audio generation")

            audio_clips = []
            section_durations = []  # Store actual durations after TTS generation

            # Use multithreading for audio generation to improve performance
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(8, len(script_sections))) as executor:
                # Create a list to hold future objects
                future_to_section = {}

                # Submit TTS generation jobs to the executor
                for i, section in enumerate(script_sections):
                    section_text = section["text"]
                    section_voice_style = section.get("voice_style", voice_style)
                    future = executor.submit(
                        self._create_tts_audio,
                        section_text,
                        None,
                        section_voice_style
                    )
                    future_to_section[future] = (i, section)

                # Process results as they complete
                for future in concurrent.futures.as_completed(future_to_section):
                    i, section = future_to_section[future]
                    min_section_duration = section.get("duration", 5)

                    try:
                        audio_path = future.result()

                        # Process the completed audio file
                        if audio_path and os.path.exists(audio_path):
                            try:
                                # Get actual audio duration
                                audio_clip = AudioFileClip(audio_path)
                                actual_duration = audio_clip.duration

                                # Check if audio has valid duration
                                if actual_duration <= 0:
                                    logger.warning(f"Audio file for section {i} has zero duration, creating fallback silent audio")
                                    # Create silent audio as fallback
                                    from moviepy.audio.AudioClip import AudioClip
                                    import numpy as np

                                    def make_frame(t):
                                        return np.zeros(2)  # Stereo silence

                                    # Use minimum section duration for silent audio
                                    audio_clip = AudioClip(make_frame=make_frame, duration=min_section_duration)
                                    audio_clip = audio_clip.set_fps(44100)
                                    actual_duration = min_section_duration

                                # Ensure minimum duration
                                actual_duration = max(actual_duration, min_section_duration)

                                # Store the final duration and clip
                                section_durations.append((i, actual_duration))
                                audio_clips.append((i, audio_clip, actual_duration))
                            except Exception as e:
                                logger.error(f"Error processing audio for section {i}: {e}")
                                section_durations.append((i, min_section_duration))
                        else:
                            # If no audio was created, use minimum duration
                            section_durations.append((i, min_section_duration))
                    except Exception as e:
                        logger.error(f"Error getting TTS result for section {i}: {e}")
                        section_durations.append((i, min_section_duration))

            # Sort durations by section index
            section_durations.sort(key=lambda x: x[0])

            # Update script sections with actual durations
            for i, duration in section_durations:
                if i < len(script_sections):
                    script_sections[i]['duration'] = duration

            # Recalculate total duration based on actual audio lengths
            total_duration = sum(duration for _, duration in section_durations)

            logger.info(f"Completed TTS audio generation in {time.time() - tts_start_time:.2f} seconds")
            logger.info(f"Updated total duration: {total_duration:.1f}s")

            # Process each section
            section_clips = []

            # Process intro section
            if intro_section and not should_fallback_to_video:
                intro_text = intro_section['text']
                intro_duration = intro_section['duration']

                # Generate image for intro
                image_start_time = time.time()
                logger.info(f"Generating image for intro section")

                if background_queries and len(background_queries) > 0:
                    intro_image_query = background_queries[0]
                else:
                    intro_image_query = background_query

                intro_image_path = self._generate_image_from_prompt(intro_image_query, style=style)
                logger.info(f"Completed image generation for intro in {time.time() - image_start_time:.2f} seconds")

                if not intro_image_path:
                    # If intro image generation failed, fallback to video
                    logger.warning("⚠️ FALLBACK: Image generation failed for intro")
                    should_fallback_to_video = True

                if not should_fallback_to_video:
                    # Create base image clip
                    intro_base_clip = self._create_still_image_clip(
                        intro_image_path,
                        duration=intro_duration,
                        with_zoom=True
                    )

                    # Create components to overlay on the base clip
                    components = [intro_base_clip]

                    # Create title text if provided (only in intro)
                    if title:
                        title_clip = self.v_creator._create_text_clip(
                            title,
                            duration=intro_duration,
                            font_size=70,
                            position=("center", 150),
                            animation="fade",
                            animation_duration=0.8,
                            with_pill=True,
                            pill_color=(0, 0, 0, 180),
                            pill_radius=30
                        )
                        components.append(title_clip)

                    # Create intro text separately from title
                    intro_text_clip = self.v_creator._create_text_clip(
                        intro_text,
                        duration=intro_duration,
                        font_size=60,
                        position=('center', 'center'),
                        animation="fade",
                        animation_duration=0.8,
                        with_pill=True,
                        pill_color=(0, 0, 0, 160),
                        pill_radius=30
                    )
                    components.append(intro_text_clip)

                    # Combine all components
                    intro_clip = CompositeVideoClip(components, size=self.resolution)

                    # Add audio if available
                    for idx, audio_clip, duration in audio_clips:
                        if idx == 0:  # First section is intro
                            # Verify audio clip is valid before using it
                            try:
                                if audio_clip.duration <= 0:
                                    logger.warning(f"Intro audio clip has invalid duration: {audio_clip.duration}s. Creating silent audio.")
                                    # Create silent audio as fallback
                                    from moviepy.audio.AudioClip import AudioClip
                                    import numpy as np

                                    def make_frame(t):
                                        return np.zeros(2)  # Stereo silence

                                    # Create silent audio matching the intro duration
                                    silent_audio = AudioClip(make_frame=make_frame, duration=intro_duration)
                                    silent_audio = silent_audio.set_fps(44100)
                                    intro_clip = intro_clip.set_audio(silent_audio)
                                else:
                                    intro_clip = intro_clip.set_audio(audio_clip)
                            except Exception as e:
                                logger.error(f"Error setting audio for intro: {e}")
                            break

                    section_clips.append(intro_clip)

            # If we need to fall back to video mode, do it now
            if should_fallback_to_video:
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
                    add_watermark_text=add_watermark_text
                )

            # Process middle sections
            if middle_sections and not should_fallback_to_video:
                # Use parallel image generation to improve performance
                future_to_section = {}
                middle_section_images = [None] * len(middle_sections)

                with concurrent.futures.ThreadPoolExecutor(max_workers=min(4, len(middle_sections))) as executor:
                    # Submit image generation jobs
                    for i, section in enumerate(middle_sections):
                        section_idx = i + 1  # Middle sections start at index 1

                        # Get the image prompt for this section
                        if background_queries and section_idx < len(background_queries):
                            image_query = background_queries[section_idx]
                        else:
                            image_query = background_query

                        future = executor.submit(
                            self._generate_image_from_prompt,
                            image_query,
                            style
                        )
                        future_to_section[future] = (i, section, section_idx)

                    # Process results as they complete
                    for future in concurrent.futures.as_completed(future_to_section):
                        i, section, section_idx = future_to_section[future]
                        try:
                            image_path = future.result()
                            if image_path:
                                middle_section_images[i] = image_path
                            else:
                                logger.warning(f"⚠️ FALLBACK: Image generation failed for section {section_idx}")
                                should_fallback_to_video = True
                                break
                        except Exception as e:
                            logger.error(f"Error generating image for section {section_idx}: {e}")
                            should_fallback_to_video = True
                            break

                # If any image generation failed, fall back to video mode
                if should_fallback_to_video:
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
                        add_watermark_text=add_watermark_text
                    )

                # Process each middle section
                middle_clips = []
                for i, section in enumerate(middle_sections):
                    section_text = section['text']
                    section_duration = section['duration']
                    section_idx = i + 1  # Middle sections start at index 1
                    image_path = middle_section_images[i]

                    try:
                        # Create base image clip
                        base_clip = self._create_still_image_clip(
                            image_path,
                            duration=section_duration,
                            with_zoom=True
                        )

                        # Create word-by-word text animation
                        word_clip = self.v_creator._create_word_by_word_clip(
                            text=section_text,
                            duration=section_duration,
                            font_size=60,
                            position=('center', 'center'),
                            text_color=(255, 255, 255, 255),
                            pill_color=(0, 0, 0, 160)
                        )

                        # Combine image with text
                        section_clip = CompositeVideoClip([base_clip, word_clip], size=self.resolution)

                        # Add audio if available
                        for idx, audio_clip, duration in audio_clips:
                            if idx == section_idx:
                                # Verify audio clip is valid before using it
                                try:
                                    if audio_clip.duration <= 0:
                                        logger.warning(f"Section {idx} audio clip has invalid duration: {audio_clip.duration}s. Creating silent audio.")
                                        # Create silent audio as fallback
                                        from moviepy.audio.AudioClip import AudioClip
                                        import numpy as np

                                        def make_frame(t):
                                            return np.zeros(2)  # Stereo silence

                                        # Create silent audio matching the section duration
                                        silent_audio = AudioClip(make_frame=make_frame, duration=section_duration)
                                        silent_audio = silent_audio.set_fps(44100)
                                        section_clip = section_clip.set_audio(silent_audio)
                                    else:
                                        section_clip = section_clip.set_audio(audio_clip)
                                except Exception as e:
                                    logger.error(f"Error setting audio for section {idx}: {e}")
                                break

                        middle_clips.append(section_clip)

                    except Exception as e:
                        logger.error(f"Error processing middle section {section_idx}: {e}")
                        # Fallback to video for all sections if any middle section fails
                        logger.warning(f"⚠️ FALLBACK: Error in section {section_idx}, switching to video")
                        should_fallback_to_video = True
                        break

                # If we need to fall back to video mode, do it now
                if should_fallback_to_video:
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
                            add_watermark_text=add_watermark_text
                        )

                section_clips.extend(middle_clips)

            # Process outro section
            if outro_section and outro_section != intro_section and not should_fallback_to_video:
                outro_text = outro_section['text']
                outro_duration = outro_section['duration']
                outro_idx = len(script_sections) - 1

                # Generate image for outro
                image_start_time = time.time()
                logger.info(f"Generating image for outro section")

                if background_queries and outro_idx < len(background_queries):
                    outro_image_query = background_queries[outro_idx]
                else:
                    outro_image_query = background_query

                outro_image_path = self._generate_image_from_prompt(outro_image_query, style=style)
                logger.info(f"Completed image generation for outro in {time.time() - image_start_time:.2f} seconds")

                if not outro_image_path:
                    # If outro image generation failed, fallback to video
                    logger.warning("⚠️ FALLBACK: Image generation failed for outro")
                    should_fallback_to_video = True

                # If we need to fall back to video mode, do it now
                if should_fallback_to_video:
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
                            add_watermark_text=add_watermark_text
                        )

                # Create base image clip
                outro_base_clip = self._create_still_image_clip(
                    outro_image_path,
                    duration=outro_duration,
                                with_zoom=True
                            )

                # Create outro text
                outro_text_clip = self.v_creator._create_text_clip(
                    outro_text,
                    duration=outro_duration,
                                    font_size=60,
                                    position=('center', 'center'),
                                    animation="fade",
                                    animation_duration=0.8,
                                    with_pill=True,
                                    pill_color=(0, 0, 0, 160),
                                    pill_radius=30
                                )

                # Combine image with text
                outro_clip = CompositeVideoClip([outro_base_clip, outro_text_clip], size=self.resolution)

                # Add audio if available
                for idx, audio_clip, duration in audio_clips:
                    if idx == len(script_sections) - 1:  # Last section is outro
                        # Verify audio clip is valid before using it
                        try:
                            if audio_clip.duration <= 0:
                                logger.warning(f"Outro audio clip has invalid duration: {audio_clip.duration}s. Creating silent audio.")
                                # Create silent audio as fallback
                                from moviepy.audio.AudioClip import AudioClip
                                import numpy as np

                                def make_frame(t):
                                    return np.zeros(2)  # Stereo silence

                                # Create silent audio matching the outro duration
                                silent_audio = AudioClip(make_frame=make_frame, duration=outro_duration)
                                silent_audio = silent_audio.set_fps(44100)
                                outro_clip = outro_clip.set_audio(silent_audio)
                            else:
                                outro_clip = outro_clip.set_audio(audio_clip)
                        except Exception as e:
                            logger.error(f"Error setting audio for outro: {e}")
                        break

                section_clips.append(outro_clip)

            # Add captions at the bottom if requested
            if add_captions and not should_fallback_to_video:
                for i, clip in enumerate(section_clips):
                    if i < len(script_sections):
                        section_text = script_sections[i]['text']
                        caption = self.v_creator._create_text_clip(
                            section_text, duration=clip.duration, font_size=40,
                            font_path=self.body_font_path, position=('center', self.resolution[1] - 200),
                            animation="fade", animation_duration=0.5
                        )
                        section_clips[i] = CompositeVideoClip([clip, caption], size=self.resolution)

            # Check if we have any clips
            if not section_clips:
                if not should_fallback_to_video:
                    logger.warning("No clips were created, falling back to video mode")
                    should_fallback_to_video = True

                if should_fallback_to_video:
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
                        add_watermark_text=add_watermark_text
                    )

            logger.info(f"Successfully processed {len(section_clips)}/{len(script_sections)} sections")

            # Use parallel rendering if available
            try:
                # Check for dill library - needed for optimal parallel rendering
                try:
                    import dill
                    logger.info(f"Found dill {dill.__version__} for improved serialization")
                except ImportError:
                    logger.warning("Dill library not found - parallel rendering may be less efficient")
                    logger.warning("Consider installing dill with: pip install dill")

                from parallel_renderer import render_clips_in_parallel
                logger.info("Using parallel renderer for improved performance")

                # Create temp directory for parallel rendering
                parallel_temp_dir = os.path.join(self.temp_dir, "parallel_render")
                os.makedirs(parallel_temp_dir, exist_ok=True)

                # Concatenate all clips
                final_clip = concatenate_videoclips(section_clips)

                # Ensure we don't exceed maximum duration
                if final_clip.duration > max_duration:
                    logger.warning(f"Video exceeds maximum duration ({final_clip.duration}s > {max_duration}s), trimming")
                    final_clip = final_clip.subclip(0, max_duration)

                # Add watermark if requested
                if add_watermark_text:
                    final_clip = self.add_watermark(final_clip, watermark_text=add_watermark_text)

                # Render clips in parallel
                render_start_time = time.time()
                logger.info(f"Starting parallel video rendering")

                output_filename = render_clips_in_parallel(
                    [final_clip],
                    output_filename,
                    temp_dir=parallel_temp_dir,
                    fps=self.fps,
                    preset="veryfast"
                )

                logger.info(f"Completed video rendering in {time.time() - render_start_time:.2f} seconds")
            except Exception as e:
                logger.warning(f"Parallel renderer failed: {e}. Using standard rendering.")

                # Concatenate all clips
                concat_start_time = time.time()
                logger.info(f"Starting standard video rendering")

                final_clip = concatenate_videoclips(section_clips)

                # Ensure we don't exceed maximum duration
                if final_clip.duration > max_duration:
                    logger.warning(f"Video exceeds maximum duration ({final_clip.duration}s > {max_duration}s), trimming")
                    final_clip = final_clip.subclip(0, max_duration)

                # Add watermark if requested
                if add_watermark_text:
                    final_clip = self.add_watermark(final_clip, watermark_text=add_watermark_text)

                # Write the final video with improved settings
                logger.info(f"Writing video to {output_filename} (duration: {final_clip.duration:.2f}s)")

                final_clip.write_videofile(
                    output_filename,
                    fps=self.fps,
                    codec="libx264",
                    audio_codec="aac",
                    threads=4,
                    preset="veryfast",
                    ffmpeg_params=[
                        "-bufsize", "24M",      # Larger buffer
                        "-maxrate", "8M",       # Higher max rate
                        "-b:a", "192k",         # Higher audio bitrate
                        "-ar", "48000",         # Audio sample rate
                        "-pix_fmt", "yuv420p"   # Compatible pixel format for all players
                    ]
                )

                logger.info(f"Completed video rendering in {time.time() - concat_start_time:.2f} seconds")

            # Print summary of creation process
            overall_duration = time.time() - overall_start_time
            logger.info(f"YouTube short creation completed in {overall_duration:.2f} seconds")
            logger.info(f"Video saved to: {output_filename}")

            # Clean up temporary files
            self._cleanup()

            return output_filename

        except Exception as e:
            logger.error(f"Error creating YouTube Short: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise

    @measure_time
    def _cleanup(self):
        """Clean up temporary files"""
        try:
            for filename in os.listdir(self.temp_dir):
                file_path = os.path.join(self.temp_dir, filename)
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
        except Exception as e:
            logger.error(f"Error cleaning up temporary files: {e}")



