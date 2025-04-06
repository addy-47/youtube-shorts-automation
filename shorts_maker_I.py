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
        if os.getenv("USE_AZURE_TTS", "false").lower() == "true":
            try:
                from voiceover import AzureVoiceover
                self.azure_tts = AzureVoiceover(
                    voice=os.getenv("AZURE_VOICE", "en-US-JennyNeural"),
                    output_dir=self.temp_dir
                )
                logger.info("Azure TTS initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize Azure TTS: {e}. Will use gTTS instead.")

        # Define transition effects
        self.transitions = {
            "fade": lambda clip, duration: clip.fadein(duration).fadeout(duration),
            "slide_left": lambda clip, duration: clip.set_position(lambda t: ((t/duration) * self.resolution[0] - clip.w if t < duration else 0, 'center')),
            "zoom_in": lambda clip, duration: clip.resize(lambda t: max(1, 1 + 0.5 * min(t/duration, 1)))
        }

        # Define video transition effects between background segments
        self.video_transitions = {
            "crossfade": lambda clip1, clip2, duration: concatenate_videoclips([
                clip1.set_end(clip1.duration),
                clip2.set_start(0).crossfadein(duration)
            ], padding=-duration, method="compose"),

            "fade_black": lambda clip1, clip2, duration: concatenate_videoclips([
                clip1.fadeout(duration),
                clip2.fadein(duration)
            ])
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

            image = image.resize(lambda t: zoom(t))

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
    def create_youtube_short(self, title, script_sections, background_query="abstract background",
                        output_filename=None, add_captions=True, style="photorealistic", voice_style=None, max_duration=25,
                        background_queries=None, blur_background=False, edge_blur=False):
        """
        Create a YouTube Short using AI-generated images for each script section
        Falls back to shorts_maker_V (video-based) if image generation fails

        Args:
            title (str): Title of the short
            script_sections (list): List of dictionaries with text and duration for each section
            background_query (str): Fallback query for image generation
            output_filename (str): Output file path
            add_captions (bool): Whether to add captions to the video
            style (str): Style of images to generate (e.g., "digital art", "cinematic", "photorealistic")
            voice_style (str): Style of TTS voice
            max_duration (int): Maximum duration in seconds
            background_queries (list): List of queries for each section's background
            blur_background (bool): Whether to apply blur effect to backgrounds
            edge_blur (bool): Whether to apply edge blur to backgrounds

        Returns:
            str: Path to the created video
        """
        try:
            if not output_filename:
                timestamp = int(time.time())
                output_filename = os.path.join(self.output_dir, f"youtube_short_{timestamp}.mp4")

            logger.info(f"Creating YouTube Short: {title}")

            # Ensure the output directory exists
            os.makedirs(os.path.dirname(output_filename), exist_ok=True)

            # Try to generate AI images for sections
            section_images = []
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
                    edge_blur=edge_blur
                )

            # If we get here, Hugging Face is working, proceed with image-based short
            # Generate images for each section
            section_clips = []
            section_audios = []
            text_clips = []  # List to hold text clips for each section
            total_duration = 0

            # Track how many sections we successfully process
            successful_sections = 0

            # Timing section generation
            section_gen_start_time = time.time()
            logger.info(f"⏱️ STARTING section generation at {datetime.now().strftime('%H:%M:%S.%f')[:-3]}")

            # Add title text at the beginning if provided
            if title:

                # Fixed title clip position
                title_duration = 3.0  # Set a fixed duration for the title
                title_clip = self.v_creator._create_text_clip(
                    text=title,
                    duration=title_duration,
                    font_size=70,
                    position=("center", 150),
                    animation="fade",
                    animation_duration=0.8,
                    with_pill=True,
                    pill_color=(0, 0, 0, 180),
                    pill_radius=30
                )
                # This title will be overlaid on the first section's clip later
            else:
                title_clip = None
                title_duration = 0

            # Process each section
            for i, section in enumerate(script_sections):
                section_start_time = time.time()
                logger.info(f"⏱️ STARTING section {i+1} at {datetime.now().strftime('%H:%M:%S.%f')[:-3]}")

                try:
                    logger.info(f"Processing section {i+1}/{len(script_sections)}")
                    section_text = section["text"]
                    section_duration = section.get("duration", 5)
                    section_voice_style = section.get("voice_style", voice_style)

                    # Get the image prompt for this section
                    if background_queries and i < len(background_queries):
                        image_query = background_queries[i]
                    else:
                        image_query = background_query

                    # Generate image for this section
                    image_start_time = time.time()
                    logger.info(f"⏱️ STARTING image generation for section {i+1} at {datetime.now().strftime('%H:%M:%S.%f')[:-3]}")

                    image_path = None
                    try:
                        image_path = self._generate_image_from_prompt(image_query, style=style)
                    except Exception as e:
                        logger.error(f"Error generating image for section {i}: {e}")
                        # If any image fails, switch to video mode
                        logger.warning("⚠️ FALLBACK: Image generation failed during processing")
                        logger.warning("⚠️ SWITCHING to shorts_maker_V to create video with stock videos instead of AI images")
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
                            edge_blur=edge_blur
                        )

                    image_end_time = time.time()
                    logger.info(f"⏱️ COMPLETED image generation for section {i+1} in {image_end_time - image_start_time:.2f} seconds")

                    # Skip this section if we couldn't get an image
                    if not image_path or not os.path.exists(image_path):
                        logger.warning(f"Image generation failed for section {i}")
                        logger.warning("⚠️ FALLBACK: Switching to shorts_maker_V for video-based short")
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
                            edge_blur=edge_blur
                        )

                    # Create TTS audio for this section
                    audio_start_time = time.time()
                    logger.info(f"⏱️ STARTING audio generation for section {i+1} at {datetime.now().strftime('%H:%M:%S.%f')[:-3]}")

                    audio_path = None
                    try:
                        audio_path = self._create_tts_audio(section_text, voice_style=section_voice_style)
                    except Exception as e:
                        logger.error(f"Error creating TTS for section {i}: {e}")

                    audio_end_time = time.time()
                    logger.info(f"⏱️ COMPLETED audio generation for section {i+1} in {audio_end_time - audio_start_time:.2f} seconds")

                    if audio_path and os.path.exists(audio_path):
                        try:
                            # Get actual audio duration
                            audio_clip = AudioFileClip(audio_path)
                            actual_duration = audio_clip.duration

                            # Ensure minimum duration
                            actual_duration = max(actual_duration, section_duration)

                            # Track total duration
                            total_duration += actual_duration

                            # Create base image clip without text
                            clip_creation_start = time.time()
                            logger.info(f"⏱️ STARTING clip creation for section {i+1} at {datetime.now().strftime('%H:%M:%S.%f')[:-3]}")

                            # Create the base image clip without text
                            section_clip = self._create_still_image_clip(
                                image_path,
                                duration=actual_duration,
                                with_zoom=True
                            )

                            # Create text based on section position
                            text_start_time = time.time()
                            logger.info(f"⏱️ STARTING text creation for section {i+1} at {datetime.now().strftime('%H:%M:%S.%f')[:-3]}")

                            # Determine text type based on position in the script
                            if i == 0 or i == len(script_sections) - 1:
                                # Intro or outro - use text_clip
                                text_clip = self.v_creator._create_text_clip(
                                    text=section_text,
                                    duration=actual_duration,
                                    font_size=60,
                                    position=('center', 'center'),
                                    animation="fade",
                                    animation_duration=0.8,
                                    with_pill=True,
                                    pill_color=(0, 0, 0, 160),
                                    pill_radius=30
                                )
                            else:
                                # Middle sections - use word_by_word_clip
                                text_clip = self.v_creator._create_word_by_word_clip(
                                    text=section_text,
                                    duration=actual_duration,
                                    font_size=60,
                                    position=('center', 'center'),
                                    text_color=(255, 255, 255, 255),
                                    pill_color=(0, 0, 0, 160)
                                )

                            # Add title to the first section if available
                            if i == 0 and title_clip:
                                # Create composite with image, text, and title
                                section_clip = CompositeVideoClip(
                                    [section_clip, text_clip, title_clip],
                                    size=self.resolution
                                )
                            else:
                                # Create composite with just image and text
                                section_clip = CompositeVideoClip(
                                    [section_clip, text_clip],
                                    size=self.resolution
                                )

                            text_end_time = time.time()
                            logger.info(f"⏱️ COMPLETED text creation for section {i+1} in {text_end_time - text_start_time:.2f} seconds")

                            clip_creation_end = time.time()
                            logger.info(f"⏱️ COMPLETED clip creation for section {i+1} in {clip_creation_end - clip_creation_start:.2f} seconds")

                            # Add audio to the clip
                            section_clip = section_clip.set_audio(audio_clip)

                            section_clips.append(section_clip)
                            section_audios.append(audio_clip)
                            successful_sections += 1

                        except Exception as e:
                            logger.error(f"Error creating clip for section {i}: {e}")
                    else:
                        logger.warning(f"No audio for section {i}, using silent clip")
                        try:
                            # Create silent clip with just the image
                            silent_clip_start = time.time()
                            logger.info(f"⏱️ STARTING silent clip creation for section {i+1} at {datetime.now().strftime('%H:%M:%S.%f')[:-3]}")

                            # Create base image clip
                            base_clip = self._create_still_image_clip(
                                image_path,
                                duration=section_duration,
                                with_zoom=True
                            )

                            # Create text based on section position (same logic as above)
                            if i == 0 or i == len(script_sections) - 1:
                                # Intro or outro - use text_clip
                                text_clip = self.v_creator._create_text_clip(
                                    text=section_text,
                                    duration=section_duration,
                                    font_size=60,
                                    position=('center', 'center'),
                                    animation="fade",
                                    animation_duration=0.8,
                                    with_pill=True,
                                    pill_color=(0, 0, 0, 160),
                                    pill_radius=30
                                )
                            else:
                                # Middle sections - use word_by_word_clip
                                text_clip = self.v_creator._create_word_by_word_clip(
                                    text=section_text,
                                    duration=section_duration,
                                    font_size=60,
                                    position=('center', 'center'),
                                    text_color=(255, 255, 255, 255),
                                    pill_color=(0, 0, 0, 160)
                                )

                            # Add title to the first section if available
                            if i == 0 and title_clip:
                                section_clip = CompositeVideoClip(
                                    [base_clip, text_clip, title_clip],
                                    size=self.resolution
                                )
                            else:
                                section_clip = CompositeVideoClip(
                                    [base_clip, text_clip],
                                    size=self.resolution
                                )

                            silent_clip_end = time.time()
                            logger.info(f"⏱️ COMPLETED silent clip creation for section {i+1} in {silent_clip_end - silent_clip_start:.2f} seconds")

                            section_clips.append(section_clip)
                            successful_sections += 1
                        except Exception as e:
                            logger.error(f"Error creating silent clip for section {i}: {e}")
                except Exception as e:
                    logger.error(f"Error processing section {i}: {e}")
                    # Continue to next section instead of failing completely

                section_end_time = time.time()
                logger.info(f"⏱️ COMPLETED section {i+1} in {section_end_time - section_start_time:.2f} seconds")

            section_gen_end_time = time.time()
            logger.info(f"⏱️ COMPLETED section generation in {section_gen_end_time - section_gen_start_time:.2f} seconds")

            # Concatenate all sections
            if not section_clips:
                raise ValueError("No clips were created. Cannot produce video.")

            logger.info(f"Successfully processed {successful_sections}/{len(script_sections)} sections")

            # Create the final composite video
            concat_start_time = time.time()
            logger.info(f"⏱️ STARTING video concatenation at {datetime.now().strftime('%H:%M:%S.%f')[:-3]}")

            final_clip = concatenate_videoclips(section_clips)

            concat_end_time = time.time()
            logger.info(f"⏱️ COMPLETED video concatenation in {concat_end_time - concat_start_time:.2f} seconds")

            # Ensure we don't exceed maximum duration
            if final_clip.duration > max_duration:
                logger.warning(f"Video exceeds maximum duration ({final_clip.duration}s > {max_duration}s), trimming")
                final_clip = final_clip.subclip(0, max_duration)

            # Write the final video
            render_start_time = time.time()
            logger.info(f"⏱️ STARTING video rendering at {datetime.now().strftime('%H:%M:%S.%f')[:-3]}")
            logger.info(f"Writing video to {output_filename} (duration: {final_clip.duration:.2f}s)")

            final_clip.write_videofile(
                output_filename,
                fps=self.fps,
                codec="libx264",
                audio_codec="aac",
                threads=4,
                preset="veryfast"
            )

            render_end_time = time.time()
            logger.info(f"⏱️ COMPLETED video rendering in {render_end_time - render_start_time:.2f} seconds")

            # Clean up temporary files
            cleanup_start_time = time.time()
            logger.info(f"⏱️ STARTING cleanup at {datetime.now().strftime('%H:%M:%S.%f')[:-3]}")

            self._cleanup()

            cleanup_end_time = time.time()
            logger.info(f"⏱️ COMPLETED cleanup in {cleanup_end_time - cleanup_start_time:.2f} seconds")

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



