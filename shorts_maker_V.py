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
    AudioFileClip, concatenate_videoclips, ColorClip, CompositeAudioClip
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

# Configure logging for easier debugging
# Do NOT initialize basicConfig here - this will be handled by main.py
logger = logging.getLogger(__name__)

# Timer function for performance monitoring
def measure_time(func):
    """Decorator to measure the execution time of functions"""
    def wrapper(*args, **kwargs):
        # Only log timing for major functions (create_youtube_short)
        if func.__name__ == "create_youtube_short":
            start_time = time.time()
            logger.info(f"Starting YouTube short creation")
            result = func(*args, **kwargs)
            end_time = time.time()
            duration = end_time - start_time
            logger.info(f"Completed YouTube short creation in {duration:.2f} seconds")
        else:
            # For all other functions, just run without detailed logging
            result = func(*args, **kwargs)
        return result
    return wrapper

class YTShortsCreator_V:
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
        self.audio_sync_offset = 0.0  # Remove audio delay to improve sync

        # Font settings
        self.fonts_dir = os.path.join(os.path.dirname(__file__), 'fonts')
        os.makedirs(self.fonts_dir, exist_ok=True)
        self.title_font_path = r"D:\youtube-shorts-automation\fonts\default_font.ttf"
        self.body_font_path = r"D:\youtube-shorts-automation\fonts\default_font.ttf"

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

        # Load Pexels API key for background videos
        load_dotenv()
        self.pexels_api_key = os.getenv("PEXELS_API_KEY")
        self.pixabay_api_key = os.getenv("PIXABAY_API_KEY")

    @measure_time
    def _fetch_videos(self, query, count=5, min_duration=5):
        """
        Fetch background videos from multiple sources with randomized API selection

        Args:
            query (str): Search term for videos
            count (int): Number of videos to fetch
            min_duration (int): Minimum video duration in seconds

        Returns:
            list: Paths to downloaded video files
        """
        # Determine how many videos to fetch from each source
        videos = []

        # Randomly decide which API to try first
        apis = ["pexels", "pixabay"]
        random.shuffle(apis)  # Shuffle the list to try APIs in random order

        # Try each API in sequence
        for api in apis:
            logger.info(f"Fetching {count} videos using {api} API")

            if api == "pexels":
                try:
                    api_videos = self._fetch_from_pexels(query, count, min_duration)
                    if api_videos:  # If we got videos, add them and stop trying
                        videos.extend(api_videos)
                        break
                except Exception as e:
                    logger.error(f"Error fetching videos from Pexels: {e}")
                    # Continue to next API
            else:  # pixabay
                try:
                    api_videos = self._fetch_from_pixabay(query, count, min_duration)
                    if api_videos:  # If we got videos, add them and stop trying
                        videos.extend(api_videos)
                        break
                except Exception as e:
                    logger.error(f"Error fetching videos from Pixabay: {e}")
                    # Continue to next API

        # If we have fewer videos than requested, but at least one, return what we have
        if not videos:
            logger.warning(f"Could not fetch any videos for query: {query}")

        return videos[:count]

    @measure_time
    def _fetch_from_pixabay(self, query, count, min_duration):
        """
        Fetch background videos from Pixabay API

        Args:
            query (str): Search term for videos
            count (int): Number of videos to fetch
            min_duration (int): Minimum video duration in seconds

        Returns:
            list: Paths to downloaded video files
        """
        try:
            url = f"https://pixabay.com/api/videos/?key={self.pixabay_api_key}&q={query}&min_width=1080&min_height=1920&per_page=20"
            response = requests.get(url) # make a request to the API
            if response.status_code == 200: # if the request is successful
                data = response.json()  # changes the response in json to py dict data
                videos = data.get("hits", []) # get the videos from the data
                video_paths = []
                # Randomly select videos from the top 10
                top_videos = videos[:10]
                # Then randomly select 'count' videos from the top 10
                if len(top_videos) > count:
                    selected_videos = random.sample(top_videos, count)
                else:
                    selected_videos = top_videos

                def download_and_check_video(video):
                    try:
                        video_url = video["videos"]["large"]["url"]
                        video_path = os.path.join(self.temp_dir, f"pixabay_{video['id']}.mp4") # create a path for the video
                        with requests.get(video_url, stream=True) as r:  # get the video from the url
                            r.raise_for_status() # raise an error if the request is not successful
                            with open(video_path, 'wb') as f: # open the video file in write binary mode
                                for chunk in r.iter_content(chunk_size=8192): # iterate over the content of the video
                                    f.write(chunk)
                        clip = VideoFileClip(video_path)
                        if clip.duration >= min_duration:
                            clip.close()
                            return video_path
                        clip.close()
                        # Remove the video if it's too short
                        if os.path.exists(video_path):
                            os.remove(video_path)
                        return None
                    except Exception as e:
                        logger.error(f"Error downloading video from Pixabay: {e}")
                        return None

                # Use ThreadPoolExecutor to download videos in parallel
                with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(selected_videos), 5)) as executor:
                    # Submit all download tasks and collect futures
                    future_to_video = {executor.submit(download_and_check_video, video): video for video in selected_videos}

                    # Process completed futures
                    for future in concurrent.futures.as_completed(future_to_video):
                        video_path = future.result()
                        if video_path:
                            video_paths.append(video_path)

                return video_paths

            # If response wasn't 200, return empty list
            logger.warning(f"Pixabay API returned status code {response.status_code}")
            return []
        except Exception as e:
            logger.error(f"Error fetching videos from Pixabay: {e}")
            return []

    @measure_time
    def _fetch_from_pexels(self, query, count=5, min_duration=15):
        """
        Fetch background videos from Pexels API

        Args:
            query (str): Search term for videos
            count (int): Number of videos to fetch
            min_duration (int): Minimum video duration in seconds

        Returns:
            list: Paths to downloaded video files
        """
        try:
            url = f"https://api.pexels.com/videos/search?query={query}&per_page=20&orientation=portrait"
            headers = {"Authorization": self.pexels_api_key}
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                data = response.json()
                videos = data.get("videos", [])
                video_paths = []
                # Randomly select videos from the top 10
                top_videos = videos[:10]
                # Then randomly select 'count' videos from those top 10
                if len(top_videos) > count:
                    selected_videos = random.sample(top_videos, count)
                else:
                    selected_videos = top_videos

                def download_and_check_video(video):
                    try:
                        video_files = video.get("video_files", []) # get the video files
                        if not video_files:
                            return None

                        video_url = video_files[0].get("link") # get the video link
                        if not video_url:
                            return None

                        video_path = os.path.join(self.temp_dir, f"pexels_{video['id']}.mp4")
                        with requests.get(video_url, stream=True) as r:
                            r.raise_for_status()
                            with open(video_path, 'wb') as f:
                                for chunk in r.iter_content(chunk_size=8192):
                                    f.write(chunk)

                        clip = VideoFileClip(video_path)
                        if clip.duration >= min_duration:
                            clip.close()
                            return video_path
                        clip.close()
                        # Remove the video if it's too short
                        if os.path.exists(video_path):
                            os.remove(video_path)
                        return None
                    except Exception as e:
                        logger.error(f"Error downloading video from Pexels: {e}")
                        return None

                # Use ThreadPoolExecutor to download videos in parallel
                with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(selected_videos), 5)) as executor:
                    # Submit all download tasks and collect futures
                    future_to_video = {executor.submit(download_and_check_video, video): video for video in selected_videos}

                    # Process completed futures
                    for future in concurrent.futures.as_completed(future_to_video):
                        video_path = future.result()
                        if video_path:
                            video_paths.append(video_path)

                return video_paths

            # If response wasn't 200, return empty list
            logger.warning(f"Pexels API returned status code {response.status_code}")
            return []
        except Exception as e:
            logger.error(f"Error fetching videos from Pexels: {e}")
            return []

    @measure_time
    def _create_pill_image(self, size, color=(0, 0, 0, 160), radius=30):
        """
        Create a pill-shaped background image with rounded corners.

        Args:
            size (tuple): Size of the image (width, height)
            color (tuple): Color of the pill background (RGBA)
            radius (int): Radius of the rounded corners

        Returns:
            Image: PIL Image with the pill-shaped background
        """
        width, height = size
        img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        # Draw the rounded rectangle
        draw.rectangle([(radius, 0), (width - radius, height)], fill=color)
        draw.rectangle([(0, radius), (width, height - radius)], fill=color)
        draw.ellipse([(0, 0), (radius * 2, radius * 2)], fill=color)
        draw.ellipse([(width - radius * 2, 0), (width, radius * 2)], fill=color)
        draw.ellipse([(0, height - radius * 2), (radius * 2, height)], fill=color)
        draw.ellipse([(width - radius * 2, height - radius * 2), (width, height)], fill=color)

        return img

    @measure_time
    def _create_text_clip(self, text, duration=5, font_size=60, font_path=None, color='white',
                          position='center', animation="fade", animation_duration=1.0, shadow=True,
                          outline=True, with_pill=False, pill_color=(0, 0, 0, 160), pill_radius=30):
        """
        Create a text clip with various effects and animations.

        Args:
            text (str): Text content
            duration (float): Duration in seconds
            font_size (int): Font size
            font_path (str): Path to font file
            color (str): Text color
            position (str): Position of text (top, center, bottom)
            animation (str): Animation type
            animation_duration (float): Duration of animation effects
            shadow (bool): Whether to add shadow
            outline (bool): Whether to add outline
            with_pill (bool): Whether to add pill background
            pill_color (tuple): RGBA color for pill background
            pill_radius (int): Radius for pill corners

        Returns:
            TextClip: MoviePy text clip with effects
        """
        if not font_path:
            font_path = self.body_font_path

        try:
            txt_clip = TextClip(
                txt=text,
                font=font_path,
                fontsize=font_size,
                color=color,
                method='caption',
                align='center',
                size=(self.resolution[0] - 100, None)
            )
        except Exception as e:
            logger.warning(f"Text rendering error with custom font: {e}. Using default.")
            txt_clip = TextClip(
                txt=text,
                fontsize=font_size,
                color=color,
                method='caption',
                align='center',
                size=(self.resolution[0] - 100, None)
            )

        txt_clip = txt_clip.set_duration(duration)
        clips = []

        # Add pill-shaped background if requested
        if with_pill:
            pill_image = self._create_pill_image(txt_clip.size, color=pill_color, radius=pill_radius)
            pill_clip = ImageClip(np.array(pill_image), duration=duration)
            clips.append(pill_clip)

        # Add shadow effect
        if shadow:
            shadow_clip = TextClip(
                txt=text,
                font=font_path,
                fontsize=font_size,
                color='black',
                method='caption',
                align='center',
                size=(self.resolution[0] - 100, None)
            ).set_position((5, 5), relative=True).set_opacity(0.7).set_duration(duration)
            clips.append(shadow_clip)

        # Add outline effect
        if outline:
            outline_clips = []
            for dx, dy in [(-1,-1), (-1,1), (1,-1), (1,1)]:
                oc = TextClip(
                    txt=text,
                    font=font_path,
                    fontsize=font_size,
                    color='black',
                    method='caption',
                    align='center',
                    size=(self.resolution[0] - 100, None)
                ).set_position((dx, dy), relative=True).set_opacity(0.5).set_duration(duration)
                outline_clips.append(oc)
            clips.extend(outline_clips)

        clips.append(txt_clip)
        text_composite = CompositeVideoClip(clips)

        # Set the position of the entire composite
        text_composite = text_composite.set_position(position)

        # Apply animation
        if animation in self.transitions:
            anim_func = self.transitions[animation]
            text_composite = anim_func(text_composite, animation_duration)

        # Create transparent background for the text
        bg = ColorClip(size=self.resolution, color=(0,0,0,0)).set_duration(duration)
        final_clip = CompositeVideoClip([bg, text_composite], size=self.resolution)

        return final_clip

    @measure_time
    def _create_word_by_word_clip(self, text, duration, font_size=60, font_path=None,
                             text_color=(255, 255, 255, 255),
                             pill_color=(0, 0, 0, 160),  # Semi-transparent black
                             position=('center', 'center')):
        """
        Create a word-by-word animation clip with pill-shaped backgrounds

            text: text to be animated
            duration: duration of the animation
            font_size: size of the font
            font_path: path to the font file
            text_color: color of the text
            pill_color: color of the pill background (with transparency)
            position: position of the text

        Returns:
            VideoClip: Word-by-word animation clip
        """
        if not font_path:
            font_path = self.body_font_path

        # Split text into words and calculate durations
        words = text.split()
        char_counts = [len(word) for word in words]
        total_chars = sum(char_counts)
        transition_duration = 0.02  # Faster transitions for better sync
        total_transition_time = transition_duration * (len(words) - 1)
        speech_duration = duration * 0.98  # Use more of the time for speech
        effective_duration = speech_duration - total_transition_time

        word_durations = []
        min_word_time = 0.2  # Slightly faster minimum word display time
        for word in words:
            char_ratio = len(word) / max(1, total_chars)
            word_time = min_word_time + (effective_duration - min_word_time * len(words)) * char_ratio
            word_durations.append(word_time)

        # Adjust durations to match total duration
        actual_sum = sum(word_durations) + total_transition_time
        if abs(actual_sum - duration) > 0.01:
            adjust_factor = (duration - total_transition_time) / sum(word_durations)
            word_durations = [d * adjust_factor for d in word_durations]

        clips = []
        current_time = 0

        for i, (word, word_duration) in enumerate(zip(words, word_durations)):
            # Create a function to draw the frame with the word on a pill background
            def make_frame_with_pill(word=word, font_size=font_size, font_path=font_path,
                                    text_color=text_color, pill_color=pill_color):
                # Load font
                font = ImageFont.truetype(font_path, font_size)

                # Calculate text size
                dummy_img = Image.new('RGBA', (1, 1))
                dummy_draw = ImageDraw.Draw(dummy_img)
                text_bbox = dummy_draw.textbbox((0, 0), word, font=font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]

                # Get ascent and descent for more precise vertical positioning
                ascent, descent = font.getmetrics()

                # Add padding for the pill
                padding_x = int(font_size * 0.7)  # Horizontal padding
                padding_y = int(font_size * 0.35)  # Vertical padding

                # Create image
                img_width = text_width + padding_x * 2
                img_height = text_height + padding_y * 2

                # Create a transparent image
                img = Image.new('RGBA', (img_width, img_height), (0, 0, 0, 0))
                draw = ImageDraw.Draw(img)

                # Create the pill shape (rounded rectangle)
                radius = img_height // 2

                # Draw the pill
                # Draw the center rectangle
                draw.rectangle([(radius, 0), (img_width - radius, img_height)], fill=pill_color)
                # Draw the left semicircle
                draw.ellipse([(0, 0), (radius * 2, img_height)], fill=pill_color)
                # Draw the right semicircle
                draw.ellipse([(img_width - radius * 2, 0), (img_width, img_height)], fill=pill_color)

                # For horizontal centering:
                text_x = (img_width - text_width) // 2
                # For vertical centering:
                offset_y = (descent - ascent) // 4 # This small adjustment often helps
                text_y = (img_height - text_height) // 2 + offset_y

                draw.text((text_x, text_y), word, font=font, fill=text_color)

                return img

            # Create the frame with the word on a pill
            word_image = make_frame_with_pill()

            # Convert to clip
            word_clip = ImageClip(np.array(word_image), duration=word_duration)

            # Add to clips list
            clips.append(word_clip)

            # Update current time
            current_time += word_duration + transition_duration

        # Concatenate clips
        clips_with_transitions = []
        for i, clip in enumerate(clips):
            if i < len(clips) - 1:  # Not the last clip
                clip = clip.crossfadein(transition_duration)
            clips_with_transitions.append(clip)

        word_sequence = concatenate_videoclips(clips_with_transitions, method="compose")

        # Create a transparent background the size of the entire clip
        bg = ColorClip(size=self.resolution, color=(0,0,0,0)).set_duration(word_sequence.duration)

        # Position the word sequence in the center of the background
        positioned_sequence = word_sequence.set_position(position)

        # Combine the background and positioned sequence
        final_clip = CompositeVideoClip([bg, positioned_sequence], size=self.resolution)

        return final_clip

    @measure_time
    def custom_blur(self, clip, radius=5):
        """
        Apply a Gaussian blur effect to video clips

        Args:
            clip (VideoClip): Video clip to blur
            radius (int): Blur radius

        Returns:
            VideoClip: Blurred video clip
        """
        def blur_frame(get_frame, t):
            frame = get_frame(t)
            img = Image.fromarray(frame)
            blurred = img.filter(ImageFilter.GaussianBlur(radius=radius))
            return np.array(blurred)

        def apply_blur(get_frame, t):
            return blur_frame(get_frame, t)

        return clip.fl(apply_blur)

    @measure_time
    def custom_edge_blur(self, clip, edge_width=50, radius=10):
        """
        Apply blur only to the edges of a video clip

        Args:
            clip (VideoClip): Video clip to blur edges of
            edge_width (int): Width of the edge to blur
            radius (int): Blur radius

        Returns:
            VideoClip: Video clip with blurred edges
        """
        def blur_frame(get_frame, t):
            frame = get_frame(t)
            img = Image.fromarray(frame)
            width, height = img.size

            # Create a mask for the unblurred center
            mask = Image.new('L', (width, height), 0)
            draw = ImageDraw.Draw(mask)
            draw.rectangle(
                [(edge_width, edge_width), (width - edge_width, height - edge_width)],
                fill=255
            )

            # Blur the entire image
            blurred = img.filter(ImageFilter.GaussianBlur(radius=radius))

            # Composite the blurred image with the original using the mask
            composite = Image.composite(img, blurred, mask)

            return np.array(composite)

        def apply_edge_blur(get_frame, t):
            return blur_frame(get_frame, t)

        return clip.fl(apply_edge_blur)

    @measure_time
    def _process_background_clip(self, clip, target_duration, blur_background=False, edge_blur=False):
        """
        Process a background clip to match the required duration

        Args:
            clip (VideoClip): The video clip to process
            target_duration (float): The desired duration in seconds
            blur_background (bool): Whether to apply blur effect
            edge_blur (bool): Whether to apply edge blur effect

        Returns:
            VideoClip: Processed clip with matching duration
        """
        # Handle videos shorter than needed duration with proper looping
        if clip.duration < target_duration:
            # Create enough loops to cover the needed duration
            loops_needed = int(np.ceil(target_duration / clip.duration))
            looped_clips = []

            for loop in range(loops_needed):
                if loop == loops_needed - 1:
                    # For the last segment, only take what we need
                    remaining_needed = target_duration - (loop * clip.duration)
                    if remaining_needed > 0:
                        segment = clip.subclip(0, min(remaining_needed, clip.duration))
                        looped_clips.append(segment)
                else:
                    looped_clips.append(clip.copy())

            clip = concatenate_videoclips(looped_clips)
        else:
            # If longer than needed, take a random segment
            if clip.duration > target_duration + 1:
                max_start = clip.duration - target_duration - 0.5
                start_time = random.uniform(0, max_start)
                clip = clip.subclip(start_time, start_time + target_duration)
            else:
                # Just take from the beginning if not much longer
                clip = clip.subclip(0, target_duration)

        # Resize to match height
        clip = clip.resize(height=self.resolution[1])

       # Apply blur effect only if requested
        if blur_background and not edge_blur:
            clip = self.custom_blur(clip, radius=5)
        elif edge_blur:
            clip = self.custom_edge_blur(clip, edge_width=75, radius=10)

        # Center the video if it's not wide enough
        if clip.w < self.resolution[0]:
            bg = ColorClip(size=self.resolution, color=(0, 0, 0)).set_duration(clip.duration)
            x_pos = (self.resolution[0] - clip.w) // 2
            clip = CompositeVideoClip([bg, clip.set_position((x_pos, 0))], size=self.resolution)

        # Crop width if wider than needed
        elif clip.w > self.resolution[0]:
            x_centering = (clip.w - self.resolution[0]) // 2
            clip = clip.crop(x1=x_centering, x2=x_centering + self.resolution[0])

        # Make sure we have exact duration to prevent timing issues
        clip = clip.set_duration(target_duration)

        return clip

    @measure_time
    def create_youtube_short(self, title, script_sections, background_query="abstract background",
                            output_filename=None, add_captions=False, style="video", voice_style=None, max_duration=25,
                            background_queries=None, blur_background=False, edge_blur=False):
        """
        Create a YouTube Short with the given script sections.

        Args:
            title (str): Title for the video
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
                    section_videos = self._fetch_videos(query, count=1, min_duration=int(script_sections[i]['duration']) + 2)
                    if section_videos:
                        section_backgrounds.append(section_videos[0])
                    else:
                        # If no video found for section, try the generic background query
                        fallback_videos = self._fetch_videos(background_query, count=1, min_duration=int(script_sections[i]['duration']) + 2)
                        if fallback_videos:
                            section_backgrounds.append(fallback_videos[0])

                # Add backgrounds for first and last sections
                first_section_videos = self._fetch_videos(background_queries[0], count=1, min_duration=int(script_sections[0]['duration']) + 2)
                if first_section_videos:
                    section_backgrounds.insert(0, first_section_videos[0])
                else:
                    # If no specific video found, add a generic one
                    first_generic = self._fetch_videos(background_query, count=1, min_duration=int(script_sections[0]['duration']) + 2)
                    if first_generic:
                        section_backgrounds.insert(0, first_generic[0])

                last_section_videos = self._fetch_videos(background_queries[-1], count=1, min_duration=int(script_sections[-1]['duration']) + 2)
                if last_section_videos:
                    section_backgrounds.append(last_section_videos[0])
                else:
                    # If no specific video found, add a generic one
                    last_generic = self._fetch_videos(background_query, count=1, min_duration=int(script_sections[-1]['duration']) + 2)
                    if last_generic:
                        section_backgrounds.append(last_generic[0])

                # Ensure we have enough backgrounds
                while len(section_backgrounds) < len(script_sections):
                    # Add generic backgrounds if needed
                    generic_videos = self._fetch_videos(background_query, count=1, min_duration=5)
                    if generic_videos:
                        section_backgrounds.append(generic_videos[0])

                end_time = time.time()
                logger.info(f"Completed background video fetch in {end_time - start_time:.2f} seconds")
                background_videos = section_backgrounds
            else:
                # Use a single query for all backgrounds
                logger.info("Starting background video fetch")
                start_time = time.time()
                background_videos = self._fetch_videos(background_query, count=len(script_sections), min_duration=5)
                end_time = time.time()
                logger.info(f"Completed background video fetch in {end_time - start_time:.2f} seconds")

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

            # Make sure we have enough background videos
            if len(background_videos) < len(script_sections):
                # Fetch more background videos if needed
                logger.info(f"Fetching {len(script_sections) - len(background_videos)} more background videos")
                more_videos = self._fetch_videos(
                    background_query,
                    count=len(script_sections) - len(background_videos),
                    min_duration=5
                )
                background_videos.extend(more_videos)

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
                        bg_clip = bg_clip.subclip(0, section_duration)

                    # Set audio to background
                    bg_with_audio = bg_clip.set_duration(section_duration).set_audio(audio_clip)

                    # Add text captions if requested
                    if add_captions:
                        # Use different text approaches based on section position
                        if i == 0 or i == len(script_sections) - 1:  # First section (intro) or last section (outro)
                            # Use regular text clip for intro and outro
                            text_clip = self._create_text_clip(
                                section['text'],
                                duration=section_duration,
                                animation="fade",
                                with_pill=True,
                                font_size=70,  # Slightly larger font for intro/outro
                                position=('center', 'center')
                            )
                        else:  # Middle sections
                            # Use word-by-word animation for middle sections
                            text_clip = self._create_word_by_word_clip(
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
                            text_clip = self._create_text_clip(
                                section['text'],
                                duration=section_duration,
                                animation="fade",
                                with_pill=True,
                                font_size=70,  # Larger font size for better visibility
                                position=('center', 'center')
                            )
                        else:  # Middle sections
                            # Use word-by-word animation for middle sections
                            text_clip = self._create_word_by_word_clip(
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

                    validated_section_clips.append(clip)
                except Exception as e:
                    logger.error(f"Error validating section clip {i}: {e}")
                    # Use clip as-is if validation fails
                    validated_section_clips.append(clip)

            # Try parallel rendering first
            try:
                logger.info("Using parallel renderer for improved performance")

                # Check for dill library - needed for optimal parallel rendering
                try:
                    import dill
                    logger.info(f"Found dill {dill.__version__} for improved serialization")
                except ImportError:
                    logger.warning("Dill library not found - parallel rendering may be less efficient")
                    logger.warning("Consider installing dill with: pip install dill")

                from parallel_renderer import render_clips_in_parallel
                output_filename = render_clips_in_parallel(
                    validated_section_clips,
                    output_filename,
                    temp_dir=self.temp_dir,
                    fps=self.fps,
                    preset="veryfast"
                )
            except Exception as parallel_error:
                logger.warning(f"Parallel renderer failed: {parallel_error}. Using standard rendering.")

                # Use standard rendering as fallback
                logger.info("Starting standard video rendering")
                try:
                    # Concatenate all section clips
                    final_clip = concatenate_videoclips(validated_section_clips)

                    # Write final video
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


