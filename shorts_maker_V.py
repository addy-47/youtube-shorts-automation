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
from moviepy.config import change_settings
change_settings({"IMAGEMAGICK_BINARY": "magick"}) # for windows users
from gtts import gTTS
from dotenv import load_dotenv
import shutil # for file operations like moving and deleting files
import tempfile # for creating temporary files
from datetime import datetime # for more detailed time tracking

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

        # Video settings
        self.resolution = (1080, 1920)  # Portrait mode for shorts (width, height)
        self.fps = fps
        self.audio_sync_offset = 0.25  # Delay audio slightly to sync with visuals

        # Font settings
        self.fonts_dir = os.path.join(os.path.dirname(__file__), 'fonts')
        os.makedirs(self.fonts_dir, exist_ok=True)
        self.title_font_path = r"D:\youtube-shorts-automation\fonts\default_font.ttf"
        self.body_font_path = r"D:\youtube-shorts-automation\fonts\default_font.ttf"

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
        api = random.choice(apis)
        logger.info(f"Fetching {count} videos using {api} API")

        if api == "pexels":
            try:
                api_videos = self._fetch_from_pexels(query, count, min_duration)
            except Exception as e:
                logger.error(f"Error fetching videos from Pexels: {e}")
                return []
        else:  # pixabay
            try:
                api_videos = self._fetch_from_pixabay(query, count, min_duration)
            except Exception as e:
                logger.error(f"Error fetching videos from Pixabay: {e}")
                return []

        videos.extend(api_videos)

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

                for video in selected_videos:
                    video_url = video["videos"]["large"]["url"]
                    video_path = os.path.join(self.temp_dir, f"pixabay_{video['id']}.mp4") # create a path for the video
                    with requests.get(video_url, stream=True) as r:  # get the video from the url
                        r.raise_for_status() # raise an error if the request is not successful
                        with open(video_path, 'wb') as f: # open the video file in write binary mode
                            for chunk in r.iter_content(chunk_size=8192): # iterate over the content of the video
                                f.write(chunk)
                    clip = VideoFileClip(video_path)
                    if clip.duration >= min_duration:
                        video_paths.append(video_path)
                    clip.close()
                return video_paths

            return self._fetch_from_pexels(query, count, min_duration)
        except Exception as e:
            logger.error(f"Error fetching videos from Pixabay: {e}")
            return self._fetch_from_pexels(query, count, min_duration)

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
            response = requests.get(url)
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

                for video in selected_videos:
                    video_files = video.get("video_files", []) # get the video files
                    if video_files:
                        video_url = video_files[0].get("link") # get the video link
                    video_path = os.path.join(self.temp_dir, f"pexels_{video['id']}.mp4")
                    with requests.get(video_url, stream=True) as r:
                        r.raise_for_status()
                        with open(video_path, 'wb') as f:
                            for chunk in r.iter_content(chunk_size=8192):
                                f.write(chunk)
                    clip = VideoFileClip(video_path)
                    if clip.duration >= min_duration:
                        video_paths.append(video_path)
                    clip.close()
                return video_paths

            return self._fetch_from_pixabay(query, count, min_duration)
        except Exception as e:
            logger.error(f"Error fetching videos from Pexels: {e}")
            return self._fetch_from_pixabay(query, count, min_duration)

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
        transition_duration = 0.05
        total_transition_time = transition_duration * (len(words) - 1)
        speech_duration = duration * 0.95
        effective_duration = speech_duration - total_transition_time

        word_durations = []
        min_word_time = 0.3
        for word in words:
            char_ratio = len(word) / max(1, total_chars)
            word_time = min_word_time + (effective_duration - min_word_time * len(words)) * char_ratio * 1.2
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

        return clip.fl(lambda gf, t: blur_frame(gf, t))

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

        return clip.fl(lambda gf, t: blur_frame(gf, t))

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
        Create a YouTube Short video with seamless backgrounds and no black screens

        Args:
            title (str): Video title
            script_sections (list): List of dictionaries with text and duration
            background_query (str): Fallback search term for background videos
            output_filename (str): Output file path
            add_captions (bool): Add captions at the bottom
            style (str): Video style
            voice_style (str): Voice style for TTS
            max_duration (int): Maximum duration in seconds (default: 25)
            background_queries (list): List of search terms for background videos, one per segment
            blur_background (bool): Whether to apply blur effect to background videos (default: True)
            edge_blur (bool): Whether to apply edge blur effect to background videos (default: False)
        Returns:
            str: Path to the created video
        """
        # Set output filename if not provided
        if output_filename is None:
            output_filename = os.path.join(self.output_dir, f"short_{int(time.time())}.mp4")

        # Start timing the overall process
        overall_start_time = time.time()
        logger.info(f"Starting YouTube short creation")

        # Calculate total duration and scale if needed
        total_duration = sum(section.get('duration', 5) for section in script_sections)

        if total_duration > max_duration:
            scale_factor = max_duration / total_duration
            logger.info(f"Scaling durations by factor {scale_factor:.2f} to fit max duration of {max_duration}s")
            for section in script_sections:
                section['duration'] *= scale_factor
            total_duration = max_duration

        logger.info(f"Total video duration: {total_duration:.1f}s")

        # Identify intro and outro sections
        intro_section = script_sections[0] if script_sections else None
        outro_section = script_sections[-1] if len(script_sections) > 1 else None

        # Middle sections (excluding intro and outro)
        middle_sections = script_sections[1:-1] if len(script_sections) > 2 else []

        # Calculate optimal number of background segments for middle sections
        if len(middle_sections) == 0:
            num_backgrounds = 1  # Just one background for very short videos
        else:
            # One background per section, but maximum of 5
            num_backgrounds = min(len(middle_sections), 5)

        logger.info(f"Creating video with {num_backgrounds} background segments for middle sections")

        # Prepare background queries
        if background_queries is None or len(background_queries) < num_backgrounds:
            # If no specific queries provided or not enough queries, use the default/fallback
            if background_queries is None:
                background_queries = []

            # Fill remaining slots with the fallback query
            while len(background_queries) < num_backgrounds:
                background_queries.append(background_query)

        # Fetch background videos for each segment
        fetch_start_time = time.time()
        logger.info(f"Starting background video fetch")

        bg_paths = []
        for i in range(num_backgrounds):
            query = background_queries[i]
            segment_paths = self._fetch_videos(query, count=1, min_duration=5)

            if segment_paths:
                bg_paths.extend(segment_paths)
            else:
                # Fallback to main query if this specific query fails
                fallback_paths = self._fetch_videos(background_query, count=1, min_duration=5)
                if fallback_paths:
                    bg_paths.extend(fallback_paths)

        logger.info(f"Completed background video fetch in {time.time() - fetch_start_time:.2f} seconds")

        # Final check if we have any backgrounds
        if not bg_paths:
            raise ValueError("No background videos available. Aborting video creation.")

        # If we have fewer backgrounds than needed, duplicate some
        while len(bg_paths) < num_backgrounds:
            bg_paths.append(random.choice(bg_paths))

        # Transition duration between background clips
        transition_duration = 0.5  # Shorter transitions for better timing

        # Generate audio clips with TTS for each section
        tts_start_time = time.time()
        logger.info(f"Starting TTS audio generation")

        audio_clips = []
        section_durations = []  # Store actual durations after TTS generation

        for i, section in enumerate(script_sections):
            section_text = section["text"]
            section_voice_style = section.get("voice_style", voice_style)
            min_section_duration = section.get("duration", 5)

            # Try to create TTS audio file
            audio_path = os.path.join(self.temp_dir, f"section_{i}.mp3")

            if self.azure_tts:
                try:
                    voice = os.getenv("AZURE_VOICE", "en-US-JennyNeural")
                    self.azure_tts.generate_speech(section_text, output_filename=audio_path)
                except Exception as e:
                    logger.error(f"Azure TTS failed: {e}, falling back to gTTS")
                    try:
                        tts = gTTS(text=section_text, lang='en', slow=False)
                        tts.save(audio_path)
                    except Exception as e2:
                        logger.error(f"gTTS also failed: {e2}, section {i} will be silent")
                        audio_path = None
            else:
                try:
                    tts = gTTS(text=section_text, lang='en', slow=False)
                    tts.save(audio_path)
                except Exception as e:
                    logger.error(f"gTTS failed: {e}, section {i} will be silent")
                    audio_path = None

            # If audio file was created successfully, get its actual duration
            if audio_path and os.path.exists(audio_path):
                try:
                    audio_clip = AudioFileClip(audio_path)
                    actual_duration = audio_clip.duration

                    # Make sure audio is at least as long as specified in JSON
                    if actual_duration < min_section_duration:
                        actual_duration = min_section_duration

                    # Store the final duration
                    section_durations.append(actual_duration)
                    audio_clips.append((i, audio_clip, actual_duration))
                except Exception as e:
                    logger.error(f"Error processing audio for section {i}: {e}")
                    section_durations.append(min_section_duration)
            else:
                # If no audio was created, use minimum duration
                section_durations.append(min_section_duration)

        # Update script sections with actual durations
        for i, duration in enumerate(section_durations):
            if i < len(script_sections):
                script_sections[i]['duration'] = duration

        # Recalculate total duration based on actual audio lengths
        total_duration = sum(section_durations)

        logger.info(f"Completed TTS audio generation in {time.time() - tts_start_time:.2f} seconds")
        logger.info(f"Updated total duration: {total_duration:.1f}s")

        # Process intro section background
        if intro_section:
            intro_duration = intro_section['duration']
            intro_query = background_queries[0] if background_queries else background_query
            intro_bg_path = self._fetch_videos(intro_query, count=1, min_duration=intro_duration)

            if not intro_bg_path:
                # Use first background from main fetch if intro-specific fetch fails
                intro_bg_path = [bg_paths[0]] if bg_paths else None

            if intro_bg_path:
                intro_bg_clip = VideoFileClip(intro_bg_path[0])
                intro_bg_clip = self._process_background_clip(
                    intro_bg_clip,
                    intro_duration,
                    blur_background=blur_background,
                    edge_blur=edge_blur
                )
            else:
                # Create a colored background if no video available
                intro_bg_clip = ColorClip(size=self.resolution, color=(20, 20, 20)).set_duration(intro_duration)

        # Process outro section background
        if outro_section and outro_section != intro_section:
            outro_duration = outro_section['duration']
            outro_query = background_queries[-1] if background_queries else background_query
            outro_bg_path = self._fetch_videos(outro_query, count=1, min_duration=outro_duration)

            if not outro_bg_path:
                # Use last background from main fetch if outro-specific fetch fails
                outro_bg_path = [bg_paths[-1]] if bg_paths else None

            if outro_bg_path:
                outro_bg_clip = VideoFileClip(outro_bg_path[0])
                outro_bg_clip = self._process_background_clip(
                    outro_bg_clip,
                    outro_duration,
                    blur_background=blur_background,
                    edge_blur=edge_blur
                )
            else:
                # Create a colored background if no video available
                outro_bg_clip = ColorClip(size=self.resolution, color=(20, 20, 20)).set_duration(outro_duration)

        # Calculate durations for middle section backgrounds
        middle_section_durations = [section['duration'] for section in middle_sections]
        middle_total_duration = sum(middle_section_durations)

        # Divide middle section backgrounds evenly
        if num_backgrounds > 0 and middle_total_duration > 0:
            bg_segment_durations = []
            segment_size = middle_total_duration / num_backgrounds

            for i in range(num_backgrounds):
                if i == num_backgrounds - 1:
                    # Last segment gets remaining duration
                    duration = middle_total_duration - sum(bg_segment_durations)
                else:
                    duration = segment_size

                # Add transition overlap except for the last segment
                if i < num_backgrounds - 1:
                    duration += transition_duration

                bg_segment_durations.append(duration)
        else:
            bg_segment_durations = []

        # Process middle section backgrounds
        process_start_time = time.time()
        logger.info(f"Starting background processing")

        middle_bg_clips = []

        # Map middle section words to background segments
        section_to_bg_mapping = {}
        cumulative_duration = 0

        for i, section in enumerate(middle_sections):
            section_duration = section['duration']
            section_end_time = cumulative_duration + section_duration

            # Find which background segment(s) this section belongs to
            segment_start = 0
            for bg_idx, bg_duration in enumerate(bg_segment_durations):
                segment_end = segment_start + bg_duration

                # If section overlaps with this segment
                if cumulative_duration < segment_end and section_end_time > segment_start:
                    if i not in section_to_bg_mapping:
                        section_to_bg_mapping[i] = []
                    section_to_bg_mapping[i].append(bg_idx)

                segment_start += bg_duration - (transition_duration if bg_idx < num_backgrounds - 1 else 0)

            cumulative_duration += section_duration

        # Process each background segment
        for i, bg_path in enumerate(bg_paths[:num_backgrounds]):
            try:
                target_duration = bg_segment_durations[i]
                bg_clip = VideoFileClip(bg_path)
                processed_clip = self._process_background_clip(bg_clip, target_duration, blur_background=blur_background, edge_blur=edge_blur)
                middle_bg_clips.append(processed_clip)
            except Exception as e:
                logger.error(f"Error processing background video {i+1}: {str(e)}")
                # Create a colored background as fallback
                fallback_clip = ColorClip(size=self.resolution, color=(20, 20, 20)).set_duration(bg_segment_durations[i])
                middle_bg_clips.append(fallback_clip)

        logger.info(f"Completed background processing in {time.time() - process_start_time:.2f} seconds")

        # Apply crossfade transitions between middle background clips
        if len(middle_bg_clips) > 1:
            compose_start_time = time.time()
            logger.info(f"Starting background composition")

            final_middle_bg = [middle_bg_clips[0]]

            for i in range(1, len(middle_bg_clips)):
                crossfaded = concatenate_videoclips(
                    [final_middle_bg[-1], middle_bg_clips[i].crossfadein(transition_duration)],
                    padding=-transition_duration,
                    method="compose"
                )
                final_middle_bg[-1] = crossfaded

            logger.info(f"Completed background composition in {time.time() - compose_start_time:.2f} seconds")
            middle_bg_clip = final_middle_bg[0]
        elif len(middle_bg_clips) == 1:
            middle_bg_clip = middle_bg_clips[0]
        else:
            middle_bg_clip = None

        # Create section clips with text overlays
        section_clips = []

        # Process intro
        if intro_section:
            intro_text = intro_section['text']
            intro_duration = intro_section['duration']

            # Create title text if provided
            if title:
                    title_clip = self._create_text_clip(
                    title, duration=intro_duration, font_size=70, font_path=self.title_font_path,
                        position=('center', 150), animation="fade", animation_duration=0.8,
                        with_pill=True, pill_color=(0, 0, 0, 160), pill_radius=30
                )
            else:
                title_clip = None

            # Create intro text
            intro_text_clip = self._create_text_clip(
                intro_text, duration=intro_duration, font_size=55, font_path=self.body_font_path,
                    position=('center', 'center'), animation="fade", animation_duration=0.8,
                    with_pill=True, pill_color=(0, 0, 0, 160), pill_radius=30
            )

            # Combine background with text
            clips_to_combine = [intro_bg_clip]
            if title_clip:
                clips_to_combine.append(title_clip)
            clips_to_combine.append(intro_text_clip)

            intro_clip = CompositeVideoClip(clips_to_combine, size=self.resolution)

            # Add audio if available
            for idx, audio_clip, duration in audio_clips:
                if idx == 0:  # First section is intro
                    intro_clip = intro_clip.set_audio(audio_clip)
                    break

            section_clips.append(intro_clip)

        # Process middle sections
        if middle_sections and middle_bg_clip:
            middle_clips = []
            cumulative_time = 0

            for i, section in enumerate(middle_sections):
                section_text = section['text']
                section_duration = section['duration']

                # Create word-by-word text animation
                word_clip = self._create_word_by_word_clip(
                    section_text, duration=section_duration, font_size=60, font_path=self.body_font_path,
                    text_color=(255, 255, 255, 255), pill_color=(0, 0, 0, 160), position=('center', 'center')
                )

                # Check if we need to reset the cumulative time
                if cumulative_time + section_duration > middle_bg_clip.duration:
                    logger.warning(f"Background clip too short ({middle_bg_clip.duration:.2f}s), resetting position")
                    cumulative_time = 0

                # Ensure we don't exceed the background clip's duration
                end_time = min(cumulative_time + section_duration, middle_bg_clip.duration)
                actual_duration = end_time - cumulative_time

                # Extract the correct portion of the background
                section_bg = middle_bg_clip.subclip(cumulative_time, end_time)

                # If we didn't get enough background footage, loop it
                if actual_duration < section_duration:
                    logger.warning(f"Section duration ({section_duration:.2f}s) exceeds available background ({actual_duration:.2f}s), looping")
                    remaining_duration = section_duration - actual_duration

                    # Get footage from the beginning of the clip
                    remaining_end = min(remaining_duration, middle_bg_clip.duration)
                    additional_bg = middle_bg_clip.subclip(0, remaining_end)

                    # Concatenate the two segments
                    section_bg = concatenate_videoclips([section_bg, additional_bg])

                    # Update actual duration with what we now have
                    actual_duration = section_bg.duration

                # Combine background with text
                section_clip = CompositeVideoClip([section_bg, word_clip], size=self.resolution)

                # Add audio if available
                for idx, audio_clip, duration in audio_clips:
                    if idx == i + 1:  # Middle sections start after intro
                        section_clip = section_clip.set_audio(audio_clip)
                        break

                middle_clips.append(section_clip)
                cumulative_time = (cumulative_time + actual_duration) % middle_bg_clip.duration  # Wrap around when we reach the end

            section_clips.extend(middle_clips)

        # Process outro
        if outro_section and outro_section != intro_section:
            outro_text = outro_section['text']
            outro_duration = outro_section['duration']

            # Create outro text
            outro_text_clip = self._create_text_clip(
                outro_text, duration=outro_duration, font_size=55, font_path=self.body_font_path,
                position=('center', 'center'), animation="fade", animation_duration=0.8,
                with_pill=True, pill_color=(0, 0, 0, 160), pill_radius=30
            )

            # Combine background with text
            outro_clip = CompositeVideoClip([outro_bg_clip, outro_text_clip], size=self.resolution)

            # Add audio if available
            for idx, audio_clip, duration in audio_clips:
                if idx == len(script_sections) - 1:  # Last section is outro
                    outro_clip = outro_clip.set_audio(audio_clip)
                    break

            section_clips.append(outro_clip)

        # Add captions at the bottom if requested
        if add_captions:
            for i, section_clip in enumerate(section_clips):
                if i < len(script_sections):
                    section_text = script_sections[i]['text']
                    caption = self._create_text_clip(
                        section_text, duration=section_clip.duration, font_size=40,
                        font_path=self.body_font_path, position=('center', self.resolution[1] - 200),
                        animation="fade", animation_duration=0.5
                    )
                    section_clips[i] = CompositeVideoClip([section_clip, caption], size=self.resolution)

        # Use parallel rendering if available
        try:
            from parallel_renderer import render_clips_in_parallel
            logger.info("Using parallel renderer for improved performance")

            # Create temp directory for parallel rendering
            parallel_temp_dir = os.path.join(self.temp_dir, "parallel_render")
            os.makedirs(parallel_temp_dir, exist_ok=True)

            # Render clips in parallel
            render_start_time = time.time()
            logger.info(f"Starting parallel video rendering")

            output_filename = render_clips_in_parallel(
                section_clips,
                output_filename,
                temp_dir=parallel_temp_dir,
                fps=self.fps,
                preset="veryfast"
            )

            logger.info(f"Completed video rendering in {time.time() - render_start_time:.2f} seconds")
        except (ImportError, Exception) as e:
            logger.warning(f"Parallel renderer not available or failed: {e}. Using standard rendering.")

            # Concatenate all clips
            final_start_time = time.time()
            logger.info(f"Starting standard video rendering")

            final_clip = concatenate_videoclips(section_clips)

        # Write the final video to file
            final_clip.write_videofile(
            output_filename,
            codec="libx264",
            audio_codec="aac",
            fps=self.fps,
            preset="ultrafast",
            threads=4
        )

            logger.info(f"Completed video rendering in {time.time() - final_start_time:.2f} seconds")

        # Print summary of creation process
        overall_duration = time.time() - overall_start_time
        logger.info(f"YouTube short creation completed in {overall_duration:.2f} seconds")
        logger.info(f"Video saved to: {output_filename}")

        # Clean up temporary files
        self._cleanup()

        return output_filename

    def _cleanup(self):
        """Clean up temporary files"""
        try:
            shutil.rmtree(self.temp_dir)
            logger.info("Temporary files cleaned up successfully.")
        except Exception as e:
            logger.warning(f"Error cleaning up temporary files: {str(e)}")


