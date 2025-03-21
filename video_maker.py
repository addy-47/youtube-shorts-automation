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

# Configure logging for easier debugging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class YTShortsCreator:
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
        logger.info(f"Fetching {count} videos matching '{query}' from {api}")

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
            url = f"https://pixabay.com/api/videos/?key={self.pixabay_api_key}&q={query}&min_width=1080&min_height=1920"
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                videos = data.get("hits", [])
                video_paths = []
                for video in videos[:count]:
                    video_url = video["videos"]["large"]["url"]
                    video_path = os.path.join(self.temp_dir, f"pixabay_{video['id']}.mp4")
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

            return self._fetch_from_pexels (query, count, min_duration)
        except Exception as e:
            logger.error(f"Error fetching videos from Pixabay: {e}")
            return self._fetch_from_pexels(query, count, min_duration)

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
            url = f"https://api.pexels.com/videos/search?query={query}&per_page={count}&orientation=portrait"
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                videos = data.get("hits", [])
                video_paths = []
                for video in videos[:count]:
                    video_url = video["videos"]["large"]["url"]
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

            return self._fetch_from_pixabay (query, count, min_duration)
        except Exception as e:
            logger.error(f"Error fetching videos from Pixabay: {e}")
            return self._fetch_from_pixabay(query, count, min_duration)

    def _create_text_clip(self, text, duration=5, font_size=60, font_path=None, color='white', position='center', animation="fade", animation_duration=1.0, shadow=True, outline=True):
        """
        Create a text clip with optional effects

        Args:
            text (str): Text content
            duration (float): Duration in seconds
            font_size (int): Font size
            font_path (str): Path to font file
            color (str): Text color
            position (str/tuple): Position on screen
            animation (str): Animation type
            animation_duration (float): Animation duration
            shadow (bool): Add shadow effect
            outline (bool): Add outline effect

        Returns:
            VideoClip: Text clip with effects
        """
        if not font_path:
            font_path = self.body_font_path

        # Create the main text clip
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

        text_composite = text_composite.set_position("center","center")

        # Apply animation
        if animation in self.transitions:
            anim_func = self.transitions[animation]
            text_composite = anim_func(text_composite, animation_duration)

        # Create transparent background for the text
        bg = ColorClip(size=self.resolution, color=(0,0,0,0)).set_duration(duration)
        final_clip = CompositeVideoClip([bg, text_composite], size=self.resolution)

        return final_clip

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

    def _process_background_clip(self, clip, target_duration, blur_background=True):
        """
        Process a background clip to match the required duration

        Args:
            clip (VideoClip): The input video clip
            target_duration (float): The required duration
            blur_background (bool): Whether to apply blur effect to the background

        Returns:
            VideoClip: Processed clip that matches the target duration
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
        if blur_background:
            clip = self.custom_blur(clip, radius=5)

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


    def create_youtube_short(self, title, script_sections, background_query="abstract background",
                            output_filename=None, add_captions=False, style="video", voice_style=None, max_duration=25,
                            background_queries=None, blur_background=False):
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

        Returns:
            str: Path to the created video
        """
        # Set output filename if not provided
        if output_filename is None:
            output_filename = os.path.join(self.output_dir, f"short_{int(time.time())}.mp4")

        # Calculate total duration and scale if needed
        total_duration = sum(section.get('duration', 5) for section in script_sections)

        if total_duration > max_duration:
            scale_factor = max_duration / total_duration
            logger.info(f"Scaling durations by factor {scale_factor:.2f} to fit max duration of {max_duration}s")
            for section in script_sections:
                section['duration'] *= scale_factor
            total_duration = max_duration

        logger.info(f"Total video duration: {total_duration:.1f}s")

        # Calculate optimal number of background segments based on duration
        if total_duration <= 10:
            num_backgrounds = 1  # Just one background for very short videos
        else:
            # Aim for segments of about 8-10 seconds each
            num_backgrounds = max(1, min(5, int(np.ceil(total_duration / 8))))

        logger.info(f"Creating video with {num_backgrounds} background segments for {total_duration:.1f}s")

        # Prepare background queries
        if background_queries is None or len(background_queries) < num_backgrounds:
            # If no specific queries provided or not enough queries, use the default/fallback
            if background_queries is None:
                background_queries = []

            # Fill remaining slots with the fallback query
            while len(background_queries) < num_backgrounds:
                background_queries.append(background_query)

        logger.info(f"Using {len(background_queries[:num_backgrounds])} different background queries")

        # Fetch background videos for each segment
        bg_paths = []
        for i in range(num_backgrounds):
            query = background_queries[i]
            logger.info(f"Fetching background #{i+1} with query: '{query}'")

            # Try to get one video per query
            segment_paths = self._fetch_videos(query, count=1, min_duration=5)

            if segment_paths:
                bg_paths.extend(segment_paths)
            else:
                # Fallback to main query if this specific query fails
                logger.warning(f"No videos found for query '{query}', trying fallback query")
                fallback_paths = self._fetch_videos(background_query, count=1, min_duration=5)
                if fallback_paths:
                    bg_paths.extend(fallback_paths)

        # Final check if we have any backgrounds
        if not bg_paths:
            raise ValueError("No background videos available. Aborting video creation.")

        # If we have fewer backgrounds than needed, duplicate some
        while len(bg_paths) < num_backgrounds:
            bg_paths.append(random.choice(bg_paths))

        transition_duration = 0.5  # Shorter transitions for better timing

        # Calculate exact durations needed for each background segment
        segment_durations = []
        remaining_duration = total_duration

        # Distribute the duration more evenly across background segments
        base_segment_duration = total_duration / num_backgrounds

        for i in range(num_backgrounds):
            if i == num_backgrounds - 1:
                # Last segment gets all remaining duration
                duration = remaining_duration

            else:
                # Each segment gets roughly equal duration
                duration = base_segment_duration

            # Add transition overlap except for the last clip
            if i < num_backgrounds - 1:
                duration += transition_duration

            segment_durations.append(duration)
            remaining_duration -= (duration - (transition_duration if i < num_backgrounds - 1 else 0))

        logger.info(f"Segment durations: {[round(d, 1) for d in segment_durations]}")

        # Create background clips with calculated durations
        processed_bg_clips = []

        for i, bg_path in enumerate(bg_paths):
            try:
                # Load video
                target_duration = segment_durations[i]
                bg_clip = VideoFileClip(bg_path)

                # Process the background clip to match the required duration
                processed_clip = self._process_background_clip(bg_clip, target_duration, blur_background=blur_background)
                processed_bg_clips.append(processed_clip)

            except Exception as e:
                logger.error(f"Error processing background video {i+1}: {str(e)}")
                # Instead of a black screen, use another background or loop an existing one
                if processed_bg_clips:
                    # Use a previously processed clip as a fallback
                    fallback_clip = random.choice(processed_bg_clips).copy()
                    processed_clip = self._process_background_clip(fallback_clip, target_duration, blur_background=blur_background)
                    processed_bg_clips.append(processed_clip)
                else:
                    # Try to fetch a new background if we have no processed clips yet
                    try:
                        emergency_bg_paths = self._fetch_videos(background_query, count=1, min_duration=5)
                        if emergency_bg_paths:
                            emergency_clip = VideoFileClip(emergency_bg_paths[0])
                            processed_clip = self._process_background_clip(emergency_clip, target_duration, blur_background=blur_background)
                            processed_bg_clips.append(processed_clip)
                    except Exception as e2:
                        logger.error(f"Failed to create background. ABORTING{str(e2)}")

        # Apply crossfade transitions between background clips
        final_bg_clips = [processed_bg_clips[0]]

        for i in range(1, len(processed_bg_clips)):
            # Create the crossfade effect
            crossfaded = concatenate_videoclips(
                [final_bg_clips[-1], processed_bg_clips[i].crossfadein(transition_duration)],
                padding=-transition_duration,
                method="compose"
            )

            final_bg_clips[-1] = crossfaded

        # Concatenate all background clips into one seamless background
        background = concatenate_videoclips(final_bg_clips, method="compose")

        # Double-check the background duration against total_duration
        if abs(background.duration - total_duration) > 0.5:  # Allow small rounding differences
            logger.warning(f"Background duration mismatch: {background.duration:.1f}s vs expected {total_duration:.1f}s")
            if background.duration < total_duration:
                # Instead of extending with black, create a looped version of the last clip
                needed_duration = total_duration - background.duration
                last_clip = processed_bg_clips[-1]

                # Create a copy of the last clip and loop it as needed
                extra_clip = self._process_background_clip(last_clip.copy(), needed_duration, blur_background=blur_background)

                # Add crossfade to the extension
                extra_clip = extra_clip.crossfadein(transition_duration)
                extended_background = concatenate_videoclips(
                    [background, extra_clip],
                    padding=-transition_duration,
                    method="compose"
                )
                background = extended_background
            else:
                # Trim if too long
                background = background.subclip(0, total_duration)

        logger.info(f"Final background duration: {background.duration:.1f}s")

        # Generate TTS audio for each section and adjust section durations as needed
        audio_clips = []
        current_time = 0
        use_azure = self.azure_tts is not None

        # First pass: generate TTS audio and adjust section durations if needed
        logger.info("Generating TTS audio and adjusting section durations")

        # Process each section to generate TTS and calculate precise timing
        for i, section in enumerate(script_sections):
            text = section['text']
            original_duration = section.get('duration', 5)

            tts_path = os.path.join(self.temp_dir, f"tts_{i}.mp3")

            # Generate speech using Azure TTS or gTTS
            section_voice_style = section.get('voice_style', 'normal')
            if use_azure:
                try:
                    tts_path = self.azure_tts.generate_speech(
                        text,
                        output_filename=tts_path,
                        voice_style=section_voice_style
                    )
                except Exception as e:
                    logger.warning(f"Azure TTS failed: {e}. Using gTTS.")
                    # Use a slightly lower rate for gTTS to improve sync (0.9 = 90% of normal speed)
                    tts = gTTS(text, lang='en', slow=True)
                    tts.save(tts_path)
            else:
                # Use a slightly lower rate to improve synchronization
                tts = gTTS(text, lang='en', slow=True)
                tts.save(tts_path)

            # Load audio and get actual speech duration
            speech = AudioFileClip(tts_path)
            speech_duration = speech.duration

            # Count words for timing calculations
            words = text.split()
            word_count = len(words)

            # Store speech info for later use
            section['word_count'] = word_count
            section['speech_duration'] = speech_duration

            # Adjust section duration if speech is longer than the allocated time
            # Add padding for better timing (10% extra)
            if speech_duration > original_duration - 0.5:
                padding = speech_duration * 0.15  # Add 15% extra time for better pacing
                new_duration = speech_duration + padding
                section['duration'] = new_duration
                logger.info(f"Section {i+1}: Adjusted duration from {original_duration:.1f}s to {new_duration:.1f}s")
            else:
                section['duration'] = original_duration

            # Set audio start time with offset for sync
            # Delay speech slightly to allow for intro animation
            speech = speech.set_start(current_time + 0.2)  # 200ms delay for better sync
            audio_clips.append(speech)

            # Update current time for next section
            current_time += section['duration']

        # Log final timings for each section
        logger.info("Final section durations after TTS adjustment:")
        for i, section in enumerate(script_sections):
            logger.info(f"Section {i+1}: Speech={section.get('speech_duration', 0):.1f}s, " +
                       f"Final Duration={section.get('duration', 0):.1f}s, " +
                       f"Words={section.get('word_count', 0)}")

        # Combine all audio clips
        combined_audio = CompositeAudioClip(audio_clips) if audio_clips else None

        # Recalculate total duration after TTS generation as it might have changed
        updated_total_duration = sum(section.get('duration', 5) for section in script_sections)

        # Check if total duration has increased due to TTS
        if updated_total_duration > total_duration:
            logger.info(f"Total duration increased from {total_duration:.1f}s to {updated_total_duration:.1f}s due to TTS")

            # If background is shorter than the new duration, extend it
            if background.duration < updated_total_duration:
                needed_duration = updated_total_duration - background.duration + 0.8 # +0.8 to avoid timing issues
                logger.info(f"Extending background by {needed_duration:.1f}s to match new duration")

                # Use last clip to create additional background
                last_clip = processed_bg_clips[-1].copy()
                extra_clip = self._process_background_clip(last_clip, needed_duration, blur_background=blur_background)

                # Add crossfade to the extension
                extra_clip = extra_clip.crossfadein(transition_duration)
                extended_background = concatenate_videoclips(
                    [background, extra_clip],
                    padding=-transition_duration,
                    method="compose"
                )
                background = extended_background

            # Update the duration for reference
            total_duration = updated_total_duration

        logger.info(f"Final background duration: {background.duration:.1f}s vs total content duration: {total_duration:.1f}s")

        # Generate text overlays for each section
        text_clips = []
        current_time = 0

        for i, section in enumerate(script_sections):
            text = section['text']
            duration = section.get('duration', 5)

            # Handle title and first section (intro) and last section (outro)
            if i == 0 or i == len(script_sections) - 1:  # Intro or outro
                # Add title text for intro
                if i == 0 and title:
                    title_clip = self._create_text_clip(
                        title, duration=duration, font_size=70, font_path=self.title_font_path,
                        position=('center', 150), animation="fade", animation_duration=0.8
                    ).set_start(current_time)
                    text_clips.append(title_clip)

                # Add section text with regular style for intro/outro
                text_clip = self._create_text_clip(
                    text, duration=duration, font_size=55, font_path=self.body_font_path,
                    position=('center', 'center'), animation="fade", animation_duration=0.8
                ).set_start(current_time)
                text_clips.append(text_clip)
            else:
                # Middle sections - use word-by-word animation
                word_clip = self._create_word_by_word_clip(
                    text, duration=duration, font_size=60, font_path=self.body_font_path,
                    text_color=(255, 255, 255, 255), pill_color=(0, 0, 0, 160), position=('center', 'center')
                ).set_start(current_time)
                text_clips.append(word_clip)

            current_time += duration

        # Add captions at the bottom if requested
        if add_captions:
            caption_start_time = 0
            for section in script_sections:
                caption = self._create_text_clip(
                    section['text'], duration=section.get('duration', 5), font_size=40,
                    font_path=self.body_font_path, position=('center', self.resolution[1] - 200),
                    animation="fade", animation_duration=0.5
                ).set_start(caption_start_time)
                text_clips.append(caption)
                caption_start_time += section.get('duration', 5)

        # Combine background and text overlays
        final_clips = [background] + text_clips

        # Create final video with audio
        final_video = CompositeVideoClip(final_clips, size=self.resolution)
        if combined_audio:
            final_video = final_video.set_audio(combined_audio)

        # Write the final video to file
        final_video.write_videofile(
            output_filename,
            codec="libx264",
            audio_codec="aac",
            fps=self.fps,
            preset="fast",
            threads=4
        )

        return output_filename

    def _cleanup(self):
        """Clean up temporary files"""
        try:
            shutil.rmtree(self.temp_dir)
            logger.info("Temporary files cleaned up successfully.")
        except Exception as e:
            logger.warning(f"Error cleaning up temporary files: {str(e)}")


