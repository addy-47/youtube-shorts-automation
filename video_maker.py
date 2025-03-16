import os
import time
import random
import textwrap
import requests
import numpy as np
import logging
import re
from PIL import Image, ImageFilter
from moviepy.editor import (
    VideoFileClip, VideoClip, TextClip, CompositeVideoClip,
    AudioFileClip, concatenate_videoclips, ColorClip, CompositeAudioClip
)
from moviepy.config import change_settings
change_settings({"IMAGEMAGICK_BINARY": "magick"})
from gtts import gTTS
from dotenv import load_dotenv
import shutil
import tempfile

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
        remaining = count

        # Randomly decide which API to try first
        apis = ["pexels", "pixabay"]
        random.shuffle(apis)

        for api in apis:
            # Randomly decide how many videos to fetch from this API
            if api != apis[-1]:  # If not the last API in the list
                # For all but the last API, get a random portion of the remaining videos
                api_count = random.randint(1, remaining)
            else:
                # For the last API, get all remaining videos
                api_count = remaining

            # Fetch videos from the selected API
            if api == "pexels":
                api_videos = self._fetch_from_pexels(query, api_count, min_duration)
            else:  # pixabay
                api_videos = self._fetch_from_pixabay(query, api_count, min_duration)

            videos.extend(api_videos)
            remaining -= len(api_videos)

            # If we've got enough videos, stop
            if remaining <= 0:
                break

        # If we still need more videos, try to get them from any source
        if remaining > 0:
            # Try both APIs again to fill the remaining slots
            for api in ["pexels", "pixabay"]:
                if remaining <= 0:
                    break

                if api == "pexels":
                    api_videos = self._fetch_from_pexels(query, remaining, min_duration)
                else:  # pixabay
                    api_videos = self._fetch_from_pixabay(query, remaining, min_duration)

                videos.extend(api_videos)
                remaining -= len(api_videos)

        # Make sure we don't return more videos than requested
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
        api_key = os.getenv("PIXABAY_API_KEY")
        url = f"https://pixabay.com/api/videos/?key={api_key}&q={query}&min_width=1080&min_height=1920"
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
        return []

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
        if not self.pexels_api_key:
            logger.error("No Pexels API key provided. Cannot fetch videos.")
            return []

        try:
            logger.info(f"Fetching {count} videos matching '{query}' from Pexels")

            # Request videos from Pexels API
            url = f"https://api.pexels.com/videos/search?query={query}&per_page={count*3}&orientation=portrait"
            headers = {"Authorization": self.pexels_api_key}
            response = requests.get(url, headers=headers)

            if response.status_code != 200:
                logger.error(f"Pexels API request failed with status code {response.status_code}")
                return []

            data = response.json()
            videos = data.get("videos", [])

            if not videos:
                logger.warning(f"No videos found for query '{query}'.")
                return []

            # Download and validate videos
            video_paths = []
            for video in videos:
                # Sort by highest quality that fits our criteria
                video_files = sorted(
                    [f for f in video.get("video_files", []) if f.get("width") <= 1080 and f.get("height") >= 1080],
                    key=lambda x: x.get("width", 0),
                    reverse=True
                )

                if video_files and video.get("duration", 0) >= min_duration:
                    file = video_files[0]
                    video_path = os.path.join(self.temp_dir, f"pexels_{video['id']}.mp4")

                    # Download the video
                    with requests.get(file["link"], stream=True) as r:
                        r.raise_for_status()
                        with open(video_path, 'wb') as f:
                            for chunk in r.iter_content(chunk_size=8192):
                                f.write(chunk)

                    # Validate the downloaded video
                    try:
                        clip = VideoFileClip(video_path)
                        if clip.duration >= min_duration:
                            video_paths.append(video_path)
                        clip.close()
                    except Exception as e:
                        logger.warning(f"Downloaded video {video_path} is invalid: {str(e)}")
                        os.remove(video_path)

                    if len(video_paths) >= count:
                        break

            # If we don't have enough videos, repeat the ones we have
            if video_paths and len(video_paths) < count:
                logger.info(f"Only {len(video_paths)} videos fetched. Looping to reach {count}.")
                while len(video_paths) < count:
                    video_paths.append(video_paths[len(video_paths) % len(video_paths)])
            elif not video_paths:
                logger.error("No valid videos downloaded.")
                return []

            return video_paths[:count]

        except Exception as e:
            logger.error(f"Error in Pexels video fetching: {str(e)}")
            return []

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

        # Set position
        if position == 'center':
            pos = ('center', 'center')
        elif position == 'top':
            pos = ('center', 40)
        elif position == 'bottom':
            pos = ('center', self.resolution[1] - text_composite.h - 40)
        else:
            pos = position

        text_composite = text_composite.set_position(pos)

        # Apply animation
        if animation in self.transitions:
            anim_func = self.transitions[animation]
            text_composite = anim_func(text_composite, animation_duration)

        # Create transparent background for the text
        bg = ColorClip(size=self.resolution, color=(0,0,0,0)).set_duration(duration)
        final_clip = CompositeVideoClip([bg, text_composite], size=self.resolution)

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

    def _clean_text_for_gtts(self, text):
        """
        Clean text for Google Text-to-Speech

        Args:
            text (str): Input text

        Returns:
            str: Cleaned text
        """
        text = re.sub(r'[^\w\s.,!?\'"-]', ' ', text)  # Remove special characters
        text = re.sub(r'\s+', ' ', text).strip()  # Remove extra whitespace
        return text

    def _process_background_clip(self, clip, target_duration):
        """
        Process a background clip to match the required duration

        Args:
            clip (VideoClip): The input video clip
            target_duration (float): The required duration

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

    def _create_gradient_background(self, duration):
        """
        Create a visually appealing gradient background as a last resort

        Args:
            duration (float): The required duration

        Returns:
            VideoClip: A gradient background clip
        """
        # Create a function that generates a gradient frame
        def make_frame(t):
            # Create a gradient that changes over time
            img = np.zeros((self.resolution[1], self.resolution[0], 3), dtype=np.uint8)

            # Time-based color values
            r_base = int(127 + 127 * np.sin(t * 0.5))
            g_base = int(127 + 127 * np.sin(t * 0.5 + 2))
            b_base = int(127 + 127 * np.sin(t * 0.5 + 4))

            # Create vertical gradient
            for y in range(self.resolution[1]):
                ratio = y / self.resolution[1]
                r = int(r_base * (1 - ratio) + 30 * ratio)
                g = int(g_base * (1 - ratio) + 30 * ratio)
                b = int(b_base * (1 - ratio) + 60 * ratio)
                img[y, :] = [r, g, b]

            return img

        # Create a video clip from the frame-generating function
        gradient_clip = VideoClip(make_frame, duration=duration)

        return gradient_clip

    def create_youtube_short(self, title, script_sections, background_query="abstract background",
                            output_filename=None, add_captions=False, style="video", voice_style=None, max_duration=25):
        """
        Create a YouTube Short video with seamless backgrounds and no black screens

        Args:
            title (str): Video title
            script_sections (list): List of dictionaries with text and duration
            background_query (str): Search term for background videos
            output_filename (str): Output file path
            add_captions (bool): Add captions at the bottom
            style (str): Video style
            voice_style (str): Voice style for TTS
            max_duration (int): Maximum duration in seconds (default: 25)

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

        # Fetch background videos
        bg_paths = self._fetch_videos(background_query, count=num_backgrounds, min_duration=5)
        if not bg_paths:
            raise ValueError("No background videos available. Aborting video creation.")

        # Process background videos
        bg_clips = []
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
                processed_clip = self._process_background_clip(bg_clip, target_duration)
                processed_bg_clips.append(processed_clip)

            except Exception as e:
                logger.error(f"Error processing background video {i+1}: {str(e)}")
                # Instead of a black screen, use another background or loop an existing one
                if processed_bg_clips:
                    # Use a previously processed clip as a fallback
                    fallback_clip = random.choice(processed_bg_clips).copy()
                    processed_clip = self._process_background_clip(fallback_clip, target_duration)
                    processed_bg_clips.append(processed_clip)
                else:
                    # Try to fetch a new background if we have no processed clips yet
                    try:
                        emergency_bg_paths = self._fetch_videos(background_query, count=1, min_duration=5)
                        if emergency_bg_paths:
                            emergency_clip = VideoFileClip(emergency_bg_paths[0])
                            processed_clip = self._process_background_clip(emergency_clip, target_duration)
                            processed_bg_clips.append(processed_clip)
                        else:
                            # As a last resort, create a generic colorful gradient
                            processed_clip = self._create_gradient_background(target_duration)
                            processed_bg_clips.append(processed_clip)
                    except Exception as e2:
                        logger.error(f"Failed to create fallback background: {str(e2)}")
                        # Create a gradient as absolute last resort
                        processed_clip = self._create_gradient_background(target_duration)
                        processed_bg_clips.append(processed_clip)

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
                extra_clip = self._process_background_clip(last_clip.copy(), needed_duration)

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

        # Generate TTS audio for each section
        audio_clips = []
        current_time = 0
        use_azure = self.azure_tts is not None

        for section in script_sections:
            text = section['text']
            duration = section.get('duration', 5)

            # Clean text for TTS
            tts_text = self.azure_tts.clean_text_for_tts(text) if use_azure else self._clean_text_for_gtts(text)
            tts_path = os.path.join(self.temp_dir, f"tts_{len(audio_clips)}.mp3")

            # Generate speech using Azure TTS or gTTS
            section_voice_style = section.get('voice_style', 'normal')
            tts_path = os.path.join(self.temp_dir, f"tts_{len(audio_clips)}.mp3")
            if use_azure:
                tts_path = self.azure_tts.generate_speech(text, output_filename=tts_path, voice_style=section_voice_style)
            else:
                logger.warning(f"Azure TTS failed: {e}. Using gTTS.")
                tts = gTTS(text=tts_text, lang='en', slow=False)
                tts.save(tts_path)

            # Load audio and adjust section duration if needed
            speech = AudioFileClip(tts_path)
            if speech.duration > duration - 0.5:
                # Increase section duration to fit audio
                section['duration'] = speech.duration + 1
                duration = section['duration']

            # Set audio start time with offset for sync
            speech = speech.set_start(current_time + self.audio_sync_offset)
            audio_clips.append(speech)
            current_time += duration

        # Combine all audio clips
        combined_audio = CompositeAudioClip(audio_clips) if audio_clips else None

        # Generate text overlays for each section
        text_clips = []
        current_time = 0

        for i, section in enumerate(script_sections):
            text = section['text']
            duration = section.get('duration', 5)

            # Handle title and first section
            if i == 0 and title:
                # Add title text
                title_clip = self._create_text_clip(
                    title, duration=duration, font_size=70, font_path=self.title_font_path,
                    position=('center', 150), animation="fade", animation_duration=0.8
                ).set_start(current_time)
                text_clips.append(title_clip)

                # Add first section text if present
                if text:
                    body_clip = self._create_text_clip(
                        text, duration=duration, font_size=50, font_path=self.body_font_path,
                        position=('center', 400), animation="fade", animation_duration=0.8
                    ).set_start(current_time)
                    text_clips.append(body_clip)
            else:
                # Alternate animations for variety
                anim = "fade" if i % 2 == 0 else "fade"

                # Create text overlay
                text_clip = self._create_text_clip(
                    text, duration=duration, font_size=55, font_path=self.body_font_path,
                    position=('center', 'center'), animation=anim, animation_duration=0.8
                ).set_start(current_time)
                text_clips.append(text_clip)

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


