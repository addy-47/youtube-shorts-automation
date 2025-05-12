import os
import time
import logging
import concurrent.futures
from typing import List, Dict, Any, Optional
from moviepy  import AudioFileClip, concatenate_audioclips
from helper.minor_helper import measure_time
from gtts import gTTS
from dotenv import load_dotenv

# Set up logging
logger = logging.getLogger(__name__)

load_dotenv()

# Get temp directory from environment variable or use default
TEMP_DIR = os.getenv("TEMP_DIR", "D:\\youtube-shorts-automation\\temp")
# Create audio subdirectory
audio_temp_dir = os.path.join(TEMP_DIR, "audio_clips")
os.makedirs(audio_temp_dir, exist_ok=True)  # Create temp directory if it doesn't exist

class AudioHelper:
    def __init__(self, temp_dir=None):
        """
        Initialize audio helper with necessary settings

        Args:
            temp_dir (str): Directory to save temporary audio files
        """
        self.temp_dir = temp_dir or audio_temp_dir
        os.makedirs(self.temp_dir, exist_ok=True)

        # Initialize TTS engines
        self.azure_tts = None
        self.google_tts = None

        # Initialize Google Cloud TTS if configured
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

    @measure_time
    def create_tts_audio(self, text, filename=None, voice_style="none"):
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

                return self.azure_tts.generate_speech(text, output_filename=filename, voice_style=style)
            except Exception as e:
                logger.error(f"Azure TTS failed: {e}, falling back to gTTS")

        # Fall back to gTTS if all else fails
        try:
            logger.info("Using gTTS as fallback for TTS")
            tts = gTTS(text=text, lang='en', slow=False)
            tts.save(filename)
            return filename
        except Exception as e:
            logger.error(f"gTTS failed: {e}")
            return None

    @measure_time
    def generate_audio_clips_parallel(self, script_sections, voice_style=None, max_workers=None):
        """
        Generate audio clips for all script sections in parallel

        Args:
            script_sections (list): List of script sections with 'text' key
            voice_style (str): Voice style to use
            max_workers (int): Maximum number of concurrent workers

        Returns:
            list: List of audio file paths
        """
        start_time = time.time()
        logger.info(f"Generating {len(script_sections)} audio clips in parallel")

        def process_section(section):
            section_voice = section.get('voice_style', voice_style)
            text = section.get('text', '')
            section_id = section.get('id', int(time.time()))
            filename = os.path.join(self.temp_dir, f"audio_{section_id}.mp3")

            return self.create_tts_audio(text, filename, section_voice)

        workers = max_workers or min(len(script_sections), os.cpu_count() * 2)
        audio_files = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(process_section, section): i
                      for i, section in enumerate(script_sections)}

            for future in concurrent.futures.as_completed(futures):
                idx = futures[future]
                try:
                    audio_path = future.result()
                    if audio_path:
                        # Store the result in the correct order
                        while len(audio_files) <= idx:
                            audio_files.append(None)
                        audio_files[idx] = audio_path
                except Exception as e:
                    logger.error(f"Error generating audio for section {idx}: {e}")

        total_time = time.time() - start_time
        logger.info(f"Generated {len(audio_files)} audio clips in {total_time:.2f} seconds")

        # Filter out None values
        audio_files = [f for f in audio_files if f]

        return audio_files

    @measure_time
    def combine_audio_clips(self, audio_files, output_filename=None):
        """
        Combine multiple audio clips into a single file

        Args:
            audio_files (list): List of audio file paths
            output_filename (str): Output file path

        Returns:
            str: Path to combined audio file
        """
        if not audio_files:
            logger.warning("No audio files to combine")
            return None

        if not output_filename:
            output_filename = os.path.join(self.temp_dir, f"combined_audio_{int(time.time())}.mp3")

        try:
            clips = [AudioFileClip(f) for f in audio_files]
            combined = concatenate_audioclips(clips)
            combined.write_audiofile(output_filename, logger=None)

            # Close all clips to release resources
            for clip in clips:
                clip.close()
            combined.close()

            return output_filename
        except Exception as e:
            logger.error(f"Error combining audio clips: {e}")
            return None

    @measure_time
    def process_audio_for_script(self, script_sections, voice_style=None, max_workers=None):
        """
        Process audio for all script sections and return audio files with durations

        Args:
            script_sections (list): List of script sections
            voice_style (str): Voice style to use
            max_workers (int): Maximum number of concurrent workers

        Returns:
            list: List of dicts with audio file paths and duration info
        """
        # Generate audio for all sections in parallel
        audio_files = self.generate_audio_clips_parallel(
            script_sections, voice_style, max_workers
        )

        # Get durations for each audio file
        audio_data = []
        for i, audio_file in enumerate(audio_files):
            if audio_file:
                try:
                    clip = AudioFileClip(audio_file)
                    duration = clip.duration
                    clip.close()

                    audio_data.append({
                        'path': audio_file,
                        'duration': duration,
                        'section_idx': i
                    })
                except Exception as e:
                    logger.error(f"Error getting audio duration for {audio_file}: {e}")

        return audio_data
