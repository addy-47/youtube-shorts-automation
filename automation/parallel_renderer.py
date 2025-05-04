"""
Parallel video rendering module for YouTube Shorts automation.
Implements multiprocessing capabilities for faster video rendering with keyframe animation support.
"""

import os
import time
import logging
import tempfile
import multiprocessing
from tqdm import tqdm
from moviepy.editor import VideoFileClip, concatenate_videoclips, CompositeVideoClip, AudioFileClip
import gc
import shutil
import subprocess
import re
import json
import numpy as np

logger = logging.getLogger(__name__)

class KeyframeData:
    """
    A serializable container for keyframe animation data.
    
    Instead of using callable functions for positions and sizes, this class
    stores keyframe data that can be directly serialized and used across processes.
    """
    def __init__(self, keyframes=None, attribute_name=None, interpolation="linear"):
        """
        Initialize with keyframe data
        
        Args:
            keyframes: Dictionary mapping times to values, or list of (time, value) tuples
            attribute_name: The attribute this keyframe data controls (position, size, etc.)
            interpolation: Interpolation method ("linear", "cubic", etc.)
        """
        self.keyframes = {}
        self.attribute_name = attribute_name
        self.interpolation = interpolation
        
        if keyframes:
            if isinstance(keyframes, dict):
                self.keyframes = keyframes
            elif isinstance(keyframes, list):
                self.keyframes = {t: v for t, v in keyframes}
                
        # Sort keyframe times for faster lookup
        self.times = sorted(self.keyframes.keys())
    
    def get_value_at_time(self, t):
        """
        Get interpolated value at time t
        
        Args:
            t: Time to get value for
            
        Returns:
            Interpolated value at time t
        """
        # Handle edge cases
        if not self.times:
            return None
            
        if t <= self.times[0]:
            return self.keyframes[self.times[0]]
            
        if t >= self.times[-1]:
            return self.keyframes[self.times[-1]]
        
        # Find surrounding keyframes
        for i in range(len(self.times) - 1):
            t1, t2 = self.times[i], self.times[i + 1]
            if t1 <= t <= t2:
                v1, v2 = self.keyframes[t1], self.keyframes[t2]
                
                # Linear interpolation
                if self.interpolation == "linear":
                    alpha = (t - t1) / (t2 - t1) if t2 > t1 else 0
                    
                    # Handle different value types
                    if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
                        return v1 + alpha * (v2 - v1)
                    elif isinstance(v1, tuple) and isinstance(v2, tuple) and len(v1) == len(v2):
                        return tuple(v1[i] + alpha * (v2[i] - v1[i]) for i in range(len(v1)))
                    else:
                        # For non-numeric types, just return nearest
                        return v1 if alpha < 0.5 else v2
                
                # Add more interpolation methods as needed
                
                return v1  # Default fallback
                
        return self.keyframes[self.times[-1]]  # Fallback to last value
    
    def to_dict(self):
        """Convert to serializable dictionary"""
        return {
            "keyframes": {str(k): v for k, v in self.keyframes.items()},
            "attribute_name": self.attribute_name,
            "interpolation": self.interpolation
        }
    
    @classmethod
    def from_dict(cls, data):
        """Create from dictionary"""
        keyframes = {float(k): v for k, v in data["keyframes"].items()}
        return cls(
            keyframes=keyframes,
            attribute_name=data["attribute_name"],
            interpolation=data["interpolation"]
        )
    
    @classmethod
    def from_callable(cls, func, duration, num_samples=10, attribute_name=None):
        """
        Create keyframe data by sampling a callable function
        
        Args:
            func: Callable function that takes a time value
            duration: Duration to sample over
            num_samples: Number of keyframes to generate
            attribute_name: The attribute this keyframe data controls
            
        Returns:
            KeyframeData instance
        """
        if not callable(func):
            return None
            
        keyframes = {}
        for i in range(num_samples):
            t = i * duration / (num_samples - 1) if num_samples > 1 else 0
            try:
                keyframes[t] = func(t)
except Exception as e:
                logger.warning(f"Error sampling function at time {t}: {e}")
                # Use default values based on common attribute types
                if attribute_name == "position":
                    keyframes[t] = (0, 0)
                elif attribute_name == "size":
                    keyframes[t] = (100, 100)
                else:
                    keyframes[t] = 0
        
        return cls(keyframes, attribute_name)

def render_clip_segment(clip, output_path, fps=30, preset="ultrafast", threads=2, show_progress=True):
    """
    Render a single clip segment to a file.

    Args:
        clip: The MoviePy clip to render
        output_path: Where to save the rendered clip
        fps: Frames per second
        preset: Video encoding preset
        threads: Number of threads for encoding
        show_progress: Whether to show a progress bar

    Returns:
        output_path: The path where the clip was saved
    """
    start_time = time.time()
    logger.info(f"Starting render of segment to {output_path}")

    try:
        # Pre-process audio to fix stuttering/repeating issue
        if clip.audio is not None:
            # Extract audio to a temporary file
            temp_audio_path = output_path.replace('.mp4', '_temp_audio.wav')  # Use WAV for highest quality

            try:
                # Use highest quality settings for audio to prevent stuttering
                clip.audio.write_audiofile(
                    temp_audio_path,
                    fps=48000,  # Higher sample rate for better quality
                    nbytes=4,   # Use 32-bit audio for highest quality
                    codec='pcm_s32le',  # Use uncompressed PCM audio to avoid compression artifacts
                    ffmpeg_params=[
                        "-ac", "2",      # Force stereo output
                        "-ar", "48000",  # Explicitly set sample rate
                        "-sample_fmt", "s32",  # 32-bit audio samples
                        "-bufsize", "8192k"    # Increased buffer size
                    ]
                )

                # Check that audio file was created and has content
                if not os.path.exists(temp_audio_path) or os.path.getsize(temp_audio_path) == 0:
                    logger.warning(f"Audio extraction failed: {temp_audio_path} is empty or missing")
                    # Continue without audio
                else:
                    try:
                        # Load it back as a clean AudioFileClip
                        clean_audio = AudioFileClip(temp_audio_path)

                        # Set audio to the exact duration of the clip to avoid sync issues
                        clean_audio = clean_audio.set_duration(clip.duration)

                        # Replace the clip's audio
                        clip = clip.set_audio(clean_audio)
                        logger.info(f"Successfully preprocessed audio")
                    except Exception as audio_load_err:
                        logger.error(f"Error loading clean audio: {audio_load_err}")
                        # Continue without audio processing
            except Exception as audio_write_err:
                logger.warning(f"Error writing audio to file: {audio_write_err}")
                # Continue without audio processing

        # Use optimized encoding settings
        # Show progress bar with tqdm if requested
        logger_setting = None if show_progress else "bar"

        clip.write_videofile(
            output_path,
            fps=fps,
            codec="libx264",
            audio_codec="aac",
            threads=threads,
            preset=preset,  # Using ultrafast for intermediate files
            audio_bufsize=8192,  # Increased audio buffer size
            ffmpeg_params=[
                "-bufsize", "24M",      # Larger buffer
                "-maxrate", "8M",       # Higher max rate
                "-b:a", "192k",         # Higher audio bitrate
                "-ar", "48000",         # Audio sample rate
                "-pix_fmt", "yuv420p"   # Compatible pixel format for all players
            ],
            logger=logger_setting  # None shows progress bar, "bar" hides it
        )
        duration = time.time() - start_time
        logger.info(f"Completed render of segment {output_path} in {duration:.2f} seconds")

        # Clean up temp audio file if it exists
        if os.path.exists(temp_audio_path):
            try:
                os.remove(temp_audio_path)
            except Exception as clean_err:
                logger.warning(f"Failed to clean up temp audio file: {clean_err}")

        return output_path
    except Exception as e:
        logger.error(f"Error rendering segment {output_path}: {e}")
        raise

def convert_clip_for_parallel(clip):
    """
    Convert a clip to be suitable for parallel processing.

    This converts callable attributes to keyframe data.

    Args:
        clip: The clip to convert

    Returns:
        A clip with all callable attributes converted to keyframe data
    """
    if clip is None:
        return None
        
    # If already a string path, just return it
    if isinstance(clip, str):
        return clip

    # Extract metadata for serialization
    metadata = {
        "duration": clip.duration,
        "size": clip.size
    }
    
    # Check for callable position attribute
    if hasattr(clip, 'pos') and callable(clip.pos):
        # Sample the callable to generate keyframes
        try:
            position_keyframes = KeyframeData.from_callable(
                clip.pos, clip.duration, num_samples=20, attribute_name="position"
            )
            # Apply a static position at the middle time for now
            mid_time = clip.duration / 2
            mid_pos = position_keyframes.get_value_at_time(mid_time)
            clip = clip.set_position(mid_pos)
            
            # Store keyframe data for reference
            metadata["position_keyframes"] = position_keyframes.to_dict()
                    except Exception as e:
            logger.warning(f"Error converting position to keyframes: {e}")
            clip = clip.set_position('center')
            
    # Check for callable size attribute
    if hasattr(clip, 'size') and callable(clip.size):
        try:
            size_keyframes = KeyframeData.from_callable(
                clip.size, clip.duration, num_samples=20, attribute_name="size"
            )
            # Apply a static size at the middle time for now
            mid_time = clip.duration / 2
            mid_size = size_keyframes.get_value_at_time(mid_time)
            clip = clip.resize(mid_size)
            
            # Store keyframe data for reference
            metadata["size_keyframes"] = size_keyframes.to_dict()
        except Exception as e:
            logger.warning(f"Error converting size to keyframes: {e}")
    
    # Handle CompositeVideoClip with subclips
    if isinstance(clip, CompositeVideoClip) and hasattr(clip, 'clips') and clip.clips:
        converted_subclips = []
        for subclip in clip.clips:
            converted_subclip = convert_clip_for_parallel(subclip)
            converted_subclips.append(converted_subclip)
        
        # Recreate the composite clip with converted subclips
        clip = CompositeVideoClip(converted_subclips, size=clip.size)
        
    # Attach metadata to the clip for later reference
    clip._serializable_data = metadata
    
    return clip

def process_clip_for_parallel(task):
    """
    Process a clip for parallel rendering

    Args:
        task: Tuple containing (task_idx, clip, output_path, fps)

    Returns:
        Path to the rendered clip, or None if rendering failed
    """
    try:
        task_idx, clip, output_path, fps = task
        preset = "ultrafast"  # Use ultrafast for intermediate renders
        threads = 2

            logger.info(f"Starting render of clip {task_idx} to {output_path}")

        # Handle pre-rendered clips (paths)
            if isinstance(clip, str) and os.path.exists(clip):
                try:
                    clip = VideoFileClip(clip)
                except Exception as e:
                    logger.error(f"Error loading pre-rendered clip: {e}")
                    return None

            # Render the clip
            try:
            result = render_clip_segment(clip, output_path, fps, preset, threads)
                return result
            except Exception as e:
                logger.error(f"Error rendering clip: {e}")
                return None
    except Exception as e:
        logger.error(f"Error in parallel rendering task: {e}")
        return None

def render_clip_process(mp_tuple):
    """
    Process a single clip for parallel rendering

    Args:
        mp_tuple: Tuple of (idx, clip, output_dir, fps)

    Returns:
        Path to the rendered clip or None if rendering failed
    """
    idx, clip, output_dir, fps = mp_tuple
    output_path = None

    # If clip is already a path string (pre-rendered), just return it
    if isinstance(clip, str) and os.path.exists(clip):
        return clip

    # Generate a unique output path
    output_path = os.path.join(output_dir, f"clip_{idx}_{int(time.time() * 1000)}.mp4")

    try:
        # Ensure the clip is valid
        if clip is None:
            logging.error(f"Clip {idx} is None, skipping")
            return None

        # Different handling based on clip type to optimize performance
        if hasattr(clip, 'write_videofile'):
            # Try to use hardware acceleration if available
            hw_accel = ""
            try:
                # Check for NVIDIA GPU
                nvidia_check = subprocess.run(
                    ["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False
                )
                if nvidia_check.returncode == 0:
                    hw_accel = "h264_nvenc"
                    logging.info(f"Using NVIDIA GPU acceleration for clip {idx}")
            except:
                pass

            # Use optimized settings for intermediate files
            codec = hw_accel if hw_accel else "libx264"
            preset = "ultrafast"  # Fastest preset for intermediate files
            audio_codec = "aac"

            # Use a lower quality for intermediate clips to improve speed
            clip.write_videofile(
                output_path,
                fps=fps,
                preset=preset,
                codec=codec,
                audio_codec=audio_codec,
                threads=2,  # Lower thread count to avoid system overload
                ffmpeg_params=[
                    "-crf", "28",        # Lower quality for temp files is fine
                    "-bufsize", "12M",   # Buffer size
                    "-pix_fmt", "yuv420p" # Compatible format
                ],
                logger=None  # Disable internal progress bars
            )

            # Explicitly close the clip to free memory
            try:
                # Close main clip
                if hasattr(clip, 'close'):
                    clip.close()

                # If clip has audio, make sure to close it
                if hasattr(clip, 'audio') and clip.audio is not None:
                    try:
                        clip.audio.close()
                    except:
                        pass

                # If it's a composite clip, close subclips
                if isinstance(clip, CompositeVideoClip) and hasattr(clip, 'clips'):
                    for subclip in clip.clips:
                        try:
                            if hasattr(subclip, 'close'):
                                subclip.close()
                            # Close audio of subclips too
                            if hasattr(subclip, 'audio') and subclip.audio is not None:
                                subclip.audio.close()
                        except:
                            pass
            except Exception as e:
                logging.debug(f"Error closing clip {idx}: {e}")

            # Force garbage collection
            gc.collect()

            return output_path
        else:
            logging.error(f"Clip {idx} doesn't have write_videofile method, skipping")
            return None

    except Exception as e:
        logging.error(f"Error rendering clip {idx}: {str(e)}")

        # Try to close the clip even if rendering failed
        try:
            if hasattr(clip, 'close'):
                clip.close()
        except:
            pass

        # Force garbage collection
        gc.collect()

        # If the output file was created but is invalid, remove it
        if output_path and os.path.exists(output_path):
            try:
                os.remove(output_path)
                logging.debug(f"Removed incomplete output file: {output_path}")
            except:
                pass

        return None

def render_clips_in_parallel(clips, output_file, fps=30, num_processes=None, logger=None, temp_dir=None, preset="veryfast", codec="libx264", audio_codec="aac", section_info=None):
    """
    Render clips in parallel and concatenate them

    Args:
        clips: List of VideoClip objects to render in parallel
        output_file: Output file to write the final concatenated video
        fps: Frames per second for the output
        num_processes: Number of processes to use for parallel rendering
        logger: Logger object to use for logging
        temp_dir: Optional temporary directory path (if None, a new one will be created)
        preset: FFmpeg preset to use for final encoding (default: "veryfast")
        codec: Video codec to use (default: "libx264")
        audio_codec: Audio codec to use (default: "aac")
        section_info: Optional dictionary mapping clip indices to section info for better debugging

    Returns:
        Path to the output file
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    if num_processes is None:
        num_processes = max(1, min(multiprocessing.cpu_count() - 1, 8))

    logger.info(f"Rendering {len(clips)} clips in parallel using {num_processes} processes")

    # Function to process rendering in a temp directory
    def process_in_temp_dir(temp_directory):
        # Prepare clips for parallel processing
        logger.info("Preparing clips for parallel processing...")
        prepared_clips = []

        for idx, clip in enumerate(clips):
            # Convert any callable attributes to keyframe data
            prepared_clip = convert_clip_for_parallel(clip)
            prepared_clips.append((idx, prepared_clip, temp_directory, fps))
        
        # Process all clips in parallel
        logger.info("Rendering clips in parallel...")
        rendered_paths = []
        
        with multiprocessing.Pool(processes=num_processes) as pool:
            # Process clips in parallel with progress tracking
            for result in tqdm(pool.imap_unordered(render_clip_process, prepared_clips),
                              total=len(prepared_clips),
                              desc="Rendering clips in parallel"):
                if result is not None:
                    # Extract the original clip index from the filename
                    # Filenames should be in format "clip_{idx}_{timestamp}.mp4"
                    match = re.search(r"clip_(\d+)_", os.path.basename(result))
                    if match:
                        orig_idx = int(match.group(1))
                        rendered_paths.append((orig_idx, result))
                        logger.info(f"Clip {orig_idx} rendered: {result}")
                    else:
                        logger.warning(f"Could not extract index from rendered path: {result}")
                        rendered_paths.append((len(rendered_paths), result))

                # Force garbage collection periodically
                if len(rendered_paths) % 5 == 0:
                    gc.collect()

        # Clear prepared_clips list to free memory
        prepared_clips = []
        gc.collect()

        # Make sure we have all the rendered paths
        if not rendered_paths:
            raise ValueError("No clips were successfully rendered")

        logger.info(f"Successfully rendered {len(rendered_paths)} out of {len(clips)} clips")

        # Sort clips by index
        rendered_paths.sort(key=lambda x: x[0])
        render_paths_list = [path for _, path in rendered_paths]
        
        # Concatenate rendered clips
            logger.info("Concatenating clips using FFmpeg...")

            # Create a temporary file list for FFmpeg
            concat_list_path = os.path.join(temp_directory, "concat_list.txt")

            # Write the sorted paths to concat list
            with open(concat_list_path, "w") as f:
            for clip_path in render_paths_list:
                    # Format according to FFmpeg concat protocol
                    f.write(f"file '{os.path.abspath(clip_path)}'\n")

            # Detect hardware acceleration capability
            hw_accel = ""
            try:
                # Check for NVIDIA GPU
                nvidia_check = subprocess.run(
                    ["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False
                )
                if nvidia_check.returncode == 0:
                    hw_accel = "h264_nvenc"
                    logger.info("Using NVIDIA GPU acceleration for final render")
            except Exception as e:
                logger.debug(f"Hardware acceleration check failed: {e}")

        # Set codec and parameters for final render
            final_codec = hw_accel if hw_accel else codec

        # Build the FFmpeg command for final render
            ffmpeg_cmd = [
                "ffmpeg", "-y",
                "-f", "concat",
                "-safe", "0",
                "-i", concat_list_path,
                "-c:v", final_codec,
            "-preset", preset,  # Using veryfast for final render
                "-crf", "23",  # Higher quality for final output
                "-pix_fmt", "yuv420p",
                "-max_muxing_queue_size", "9999",  # Prevent muxing queue issues
                "-c:a", audio_codec,
                "-b:a", "192k",
                output_file
            ]

            logger.info(f"Running FFmpeg concatenation: {' '.join(ffmpeg_cmd)}")

        # Run FFmpeg directly
            process = subprocess.run(
                ffmpeg_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                text=True
            )

            if process.returncode != 0:
            logger.error(f"FFmpeg concatenation failed: {process.stderr}")
            
            # Fall back to alternative approach
            logger.info("Trying alternative FFmpeg approach with copy...")
            
                alt_ffmpeg_cmd = [
                    "ffmpeg", "-y",
                    "-f", "concat",
                    "-safe", "0",
                    "-i", concat_list_path,
                    "-c", "copy",  # Just copy streams without re-encoding
                    "-max_muxing_queue_size", "9999",
                    output_file
                ]

                process = subprocess.run(
                    alt_ffmpeg_cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=False,
                    text=True
                )

            if process.returncode != 0:
                    logger.error(f"Alternative FFmpeg approach failed: {process.stderr}")
                raise Exception(f"FFmpeg concatenation failed: {process.stderr[:500]}...")

        logger.info(f"Successfully concatenated clips to {output_file}")
            return output_file

    # Use provided temp_dir, or check environment, or create a new one
    if temp_dir:
        # Use provided directory
        os.makedirs(temp_dir, exist_ok=True)
        return process_in_temp_dir(temp_dir)
    elif 'TEMP_DIR' in os.environ:
        # Use environment variable if set
        env_temp_dir = os.environ['TEMP_DIR']
        os.makedirs(env_temp_dir, exist_ok=True)
        return process_in_temp_dir(env_temp_dir)
    elif hasattr(tempfile, 'tempdir') and tempfile.tempdir:
        # Use tempfile.tempdir if it's set
        os.makedirs(tempfile.tempdir, exist_ok=True)
        return process_in_temp_dir(tempfile.tempdir)
    else:
        # Create and use temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            return process_in_temp_dir(temp_dir)
