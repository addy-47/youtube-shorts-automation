"""
Parallel video rendering module for YouTube Shorts automation.
Implements multiprocessing capabilities for faster video rendering while maintaining original quality and transitions.
"""

import os
import time
import logging
import tempfile
import multiprocessing
from tqdm import tqdm
from moviepy.editor import VideoFileClip, concatenate_videoclips, CompositeVideoClip, AudioFileClip
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import gc
import shutil
import subprocess

logger = logging.getLogger(__name__)

# Simple workaround flag to avoid patching errors
USING_DILL = False

# Only try to load dill if needed
try:
    import dill

    # Configure dill with safe settings
    dill.settings['recurse'] = True

    # Basic verification that dill works
    test_func = lambda x: x+1
    test_data = {"func": test_func}
    serialized = dill.dumps(test_data)
    deserialized = dill.loads(serialized)

    # If we got here, dill is working
    USING_DILL = True
    # Avoid patching multiprocessing, just use dill for explicit serialization

except ImportError:
    USING_DILL = False
    logger.info("Dill not available. Using standard pickle for serialization.")
except Exception as e:
    logger.warning(f"Error setting up dill: {e}")
    USING_DILL = False

def render_clip_segment(clip, output_path, fps=30, preset="veryfast", threads=2, show_progress=True):
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
            preset=preset,
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

# Pre-rendering function for complex clips to avoid serialization issues
def prerender_complex_clip(clip, temp_dir, idx, fps):
    """
    Pre-render a complex clip to avoid serialization issues

    This function takes a complex clip that may have callable attributes
    or other features that cause serialization problems, simplifies it
    by fixing those attributes, and renders it to a temporary file.

    Args:
        clip: The complex clip to pre-render
        temp_dir: Directory to store the temporary file
        idx: Index of the clip for logging
        fps: Frames per second for rendering

    Returns:
        Path to the pre-rendered file, or None if pre-rendering failed
    """
    # Generate a unique filename
    temp_path = os.path.join(temp_dir, f"prerender_{idx}_{int(time.time())}.mp4")

    # Start with original clip
    needs_cleaning = False
    modified_clip = None

    try:
        # Check if the main clip has callable attributes
        has_callable_pos = hasattr(clip, 'pos') and callable(clip.pos)
        has_callable_size = hasattr(clip, 'size') and callable(clip.size)

        # Check if it's a composite with subclips needing fixes
        has_complex_subclips = False
        modified_subclips = []

        if isinstance(clip, CompositeVideoClip) and hasattr(clip, 'clips') and clip.clips:
            # For each subclip, check if it has callable attributes
            for subclip in clip.clips:
                subclip_modified = False
                fixed_subclip = subclip

                # Check and fix position
                if hasattr(subclip, 'pos') and callable(subclip.pos):
                    try:
                        # Sample the position at the middle of the duration
                        mid_time = subclip.duration / 2
                        mid_pos = subclip.pos(mid_time)
                        fixed_subclip = subclip.set_position(mid_pos)
                        subclip_modified = True
                        logger.debug(f"Fixed callable position in subclip of clip {idx}")
                    except Exception as e:
                        logger.warning(f"Failed to fix callable position in subclip: {e}")
                        # If fixing fails, use center position
                        fixed_subclip = subclip.set_position('center')
                        subclip_modified = True

                # Check and fix size
                if hasattr(subclip, 'size') and callable(subclip.size):
                    try:
                        # Sample the size at the middle of the duration
                        mid_time = subclip.duration / 2
                        mid_size = subclip.size(mid_time)
                        fixed_subclip = fixed_subclip.resize(mid_size)
                        subclip_modified = True
                        logger.debug(f"Fixed callable size in subclip of clip {idx}")
                    except Exception as e:
                        logger.warning(f"Failed to fix callable size in subclip: {e}")

                # Add the subclip to our list (fixed or original)
                modified_subclips.append(fixed_subclip)

                # Track if any subclips were modified
                has_complex_subclips = has_complex_subclips or subclip_modified

            # Only recreate the composite if any subclips were modified
            if has_complex_subclips:
                needs_cleaning = True
                try:
                    modified_clip = CompositeVideoClip(modified_subclips, size=clip.size)
                    logger.info(f"Created new composite clip with fixed subclips for clip {idx}")
                except Exception as e:
                    logger.warning(f"Failed to recreate composite clip: {e}")
                    modified_clip = None

        # Handle main clip callable attributes
        if has_callable_pos or has_callable_size:
            needs_cleaning = True
            # Use the modified clip if we already created one, otherwise start with the original
            base_clip = modified_clip if modified_clip is not None else clip

            try:
                # Fix position if needed
                if has_callable_pos:
                    try:
                        mid_time = base_clip.duration / 2
                        mid_pos = base_clip.pos(mid_time)
                        base_clip = base_clip.set_position(mid_pos)
                        logger.debug(f"Fixed callable position in main clip {idx}")
                    except Exception as e:
                        logger.warning(f"Failed to fix callable position in main clip: {e}")
                        base_clip = base_clip.set_position('center')

                # Fix size if needed
                if has_callable_size:
                    try:
                        mid_time = base_clip.duration / 2
                        mid_size = base_clip.size(mid_time)
                        base_clip = base_clip.resize(mid_size)
                        logger.debug(f"Fixed callable size in main clip {idx}")
                    except Exception as e:
                        logger.warning(f"Failed to fix callable size in main clip: {e}")

                modified_clip = base_clip
            except Exception as e:
                logger.warning(f"Error while fixing main clip attributes: {e}")
                modified_clip = None

        # If we need cleaning but couldn't create a modified clip, just use the original
        if needs_cleaning and modified_clip is None:
            logger.warning(f"Using original clip for {idx} despite cleaning being needed")
            modified_clip = clip
        elif not needs_cleaning:
            # No cleaning needed, use original clip
            logger.info(f"No attribute fixing needed for clip {idx}, but still pre-rendering")
            modified_clip = clip

        # Render to file
        logger.info(f"Pre-rendering clip {idx} to {temp_path}")

        # Use direct FFmpeg when possible for faster rendering
        try:
            # Try with hardware acceleration first if available
            hw_accel = ""
            try:
                # Check for NVIDIA GPU
                nvidia_check = subprocess.run(
                    ["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False
                )
                if nvidia_check.returncode == 0:
                    hw_accel = "h264_nvenc"
                    logger.info("Using NVIDIA GPU acceleration for pre-rendering")
            except:
                pass

            # Use optimized settings for temporary files
            codec = hw_accel if hw_accel else "libx264"
            preset = "fast" if hw_accel else "ultrafast"  # Fastest preset for temp files

            modified_clip.write_videofile(
                temp_path,
                fps=fps,
                preset=preset,
                codec=codec,
                audio_codec="aac",
                threads=4,
                ffmpeg_params=[
                    "-crf", "28",        # Lower quality for temp files is fine
                    "-bufsize", "12M",   # Buffer size
                    "-pix_fmt", "yuv420p", # Compatible format
                    "-progress", "pipe:1" # Show progress
                ],
                logger="bar"
            )

        except Exception as e:
            logger.warning(f"Error with optimized rendering, falling back to basic: {e}")
            # Fall back to basic rendering
            modified_clip.write_videofile(
                temp_path,
                fps=fps,
                preset="ultrafast",  # Use fastest preset for temp files
                codec="libx264",
                audio_codec="aac",
                threads=2,
                ffmpeg_params=["-crf", "28"],  # Lower quality for temp files is fine
                logger="bar"
            )

        logger.info(f"Successfully pre-rendered clip {idx} to {temp_path}")

        # Close the modified clip to free memory
        try:
            # Close the modified clip if it's different from the original
            if modified_clip is not clip:
                modified_clip.close()

            # If original clip has an audio attribute, close it
            if hasattr(clip, 'audio') and clip.audio is not None:
                try:
                    clip.audio.close()
                except:
                    pass
        except Exception as e:
            logger.debug(f"Error closing clip: {e}")

        return temp_path

    except Exception as e:
        logger.error(f"Error pre-rendering clip {idx}: {e}")
        # Clean up temp file if it exists but is incomplete
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass
        return None

# Central function for processing clips that avoids lambdas
def process_clip_for_parallel(task):
    """
    Process a clip for parallel rendering

    This function works with both old-style (7-parameter) and new-style (>3-parameter) tasks
    for backward compatibility.

    Args:
        task: Tuple containing task parameters

    Returns:
        Path to the rendered clip, or None if rendering failed
    """
    try:
        # Handle both old-style and new-style task parameters
        if len(task) >= 7:  # Old style (clip, output_path, fps, preset, threads, idx, is_prerendered)
            clip, output_path, fps_val, preset_val, threads_val, idx, is_prerendered = task
            logger.info(f"Starting render of clip {idx} to {output_path}")

            # If clip is already pre-rendered, just load and render it
            if is_prerendered:
                try:
                    if isinstance(clip, str) and os.path.exists(clip):
                        pre_clip = VideoFileClip(clip)
                        result = render_clip_segment(pre_clip, output_path, fps_val, preset_val, threads_val)
                        try:
                            pre_clip.close()
                        except:
                            pass
                        return result
                except Exception as e:
                    logger.error(f"Error loading pre-rendered clip: {e}")
                    return None

            # For normal clips, render directly
            return render_clip_segment(clip, output_path, fps_val, preset_val, threads_val)

        elif len(task) >= 4:  # New style (task_idx, clip, output_path, fps, [results])
            task_idx, clip, output_path, fps = task[:4]
            results_list = task[4] if len(task) > 4 else None

            logger.info(f"Starting render of clip {task_idx} to {output_path}")

            # Handle pre-rendered clips
            if isinstance(clip, str) and os.path.exists(clip):
                try:
                    clip = VideoFileClip(clip)
                except Exception as e:
                    logger.error(f"Error loading pre-rendered clip: {e}")
                    if results_list is not None:
                        results_list[task_idx] = None
                    return None

            # Render the clip
            try:
                result = render_clip_segment(clip, output_path, fps, "veryfast", 2)

                # If we have a results list, update it
                if results_list is not None:
                    results_list[task_idx] = result

                return result
            except Exception as e:
                logger.error(f"Error rendering clip: {e}")
                if results_list is not None:
                    results_list[task_idx] = None
                return None
        else:
            logger.error(f"Invalid task format: {task}")
            return None

    except Exception as e:
        logger.error(f"Error in parallel rendering task: {e}")
        return None

# Helper function to create fixed static clip versions
def create_static_clip_version(clip):
    """Create a static version of a clip with all dynamic attributes converted to static"""
    # If already a string path, return as is
    if isinstance(clip, str):
        return clip, True

    # Fix positions if needed
    if hasattr(clip, 'pos') and callable(clip.pos):
        try:
            mid_pos = clip.pos(clip.duration / 2)
            clip = clip.set_position(mid_pos)
        except:
            clip = clip.set_position('center')

    # Fix sizes if needed
    if hasattr(clip, 'size') and callable(clip.size):
        try:
            mid_size = clip.size(clip.duration / 2)
            clip = clip.resize(mid_size)
        except:
            pass

    # Return the fixed clip
    return clip, False

def is_complex_clip(clip):
    """
    Determine if a clip is complex and needs pre-rendering to avoid serialization issues.

    Args:
        clip: The clip to check

    Returns:
        bool: True if the clip is complex and needs pre-rendering, False otherwise
    """
    # Check if it's already a file path (string)
    if isinstance(clip, str):
        return False

    # Check for callable position attribute
    if hasattr(clip, 'pos') and callable(clip.pos):
        return True

    # Check for callable size attribute
    if hasattr(clip, 'size') and callable(clip.size):
        return True

    # Check if it's a composite clip with subclips
    if isinstance(clip, CompositeVideoClip) and hasattr(clip, 'clips') and clip.clips:
        # Check each subclip for complexity
        for subclip in clip.clips:
            if hasattr(subclip, 'pos') and callable(subclip.pos):
                return True
            if hasattr(subclip, 'size') and callable(subclip.size):
                return True

    # Check for custom attributes that might cause serialization issues
    # This includes checking for lambda functions or other non-serializable objects
    for attr_name in dir(clip):
        try:
            # Skip magic methods and private attributes
            if attr_name.startswith('__') or attr_name.startswith('_'):
                continue

            # Get the attribute
            attr = getattr(clip, attr_name)

            # Check if it's a callable (function or method)
            if callable(attr) and not hasattr(attr, '__self__'):  # Exclude bound methods
                # This is a potential serialization issue
                return True
        except:
            # If we can't access an attribute, it might be problematic
            pass

    return False

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
            preset = "fast" if hw_accel else "ultrafast"  # Fastest preset for temp files
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

        # Try to close the clip even if rendering failed, with comprehensive cleanup
        try:
            if hasattr(clip, 'close'):
                clip.close()

            # Also try to clean up audio
            if hasattr(clip, 'audio') and clip.audio is not None:
                try:
                    clip.audio.close()
                except:
                    pass

            # If it's a composite clip, close all subclips
            if isinstance(clip, CompositeVideoClip) and hasattr(clip, 'clips'):
                for subclip in clip.clips:
                    try:
                        if hasattr(subclip, 'close'):
                            subclip.close()
                        if hasattr(subclip, 'audio') and subclip.audio is not None:
                            subclip.audio.close()
                    except:
                        pass
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

def render_clips_in_parallel(clips, output_file, fps=30, num_processes=None, logger=None, temp_dir=None, preset="veryfast", codec="libx264", audio_codec="aac"):
    """
    Render clips in parallel and concatenate them

    Args:
        clips: List of VideoClip objects to render in parallel
        output_file: Output file to write the final concatenated video
        fps: Frames per second for the output
        num_processes: Number of processes to use for parallel rendering
        logger: Logger object to use for logging
        temp_dir: Optional temporary directory path (if None, a new one will be created)
        preset: FFmpeg preset to use for encoding (default: "veryfast")
        codec: Video codec to use (default: "libx264")
        audio_codec: Audio codec to use (default: "aac")

    Returns:
        Path to the output file
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    if num_processes is None:
        num_processes = max(1, min(multiprocessing.cpu_count() - 1, 8))

    logger.info(f"Rendering {len(clips)} clips in parallel using {num_processes} processes with preset: {preset}")

    # Function to process rendering in a temp directory
    def process_in_temp_dir(temp_directory):
        # Store paths to rendered clips rather than clip objects to save memory
        processed_clips_paths = []

        # First step: Pre-render all complex clips to avoid serialization issues
        logger.info("Pre-processing complex clips...")
        for idx, clip in enumerate(clips):
            if is_complex_clip(clip):
                logger.debug(f"Pre-rendering complex clip {idx}")
                try:
                    # Pre-render complex clips
                    clip_path = prerender_complex_clip(clip, temp_directory, idx, fps)
                    if clip_path:
                        processed_clips_paths.append((idx, clip_path))
                        logger.info(f"Successfully pre-rendered complex clip {idx}")
                    else:
                        logger.warning(f"Failed to pre-render clip {idx}, skipping")

                    # Explicitly close original clip to free memory
                    try:
                        if hasattr(clip, 'close'):
                            clip.close()
                    except Exception as e:
                        logger.debug(f"Error closing clip {idx}: {e}")
                except Exception as e:
                    logger.error(f"Error pre-rendering clip {idx}: {e}")
            else:
                # For simple clips, add to process list
                processed_clips_paths.append((idx, clip))

        # Second step: Process all clips in parallel
        logger.info("Rendering clips in parallel...")
        mp_clips = []
        for idx, clip_or_path in processed_clips_paths:
            # If it's already a path (pre-rendered), use it directly
            if isinstance(clip_or_path, str) and os.path.exists(clip_or_path):
                mp_clips.append((idx, clip_or_path, temp_directory, fps))
            else:
                # Otherwise, it's a clip that needs rendering
                mp_clips.append((idx, clip_or_path, temp_directory, fps))

        # Clear processed_clips_paths to free memory
        processed_clips_paths = []

        # Set up a multiprocessing pool and render each clip in parallel
        rendered_paths = []
        with multiprocessing.Pool(processes=num_processes) as pool:
            # Process clips in parallel with progress tracking
            for result in tqdm(pool.imap_unordered(render_clip_process, mp_clips),
                              total=len(mp_clips),
                              desc="Rendering clips in parallel"):
                if result is not None:
                    rendered_paths.append(result)
                    logger.debug(f"Clip rendered: {result}")

                # Force garbage collection periodically
                if len(rendered_paths) % 5 == 0:
                    gc.collect()

        # Clear mp_clips list to free memory
        mp_clips = []
        gc.collect()

        if not rendered_paths:
            raise ValueError("No clips were successfully rendered")

        logger.info(f"Successfully rendered {len(rendered_paths)} out of {len(clips)} clips")

        # Third step: Concatenate rendered clips
        try:
            # Sort clips by index if needed
            # For direct file concatenation, we don't need to sort if using concat filter
            logger.info("Concatenating clips using FFmpeg...")

            # Create a temporary file list for FFmpeg
            concat_list_path = os.path.join(temp_directory, "concat_list.txt")
            with open(concat_list_path, "w") as f:
                for clip_path in rendered_paths:
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

            # Set codec and parameters
            final_codec = hw_accel if hw_accel else codec
            final_preset = "fast" if hw_accel else preset  # Use the provided preset (now defaults to veryfast)

            # Build the FFmpeg command - simplified for stability
            ffmpeg_cmd = [
                "ffmpeg", "-y",
                "-f", "concat",
                "-safe", "0",
                "-i", concat_list_path,
                "-c:v", final_codec,
                "-preset", final_preset,
                "-crf", "23",  # Higher quality for final output
                "-pix_fmt", "yuv420p",
                "-max_muxing_queue_size", "9999",  # Prevent muxing queue issues
                "-c:a", audio_codec,
                "-b:a", "192k",
                output_file
            ]

            logger.info(f"Running FFmpeg concatenation: {' '.join(ffmpeg_cmd)}")

            # Run FFmpeg directly without progress monitoring to avoid deadlocks
            process = subprocess.run(
                ffmpeg_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                text=True
            )

            if process.returncode != 0:
                logger.error(f"FFmpeg concatenation failed with return code {process.returncode}: {process.stderr}")
                raise Exception(f"FFmpeg concatenation failed: {process.stderr[:500]}...")

            logger.info(f"Successfully concatenated clips to {output_file}")
            return output_file

        except Exception as e:
            logger.error(f"Error during FFmpeg concatenation: {e}")
            logger.info("Falling back to alternative concatenation method")

            try:
                # Alternative approach: use segment concatenation
                logger.info("Trying alternative FFmpeg approach...")

                # Use ffmpeg concat with copying instead of re-encoding
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

                if process.returncode == 0:
                    logger.info(f"Alternative FFmpeg concatenation succeeded!")
                    return output_file
                else:
                    logger.error(f"Alternative FFmpeg approach failed: {process.stderr}")
            except Exception as alt_e:
                logger.error(f"Alternative approach failed: {alt_e}")

            # Last resort: MoviePy concatenation
            logger.info("Falling back to MoviePy concatenation as last resort")

            # If all direct concatenation fails, fall back to MoviePy method
            final_clips = []
            chunk_size = min(10, len(rendered_paths))  # Process in chunks

            for i in range(0, len(rendered_paths), chunk_size):
                chunk = rendered_paths[i:i+chunk_size]
                chunk_clips = []

                for clip_path in chunk:
                    try:
                        clip = VideoFileClip(clip_path)
                        chunk_clips.append(clip)
                    except Exception as clip_e:
                        logger.error(f"Error loading clip from {clip_path}: {clip_e}")

                if chunk_clips:
                    # Process this chunk
                    try:
                        chunk_output = os.path.join(temp_directory, f"chunk_{i}.mp4")
                        chunk_concat = concatenate_videoclips(chunk_clips)
                        chunk_concat.write_videofile(
                            chunk_output,
                            fps=fps,
                            preset="ultrafast",  # Speed over quality for intermediate files
                            codec=codec,
                            audio_codec=audio_codec,
                            threads=2
                        )
                        # Close all clips in this chunk
                        for clip in chunk_clips:
                            try:
                                clip.close()
                            except:
                                pass
                        chunk_concat.close()

                        # Add the chunk output to our final list
                        final_clips.append(chunk_output)
                    except Exception as chunk_e:
                        logger.error(f"Error processing chunk {i}: {chunk_e}")

                # Force garbage collection
                gc.collect()

            if not final_clips:
                raise ValueError("No clips were successfully processed for concatenation")

            # Final concatenation of chunks
            if len(final_clips) == 1:
                # Just one chunk, rename it
                shutil.copy(final_clips[0], output_file)
            else:
                # Multiple chunks, concatenate them
                final_concat_list = os.path.join(temp_directory, "final_concat.txt")
                with open(final_concat_list, "w") as f:
                    for path in final_clips:
                        f.write(f"file '{os.path.abspath(path)}'\n")

                final_cmd = [
                    "ffmpeg", "-y",
                    "-f", "concat",
                    "-safe", "0",
                    "-i", final_concat_list,
                    "-c", "copy",
                    output_file
                ]

                subprocess.run(final_cmd, check=True)

            logger.info(f"Successfully created final output at {output_file}")
            return output_file

    # Use provided temp_dir or create a new one
    if temp_dir:
        # Use provided directory
        return process_in_temp_dir(temp_dir)
    else:
        # Create and use temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            return process_in_temp_dir(temp_dir)
