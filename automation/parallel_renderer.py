import os
import logging
import tempfile
import json
import shutil
import subprocess
import time
import uuid
import traceback
from typing import List, Dict, Tuple, Any, Optional, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from dotenv import load_dotenv
# MoviePy imports
from moviepy import VideoFileClip, AudioFileClip, CompositeVideoClip, concatenate_videoclips
from moviepy.video.fx import __all__ as vfx

# Import our own helper modules
try:
    from helper.memory import optimize_workers_for_rendering
    from helper.crossfade import concatenate_with_crossfade
except ImportError:
    # Fallback if helper module is not available
    def optimize_workers_for_rendering(memory_per_task_gb=1.0):
        """Fallback function if memory helper is not available."""
        import multiprocessing
        cpu_count = multiprocessing.cpu_count()
        return {
            'worker_count': max(1, min(4, cpu_count - 1)),
            'ffmpeg_threads': 2
        }
    
    def concatenate_with_crossfade(rendered_paths, output_file, preset="ultrafast", crossfade_duration=0.5):
        """Fallback implementation of concatenate_with_crossfade if import fails."""
        if not rendered_paths:
            raise ValueError("No paths to concatenate")
            
        # Sort clips by index
        sorted_paths = sorted(rendered_paths, key=lambda x: x[0])
        
        # If we only have one clip, just copy it to the output
        if len(sorted_paths) == 1:
            _, path = sorted_paths[0]
            shutil.copy(path, output_file)
            return output_file
            
        # Use simple FFmpeg concatenation as fallback
        temp_dir = os.path.dirname(sorted_paths[0][1])
        concat_list_path = os.path.join(temp_dir, f"concat_list_{uuid.uuid4().hex[:8]}.txt")
        
        # Create the concat list file
        with open(concat_list_path, "w") as f:
            for _, path in sorted_paths:
                f.write(f"file '{os.path.abspath(path)}'\n")
        
        # Run FFmpeg
        ffmpeg_cmd = [
            "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat_list_path,
            "-c", "copy", output_file
        ]
        subprocess.run(ffmpeg_cmd, check=True)
        
        # Clean up
        if os.path.exists(concat_list_path):
            os.remove(concat_list_path)
            
        return output_file

# Load environment variables
load_dotenv()
TEMP_DIR = os.getenv("TEMP_DIR", tempfile.gettempdir())

# Default crossfade duration
DEFAULT_CROSSFADE_DURATION = 1.0  # Increased to 1 second

logger = logging.getLogger(__name__)

# ==================== DIRECT FFMPEG UTILITIES ====================

def ffmpeg_version() -> str:
    """Get the FFmpeg version to ensure compatibility."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        version_line = result.stdout.split('\n')[0]
        return version_line
    except Exception as e:
        logger.warning(f"Could not determine FFmpeg version: {e}")
        return "Unknown"

def clean_temp_files(file_paths: List[str]) -> None:
    """Clean up temporary files."""
    for path in file_paths:
        try:
            if os.path.isdir(path):
                shutil.rmtree(path, ignore_errors=True)
            elif os.path.exists(path):
                os.remove(path)
        except Exception as e:
            logger.debug(f"Could not remove temporary file {path}: {e}")

def extract_clip_info(clip) -> Dict[str, Any]:
    """Extract essential information from a MoviePy clip."""
    clip_info = {
        'duration': getattr(clip, 'duration', 0),
        'size': getattr(clip, 'size', (0, 0)),
        'fps': getattr(clip, 'fps', 30),
        'has_audio': hasattr(clip, 'audio') and clip.audio is not None,
        'is_file': False,
        'filepath': None,
        'type': type(clip).__name__,
        'section_idx': getattr(clip, '_section_idx', None),
        'debug_info': getattr(clip, '_debug_info', None)
    }

    # Check if this is a file-based clip
    if hasattr(clip, 'filename') and clip.filename and os.path.exists(clip.filename):
        clip_info['is_file'] = True
        clip_info['filepath'] = clip.filename

    return clip_info

def export_audio_from_clip(clip, output_path: str) -> Optional[str]:
    """Export audio from a MoviePy clip to a WAV file."""
    if not hasattr(clip, 'audio') or clip.audio is None:
        return None

    try:
        audio_path = output_path.replace('.mp4', '.wav')
        clip.audio.write_audiofile(
            audio_path,
            fps=48000,
            nbytes=4,
            codec='pcm_s32le',
            ffmpeg_params=['-ac', '2', '-ar', '48000', '-sample_fmt', 's32', '-bufsize', '8192k']
        )
        return audio_path
    except Exception as e:
        logger.error(f"Failed to export audio: {e}")
        return None

def is_direct_renderable(clip) -> bool:
    """Check if a clip can be directly rendered by FFmpeg without preprocessing."""
    # Simple file-based clips can be directly rendered
    if hasattr(clip, 'filename') and clip.filename and os.path.exists(clip.filename):
        return True

    # Check for common callable attributes that would prevent direct rendering
    for attr_name in ['pos', 'size', 'mask']:
        if hasattr(clip, attr_name) and callable(getattr(clip, attr_name)):
            return False

    # Check if composite clip - must check subclips
    if isinstance(clip, CompositeVideoClip) and hasattr(clip, 'clips'):
        return all(is_direct_renderable(subclip) for subclip in clip.clips)

    return False

# ==================== CLIP RENDERING ====================

def render_clip_with_ffmpeg(
    idx: int,
    clip,
    temp_dir: str,
    fps: int = 30,
    preset: str = "ultrafast",
    threads: int = 2
) -> Tuple[int, Optional[str]]:
    """
    Render a clip using FFmpeg or MoviePy as needed.

    Args:
        idx: Clip index
        clip: MoviePy clip
        temp_dir: Temporary directory
        fps: Frames per second
        preset: FFmpeg preset
        threads: Number of threads

    Returns:
        Tuple of (clip index, output file path or None if failed)
    """
    # Get section index from clip if available
    section_idx = getattr(clip, '_section_idx', idx)
    debug_info = getattr(clip, '_debug_info', f"Clip {idx}")

    output_path = os.path.join(temp_dir, f"clip_{section_idx:03d}_{uuid.uuid4().hex[:8]}.mp4")
    temp_files = []  # List of temporary files to clean up

    try:
        # Extract clip information
        clip_info = extract_clip_info(clip)
        logger.info(f"Rendering {debug_info}: duration={clip_info['duration']:.2f}s, "
                   f"size={clip_info['size']}, section_idx={section_idx}")

        # Handle different types of clips
        if isinstance(clip, str) and os.path.exists(clip):
            # This is already a file path, just return it
            return section_idx, clip

        # Use MoviePy to render the clip with enhanced buffer settings to prevent frame read errors
        max_retries = 2
        retry_count = 0
        
        while retry_count <= max_retries:
            try:
                clip.write_videofile(
                    output_path,
                    fps=fps,
                    codec="libx264",
                    audio_codec="aac" if clip_info['has_audio'] else None,
                    preset=preset,
                    threads=threads,
                    ffmpeg_params=[
                        '-bufsize', '50M',  # Increased buffer size
                        '-maxrate', '10M',  # Increased max rate
                        '-b:a', '192k',
                        '-ar', '48000',
                        '-pix_fmt', 'yuv420p',
                        '-max_muxing_queue_size', '9999'  # Handle muxing queue size issues
                    ],
                    logger=None
                )
                break  # If successful, exit the retry loop
            except Exception as e:
                retry_count += 1
                if retry_count > max_retries:
                    raise  # Re-raise the exception if we've exhausted retries
                logger.warning(f"Retry {retry_count}/{max_retries} for rendering {debug_info}: {e}")
                time.sleep(1)  # Brief pause before retrying

        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            logger.info(f"Successfully rendered {debug_info} to {os.path.basename(output_path)}")
            return section_idx, output_path
        else:
            raise ValueError(f"Output file {output_path} was not created or is empty")

    except Exception as e:
        logger.error(f"Error rendering {debug_info}: {e}")
        logger.debug(traceback.format_exc())
        return section_idx, None
    finally:
        # Clean up any temporary files
        clean_temp_files(temp_files)

        # Close MoviePy clip to free memory
        try:
            if hasattr(clip, 'close'):
                clip.close()
            if hasattr(clip, 'audio') and clip.audio:
                clip.audio.close()
        except:
            pass

# ==================== MAIN RENDERING FUNCTION ====================

def render_clips_parallel(
    clips: List[Any],
    output_file: str,
    fps: int = 30,
    logger=None,
    temp_dir: str = None,
    preset: str = "ultrafast",
    resource_config: Dict[str, Any] = None,
    clean_temp: bool = True,
    crossfade_duration: float = 0.5
) -> str:
    """
    Render clips in parallel using threads and FFmpeg.

    Args:
        clips: List of MoviePy clips
        output_file: Output file path
        fps: Frames per second
        logger: Logger (if None, a new one will be created)
        temp_dir: Temporary directory (if None, a new one will be created)
        preset: FFmpeg preset
        resource_config: System resource configuration (if None, will be determined automatically)
        clean_temp: Whether to clean up temporary files
        crossfade_duration: Duration of crossfade transitions between clips in seconds

    Returns:
        Output file path
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    start_time = time.time()

    # Use memory settings from memory.py (now defaults to 1.0GB per worker)
    if resource_config is None:
        resource_config = optimize_workers_for_rendering(memory_per_task_gb=1.0)

    num_workers = resource_config.get('worker_count', 2)
    ffmpeg_threads = resource_config.get('ffmpeg_threads', 2)

    logger.info(f"Rendering {len(clips)} clips with {num_workers} workers and {ffmpeg_threads} FFmpeg threads each")
    logger.info(f"Using FFmpeg: {ffmpeg_version()}")

    # Use the TEMP_DIR environment variable
    if temp_dir is None:
        temp_dir = os.path.join(TEMP_DIR, f"ffmpeg_render_{int(time.time())}")
        os.makedirs(temp_dir, exist_ok=True)
        logger.debug(f"Created temporary directory: {temp_dir}")

    # Log clip information for debugging order issues
    for i, clip in enumerate(clips):
        section_idx = getattr(clip, '_section_idx', i)
        debug_info = getattr(clip, '_debug_info', f"Clip {i}")
        logger.info(f"Preparing clip {i}: section_idx={section_idx}, {debug_info}")

    # Create output directory if needed
    os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)

    # Render all clips in parallel using thread pool
    rendered_paths = []

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit all rendering tasks
        futures = [
            executor.submit(
                render_clip_with_ffmpeg,
                i, clip, temp_dir, fps, preset, ffmpeg_threads
            )
            for i, clip in enumerate(clips)
        ]

        # Process results as they complete
        for future in tqdm(as_completed(futures), total=len(clips), desc="Rendering clips"):
            try:
                idx, path = future.result()
                if path:
                    rendered_paths.append((idx, path))
                    logger.debug(f"Clip {idx} rendered successfully: {path}")
                else:
                    logger.error(f"Failed to render clip {idx}")
            except Exception as e:
                logger.error(f"Error processing render result: {e}")
                logger.debug(traceback.format_exc())

    if not rendered_paths:
        raise ValueError("No clips were successfully rendered")

    logger.info(f"Successfully rendered {len(rendered_paths)}/{len(clips)} clips")

    # Concatenate the rendered clips with crossfades using our helper module
    output_path = concatenate_with_crossfade(
        rendered_paths,
        output_file,
        crossfade_duration=crossfade_duration,
        preset=preset
    )

    # Clean up temporary files if requested
    if clean_temp and temp_dir:
        try:
            # Clean up individual clip files
            for _, path in rendered_paths:
                if os.path.exists(path):
                    os.remove(path)

            # Remove the temporary directory
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)
                logger.debug(f"Cleaned up temporary directory: {temp_dir}")
        except Exception as e:
            logger.warning(f"Failed to clean up temporary files: {e}")

    total_time = time.time() - start_time
    logger.info(f"Total rendering time: {total_time:.2f} seconds")

    return output_path

# Fallback function that renders sequentially for compatibility
def render_clips_sequential(
    clips: List[Any],
    output_file: str,
    fps: int = 30,
    logger=None,
    temp_dir: str = None,
    preset: str = "ultrafast",
    crossfade_duration: float = 0.5
) -> str:
    """
    Simple fallback function that renders clips sequentially.

    Args:
        clips: List of MoviePy clips
        output_file: Output file path
        fps: Frames per second
        logger: Logger
        temp_dir: Temporary directory
        preset: FFmpeg preset
        crossfade_duration: Duration of crossfade transitions between clips in seconds

    Returns:
        Output file path
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    logger.info("Using sequential rendering")

    # Use the TEMP_DIR environment variable
    if temp_dir is None:
        temp_dir = os.path.join(TEMP_DIR, f"sequential_render_{int(time.time())}")
        os.makedirs(temp_dir, exist_ok=True)
        logger.debug(f"Created temporary directory: {temp_dir}")

    # Log clip information for debugging
    for i, clip in enumerate(clips):
        section_idx = getattr(clip, '_section_idx', i)
        debug_info = getattr(clip, '_debug_info', f"Clip {i}")
        logger.info(f"Preparing clip {i}: section_idx={section_idx}, {debug_info}")

    # Render clips sequentially
    rendered_paths = []

    for i, clip in enumerate(tqdm(clips, desc="Rendering clips sequentially")):
        try:
            idx, path = render_clip_with_ffmpeg(i, clip, temp_dir, fps, preset, 4)
            if path:
                rendered_paths.append((idx, path))
                logger.debug(f"Clip {idx} rendered successfully: {path}")
            else:
                logger.error(f"Failed to render clip {idx}")
        except Exception as e:
            logger.error(f"Error rendering clip {i}: {e}")

    if not rendered_paths:
        raise ValueError("No clips were successfully rendered")

    # Concatenate the rendered clips with the helper function
    return concatenate_with_crossfade(
        rendered_paths,
        output_file,
        preset=preset,
        crossfade_duration=crossfade_duration
    )
