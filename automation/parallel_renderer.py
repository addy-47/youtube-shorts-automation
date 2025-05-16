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
# MoviePy imports
from moviepy import VideoFileClip, AudioFileClip, CompositeVideoClip, concatenate_videoclips

# Import our memory helper
try:
    from helper.memory import optimize_workers_for_rendering
except ImportError:
    # Fallback if helper module is not available
    def optimize_workers_for_rendering(memory_per_task_gb=2.0):
        """Fallback function if memory helper is not available."""
        import multiprocessing
        cpu_count = multiprocessing.cpu_count()
        return {
            'worker_count': max(1, min(4, cpu_count - 1)),
            'ffmpeg_threads': 2
        }

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
        'type': type(clip).__name__
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
    preset: str = "veryfast",
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
    output_path = os.path.join(temp_dir, f"clip_{idx}_{uuid.uuid4().hex[:8]}.mp4")
    temp_files = []  # List of temporary files to clean up

    try:
        # Extract clip information
        clip_info = extract_clip_info(clip)
        logger.debug(f"Rendering clip {idx}: duration={clip_info['duration']:.2f}s, "
                    f"size={clip_info['size']}, has_audio={clip_info['has_audio']}")

        # Handle different types of clips
        if isinstance(clip, str) and os.path.exists(clip):
            # This is already a file path, just return it
            return idx, clip

        # Use MoviePy to render the clip
        clip.write_videofile(
            output_path,
            fps=fps,
            codec="libx264",
            audio_codec="aac" if clip_info['has_audio'] else None,
            preset=preset,
            threads=threads,
            ffmpeg_params=['-bufsize', '24M', '-maxrate', '8M', '-b:a', '192k',
                          '-ar', '48000', '-pix_fmt', 'yuv420p'],
            logger=None
        )

        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            return idx, output_path
        else:
            raise ValueError(f"Output file {output_path} was not created or is empty")

    except Exception as e:
        logger.error(f"Error rendering clip {idx}: {e}")
        logger.error(traceback.format_exc())
        return idx, None
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

def concatenate_clips_with_ffmpeg(
    rendered_paths: List[Tuple[int, str]],
    output_file: str,
    preset: str = "veryfast"
) -> str:
    """
    Concatenate rendered clips using FFmpeg.

    Args:
        rendered_paths: List of (clip index, file path) tuples
        output_file: Output file path
        preset: FFmpeg preset

    Returns:
        Output file path
    """
    if not rendered_paths:
        raise ValueError("No paths to concatenate")

    # Create a temporary directory for the concat list
    temp_dir = os.path.dirname(rendered_paths[0][1])
    concat_list_path = os.path.join(temp_dir, f"concat_list_{uuid.uuid4().hex[:8]}.txt")

    # Create the concat list file
    with open(concat_list_path, "w") as f:
        for _, path in sorted(rendered_paths, key=lambda x: x[0]):
            f.write(f"file '{os.path.abspath(path)}'\n")

    # Create the FFmpeg command for concatenation
    ffmpeg_cmd = [
        "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat_list_path,
        "-c", "copy",  # Use stream copy for faster concatenation
        "-movflags", "+faststart",  # Optimize for web streaming
        output_file
    ]

    try:
        # Run the FFmpeg command
        subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        logger.info(f"Concatenated {len(rendered_paths)} clips to {output_file}")
        return output_file
    except Exception as e:
        logger.error(f"FFmpeg concatenation failed: {e}")

        # Fallback to standard MoviePy concatenation
        try:
            logger.info("Falling back to MoviePy concatenation")
            clips = [VideoFileClip(path) for _, path in sorted(rendered_paths, key=lambda x: x[0])]
            final_clip = concatenate_videoclips(clips)
            final_clip.write_videofile(
                output_file,
                fps=30,
                codec="libx264",
                audio_codec="aac",
                preset="ultrafast"
            )

            # Close clips to free memory
            for clip in clips:
                if hasattr(clip, 'close'):
                    clip.close()
            if hasattr(final_clip, 'close'):
                final_clip.close()

            return output_file
        except Exception as e:
            logger.error(f"MoviePy concatenation also failed: {e}")
            raise ValueError("Failed to concatenate clips")

# ==================== MAIN RENDERING FUNCTION ====================

def render_clips_parallel(
    clips: List[Any],
    output_file: str,
    fps: int = 30,
    logger=None,
    temp_dir: str = None,
    preset: str = "veryfast",
    resource_config: Dict[str, Any] = None,
    clean_temp: bool = True
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

    Returns:
        Output file path
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    start_time = time.time()

    # Get system resource configuration
    if resource_config is None:
        # Estimate memory requirements (very rough estimate)
        estimated_memory_gb = 2.0  # Base memory per worker
        resource_config = optimize_workers_for_rendering(memory_per_task_gb=estimated_memory_gb)

    num_workers = resource_config.get('worker_count', 4)
    ffmpeg_threads = resource_config.get('ffmpeg_threads', 2)

    logger.info(f"Rendering {len(clips)} clips with {num_workers} workers and {ffmpeg_threads} FFmpeg threads each")
    logger.info(f"Using FFmpeg: {ffmpeg_version()}")

    # Create temporary directory if needed
    if temp_dir is not None:
        temp_dir = tempfile.mkdtemp(prefix="ffmpeg_render_")
        logger.debug(f"Created temporary directory: {temp_dir}")

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
                logger.error(traceback.format_exc())

    if not rendered_paths:
        raise ValueError("No clips were successfully rendered")

    logger.info(f"Successfully rendered {len(rendered_paths)}/{len(clips)} clips")

    # Concatenate the rendered clips
    output_path = concatenate_clips_with_ffmpeg(rendered_paths, output_file, preset)

    # Clean up temporary files if requested
    if clean_temp and temp_dir and os.path.exists(temp_dir) and temp_dir.startswith(tempfile.gettempdir()):
        try:
            # Clean up individual clip files
            for _, path in rendered_paths:
                if os.path.exists(path):
                    os.remove(path)

            # Remove the temporary directory
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)
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
    preset: str = "veryfast"
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

    Returns:
        Output file path
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    logger.info("Using sequential rendering")

    # Create a temporary directory if needed
    if temp_dir is not None:
        temp_dir = tempfile.mkdtemp(prefix="sequential_render_")
        logger.debug(f"Created temporary directory: {temp_dir}")

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

    # Concatenate the rendered clips
    return concatenate_clips_with_ffmpeg(rendered_paths, output_file, preset)
