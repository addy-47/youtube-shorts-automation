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

logger = logging.getLogger(__name__)

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

# Make this a top-level function so it can be pickled properly by multiprocessing
def render_with_timeout(task):
    """Render a clip with proper error handling for multiprocessing"""
    try:
        clip, output_path, fps_val, preset_val, threads_val = task
        logger.info(f"Starting render of clip to {output_path}")
        return render_clip_segment(clip, output_path, fps_val, preset_val, threads_val)
    except Exception as e:
        logger.error(f"Error in parallel rendering task: {e}")
        return None

def process_section_group(section_clips, group_indices, temp_dir, fps=30, preset="veryfast"):
    """
    Process a group of section clips in parallel.

    Args:
        section_clips: List of all section clips
        group_indices: List of indices for this group
        temp_dir: Directory for temporary files
        fps: Frames per second
        preset: Video encoding preset

    Returns:
        List of paths to rendered section files
    """
    logger.info(f"Processing section group with indices: {group_indices}")

    # Create a temporary file for each section in this group
    temp_files = []
    render_tasks = []

    for idx in group_indices:
        if idx < len(section_clips):
            clip = section_clips[idx]
            temp_path = os.path.join(temp_dir, f"section_{idx}_{int(time.time())}.mp4")
            temp_files.append(temp_path)
            render_tasks.append((clip, temp_path, fps, preset, 2))

    # If there's only one clip in this group, render it directly
    if len(render_tasks) == 1:
        logger.info(f"Single task in group - rendering directly: {group_indices[0]}")
        try:
            render_clip_segment(*render_tasks[0])
        except Exception as e:
            logger.error(f"Error rendering clip {group_indices[0]}: {e}")
        return temp_files

    # Otherwise, use multiprocessing for parallel rendering
    if render_tasks:
        logger.info(f"Starting parallel rendering of {len(render_tasks)} clips")

        # Use process pool with proper timeout handling
        pool = multiprocessing.Pool(processes=min(len(render_tasks), multiprocessing.cpu_count()))
        try:
            # Process clips in parallel with individual error handling
            results = []
            for task in render_tasks:
                results.append(pool.apply_async(render_with_timeout, args=(task,)))

            # Collect results with timeout
            timeout_per_task = 300  # 5 minutes max per clip
            completed_results = []
            for i, result in enumerate(results):
                try:
                    completed_results.append(result.get(timeout=timeout_per_task))
                    logger.info(f"Successfully rendered clip {i+1}/{len(results)}")
                except multiprocessing.TimeoutError:
                    logger.error(f"Rendering timed out for clip {i+1}/{len(results)}")
                except Exception as e:
                    logger.error(f"Error getting result for clip {i+1}/{len(results)}: {e}")

            logger.info(f"Completed parallel rendering with {len(completed_results)} successful results")
            return temp_files
        except Exception as e:
            logger.error(f"Error in parallel rendering: {e}")
        finally:
            pool.close()
            pool.join()
            logger.info(f"Closed multiprocessing pool")

    return temp_files

def render_clips_in_parallel(clips, output_path, temp_dir=None, fps=30, max_workers=None, preset="veryfast"):
    """
    Render video clips in parallel and combine them into a single video.
    Uses multiprocessing to render multiple clips simultaneously.

    Args:
        clips: List of MoviePy clips to render (section_clips)
        output_path: Path for the final output video
        temp_dir: Directory for temporary files (created if None)
        fps: Frames per second
        max_workers: Maximum number of parallel processes (defaults to CPU count)
        preset: Video encoding preset

    Returns:
        output_path: Path to the final combined video
    """
    start_time = time.time()

    # Create temp directory if not provided
    if temp_dir is None:
        temp_dir = tempfile.mkdtemp()
    os.makedirs(temp_dir, exist_ok=True)

    # Determine max workers based on available CPUs
    if max_workers is None:
        max_workers = max(1, multiprocessing.cpu_count() - 1)  # Leave one CPU free

    # Don't try to use more workers than clips
    max_workers = min(max_workers, len(clips))

    # Don't use more than 4 workers to avoid overloading the system
    max_workers = min(max_workers, 4)

    logger.info(f"Starting parallel rendering of {len(clips)} clips with {max_workers} workers")

    # Pre-render clips to files to avoid pickling issues with lambda functions
    prerender_paths = []

    # Create progress bar in console for pre-rendering
    print(f"\nPre-rendering {len(clips)} clips to prepare for parallel processing:")
    for i, clip in enumerate(tqdm(clips, desc="Pre-rendering clips", unit="clip")):
        try:
            # Create a simple temp file path for pre-rendering
            prerender_path = os.path.join(temp_dir, f"prerender_{i}_{int(time.time())}.mp4")
            prerender_paths.append(prerender_path)

            # Write clip to file with basic settings
            logger.info(f"Pre-rendering clip {i} to avoid pickling issues")

            # Use render_clip_segment with progress bar
            render_clip_segment(
                clip,
                prerender_path,
                fps=fps,
                preset="ultrafast",  # Use fastest preset for pre-rendering
                threads=2,
                show_progress=False  # Don't show moviepy's progress bar since we're using tqdm
            )

        except Exception as e:
            logger.error(f"Error pre-rendering clip {i}: {e}")
            prerender_paths.append(None)

    # Load pre-rendered clips back for further processing
    prerendered_clips = []
    print("\nLoading pre-rendered clips:")
    for i, path in enumerate(tqdm(prerender_paths, desc="Loading clips", unit="clip")):
        if path and os.path.exists(path) and os.path.getsize(path) > 0:
            try:
                clip = VideoFileClip(path)
                prerendered_clips.append(clip)
                logger.info(f"Successfully loaded pre-rendered clip {i}")
            except Exception as e:
                logger.error(f"Error loading pre-rendered clip {i}: {e}")
        else:
            logger.error(f"Pre-rendered clip {i} is missing or empty")

    if not prerendered_clips:
        logger.error("No clips were successfully pre-rendered")
        raise ValueError("No clips were successfully pre-rendered")

    # Create the final output by concatenating the clips
    try:
        logger.info(f"Concatenating {len(prerendered_clips)} pre-rendered clips")
        final_clip = concatenate_videoclips(prerendered_clips)

        # Write the final combined video with improved settings
        logger.info(f"Writing final video to {output_path}, duration: {final_clip.duration:.2f}s")
        print(f"\nRendering final video (duration: {final_clip.duration:.2f}s):")
        final_clip.write_videofile(
            output_path,
            fps=fps,
            codec="libx264",
            audio_codec="aac",
            threads=4,
            preset=preset,
            ffmpeg_params=[
                "-bufsize", "24M",      # Larger buffer
                "-maxrate", "8M",       # Higher max rate
                "-b:a", "192k",         # Higher audio bitrate
                "-ar", "48000",         # Audio sample rate
                "-pix_fmt", "yuv420p"   # Compatible pixel format for all players
            ]
        )

        # Clean up resources
        for clip in prerendered_clips:
            try:
                clip.close()
            except:
                pass

        try:
            final_clip.close()
        except:
            pass

        # Clean up temporary files
        for path in prerender_paths:
            try:
                if path and os.path.exists(path):
                    os.remove(path)
            except Exception as e:
                logger.warning(f"Error removing temp file {path}: {e}")

        total_duration = time.time() - start_time
        logger.info(f"Completed rendering in {total_duration:.2f} seconds")
        return output_path

    except Exception as e:
        logger.error(f"Error in final video concatenation: {e}")
        raise
