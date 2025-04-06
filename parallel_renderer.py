"""
Parallel video rendering module for YouTube Shorts automation.
Implements multiprocessing capabilities for faster video rendering while maintaining original quality and transitions.
"""

import os
import time
import logging
import tempfile
import multiprocessing
from moviepy.editor import VideoFileClip, concatenate_videoclips, CompositeVideoClip, AudioFileClip
from datetime import datetime

logger = logging.getLogger(__name__)

def render_clip_segment(clip, output_path, fps=30, preset="veryfast", threads=2):
    """
    Render a single clip segment to a file.

    Args:
        clip: The MoviePy clip to render
        output_path: Where to save the rendered clip
        fps: Frames per second
        preset: Video encoding preset
        threads: Number of threads for encoding

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
        # Always show progress bar by setting logger=None (logger prevents progress bar)
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
            logger=None  # Set to None to show progress bar
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

        # Use a separate function to handle individual rendering with timeout
        def render_with_timeout(task):
            try:
                clip, output_path, fps_val, preset_val, threads_val = task
                logger.info(f"Starting render of clip to {output_path}")
                return render_clip_segment(clip, output_path, fps_val, preset_val, threads_val)
            except Exception as e:
                logger.error(f"Error in parallel rendering task: {e}")
                return None

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

    # Divide clips into groups for parallel processing
    # Process fewer clips at once to avoid memory issues
    clip_groups = []
    max_clips_per_group = 2  # Process at most 2 clips at a time
    for i in range(0, len(clips), max_clips_per_group):
        group = list(range(i, min(i + max_clips_per_group, len(clips))))
        clip_groups.append(group)

    logger.info(f"Created {len(clip_groups)} groups of clips for rendering")

    # Process each group in sequence, but render clips in each group in parallel
    temp_paths = []
    for group_idx, group in enumerate(clip_groups):
        logger.info(f"Processing group {group_idx+1}/{len(clip_groups)}: {group}")

        try:
            group_paths = process_section_group(clips, group, temp_dir, fps, preset)
            temp_paths.extend([p for p in group_paths if os.path.exists(p) and os.path.getsize(p) > 0])
            logger.info(f"Completed group {group_idx+1}, got {len(group_paths)} valid clips")
        except Exception as e:
            logger.error(f"Error processing group {group_idx+1}: {e}")
            # Continue with next group

    logger.info(f"All groups processed, got {len(temp_paths)} valid clips")

    # Load rendered clips in the correct order
    if len(temp_paths) == 0:
        logger.error("No clips were rendered successfully")
        raise ValueError("No clips were rendered successfully")

    logger.info(f"Loading {len(temp_paths)} rendered clips in sequence order")

    # Sort by section number from the filename
    # Get only valid paths that exist and have content
    valid_temp_paths = [p for p in temp_paths if os.path.exists(p) and os.path.getsize(p) > 0]

    if len(valid_temp_paths) == 0:
        logger.error("No valid rendered clip files found")
        raise ValueError("No valid rendered clip files found")

    # Sort by section number
    ordered_temp_paths = sorted(valid_temp_paths, key=lambda p: int(os.path.basename(p).split('_')[1]))
    rendered_clips = []

    for path in ordered_temp_paths:
        try:
            clip = VideoFileClip(path)
            rendered_clips.append(clip)
            logger.info(f"Loaded clip: {path}, duration: {clip.duration:.2f}s")
        except Exception as e:
            logger.error(f"Error loading rendered clip {path}: {e}")

    # Concatenate in the original order to maintain transitions
    logger.info(f"Concatenating {len(rendered_clips)} rendered clips")
    try:
        if not rendered_clips:
            raise ValueError("No clips were successfully rendered. Cannot produce final video.")

        # Create final composition
        final_clip = concatenate_videoclips(rendered_clips)

        # Write the final combined video
        logger.info(f"Writing final video to {output_path}, duration: {final_clip.duration:.2f}s")
        final_clip.write_videofile(
            output_path,
            fps=fps,
            codec="libx264",
            audio_codec="aac",
            threads=4,
            preset=preset
        )

        # Close clips to release resources
        for clip in rendered_clips:
            try:
                clip.close()
            except:
                pass

        final_clip.close()

        # Clean up temporary files
        for path in temp_paths:
            try:
                if os.path.exists(path):
                    os.remove(path)
            except Exception as e:
                logger.warning(f"Error removing temp file {path}: {e}")

        total_duration = time.time() - start_time
        logger.info(f"Completed rendering in {total_duration:.2f} seconds")
        return output_path

    except Exception as e:
        logger.error(f"Error in final video concatenation: {e}")
        raise
