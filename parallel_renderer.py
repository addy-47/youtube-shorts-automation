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
    """Pre-render a complex clip to avoid serialization issues"""
    # Generate a unique filename
    temp_path = os.path.join(temp_dir, f"prerender_{idx}_{int(time.time())}.mp4")

    # Make fixed copies of positions or sizes if needed
    clean_clip = clip.copy()

    # Convert any position functions to fixed values
    if hasattr(clean_clip, 'pos') and callable(clean_clip.pos):
        try:
            # Get the midpoint position
            mid_pos = clean_clip.pos(clean_clip.duration / 2)
            # Set a fixed position
            clean_clip = clean_clip.set_position(mid_pos)
        except:
            # Default to center if position function fails
            clean_clip = clean_clip.set_position('center')

    # Convert any size functions to fixed values
    if hasattr(clean_clip, 'size') and callable(clean_clip.size):
        try:
            # Get the midpoint size
            mid_size = clean_clip.size(clean_clip.duration / 2)
            # Set a fixed size
            clean_clip = clean_clip.resize(mid_size)
        except:
            # Skip if size function fails
            pass

    # Render to file with basic settings
    clean_clip.write_videofile(
        temp_path,
        fps=fps,
        preset="ultrafast",
        codec="libx264",
        audio_codec="aac",
        threads=2,
        logger="bar"
    )

    # Close the original clip to free memory
    try:
        clean_clip.close()
    except:
        pass

    # Return the path to the pre-rendered file
    return temp_path

# Central function for processing clips that avoids lambdas
def process_clip_for_parallel(task):
    """Process a clip for parallel rendering - replaces render_with_timeout"""
    try:
        clip, output_path, fps_val, preset_val, threads_val, idx, is_prerendered = task
        logger.info(f"Starting render of clip {idx} to {output_path}")

        # If clip is already pre-rendered, just load and render it
        if is_prerendered:
            try:
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

def render_clips_in_parallel(clips, output_path, temp_dir=None, fps=30, max_workers=None, preset="veryfast"):
    """
    Render video clips in parallel and combine them into a single video.
    Uses multiprocessing to render multiple clips simultaneously for faster processing.

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

    # Pre-process clips that might cause serialization issues
    processed_clips = []
    prerender_paths = []

    for idx, clip in enumerate(clips):
        # Check if clip is CompositeVideoClip or has complex attributes
        is_complex = (
            isinstance(clip, CompositeVideoClip) or
            (hasattr(clip, 'fx_list') and len(getattr(clip, 'fx_list', [])) > 0) or
            (hasattr(clip, 'mask') and clip.mask is not None)
        )

        if is_complex:
            # Pre-render complex clips to avoid serialization issues
            logger.info(f"Pre-rendering complex clip {idx}")
            path = prerender_complex_clip(clip, temp_dir, idx, fps)
            processed_clips.append(path)
            prerender_paths.append(path)
        else:
            # For simpler clips, create static versions
            static_clip, is_path = create_static_clip_version(clip)
            processed_clips.append(static_clip)
            if is_path:
                prerender_paths.append(static_clip)

    # Create a list of tasks for multiprocessing
    render_tasks = []
    temp_paths = []

    # Create task paths
    for idx, clip in enumerate(processed_clips):
        temp_path = os.path.join(temp_dir, f"section_{idx}_{int(time.time())}.mp4")
        temp_paths.append(temp_path)

        # Check if clip is a path to a pre-rendered clip
        is_prerendered = isinstance(clip, str)

        # Add task parameters including index for better logging
        render_tasks.append((clip, temp_path, fps, preset, 2, idx, is_prerendered))

    # Process clips in parallel
    rendered_paths = []

    if len(render_tasks) == 1:
        # If there's only one clip, just render it directly
        task = render_tasks[0]
        try:
            temp_path = process_clip_for_parallel(task)
            if temp_path and os.path.exists(temp_path):
                rendered_paths.append(temp_path)
        except Exception as e:
            logger.error(f"Error rendering single clip: {e}")

    elif USING_DILL:
        # Use dill for better serialization
        logger.info("Using dill for parallel rendering")
        try:
            # Create pool with explicit context for better Windows compatibility
            with multiprocessing.get_context('spawn').Pool(processes=max_workers) as pool:
                # Create a progress bar for rendering
                print(f"\nRendering {len(render_tasks)} clips in parallel:")

                # Use map_async to get results
                results = pool.map_async(process_clip_for_parallel, render_tasks)

                # Monitor progress
                with tqdm(total=len(render_tasks), desc="Rendering clips", unit="clip") as pbar:
                    # Check progress until complete
                    while not results.ready():
                        # Check how many tasks are done by looking at temp paths
                        completed = sum(1 for p in temp_paths if os.path.exists(p))
                        pbar.n = completed
                        pbar.refresh()
                        time.sleep(0.5)

                    # Make sure the progress bar completes
                    pbar.n = len(render_tasks)
                    pbar.refresh()

                # Get the results
                for result in results.get():
                    if result and os.path.exists(result):
                        rendered_paths.append(result)
                        logger.info(f"Successfully rendered a clip")
                    else:
                        logger.error(f"Failed to render a clip")

        except Exception as e:
            logger.error(f"Error with parallel processing: {e}")
            # Fall back to sequential processing
            logger.warning("Falling back to sequential processing")
            print(f"\nRendering {len(render_tasks)} clips sequentially (multiprocessing failed):")
            for i, task in enumerate(tqdm(render_tasks, desc="Rendering clips", unit="clip")):
                try:
                    temp_path = process_clip_for_parallel(task)
                    if temp_path and os.path.exists(temp_path):
                        rendered_paths.append(temp_path)
                        logger.info(f"Successfully rendered clip {i+1}")
                    else:
                        logger.error(f"Failed to render clip {i+1}")
                except Exception as e:
                    logger.error(f"Error rendering clip {i+1}: {e}")
    else:
        # If dill not available, fall back to sequential processing
        print(f"\nRendering {len(render_tasks)} clips sequentially (for better parallelization, install dill):")
        for i, task in enumerate(tqdm(render_tasks, desc="Rendering clips", unit="clip")):
            try:
                temp_path = process_clip_for_parallel(task)
                if temp_path and os.path.exists(temp_path):
                    rendered_paths.append(temp_path)
                    logger.info(f"Successfully rendered clip {i+1}")
                else:
                    logger.error(f"Failed to render clip {i+1}")
            except Exception as e:
                logger.error(f"Error rendering clip {i+1}: {e}")

    # Clean up pre-rendered clips that are no longer needed
    for path in prerender_paths:
        try:
            if os.path.exists(path):
                os.remove(path)
        except Exception as e:
            logger.warning(f"Error removing pre-rendered file {path}: {e}")

    if not rendered_paths:
        logger.error("No clips were rendered successfully")
        raise ValueError("No clips were rendered successfully")

    # Load all rendered clips
    loaded_clips = []
    print("\nLoading rendered clips:")
    for i, path in enumerate(tqdm(rendered_paths, desc="Loading clips", unit="clip")):
        try:
            clip = VideoFileClip(path)
            loaded_clips.append(clip)
            logger.info(f"Successfully loaded rendered clip {i}")
        except Exception as e:
            logger.error(f"Error loading rendered clip {i}: {e}")

    if not loaded_clips:
        logger.error("No clips were successfully loaded")
        raise ValueError("No clips were successfully loaded")

    # Create the final output by concatenating the clips
    try:
        logger.info(f"Concatenating {len(loaded_clips)} rendered clips")
        final_clip = concatenate_videoclips(loaded_clips)

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
        for clip in loaded_clips:
            try:
                clip.close()
            except:
                pass

        try:
            final_clip.close()
        except:
            pass

        # Clean up temporary files
        for path in rendered_paths:
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
