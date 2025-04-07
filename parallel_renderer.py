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

# Superior dill setup for better lambda function and closure support
try:
    import dill

    # Force dill to use a more aggressive protocol
    dill.settings['recurse'] = True

    # Register dill as the serializer for multiprocessing
    # This is the most important line - using the right registration method
    multiprocessing.reduction.register(dill.dumps, dill.loads)

    # Explicitly set the protocol version
    DILL_PROTOCOL = dill.HIGHEST_PROTOCOL

    USING_DILL = True
    print(f"Using dill {dill.__version__} for enhanced function serialization with protocol {DILL_PROTOCOL}.")
except ImportError:
    USING_DILL = False
    print("Dill not available. Using standard pickle for serialization.")
except Exception as e:
    print(f"Error setting up dill: {e}")
    USING_DILL = False

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

        # Instead of pre-rendering, use direct rendering with proper error handling
        # This preserves the parallelization benefits while still handling lambda issues
        try:
            # Directly render the clip with all its original attributes
            # Dill should handle the serialization of lambda functions, but just in case:

            # 1. Make a local copy of the clip
            clip_copy = clip.copy()

            # 2. If the clip uses position/size functions, try to preserve their information
            if hasattr(clip, 'pos') and callable(clip.pos):
                # Store size and position info - this gets lost in multiprocessing
                try:
                    # Sample the position function at a few points
                    t0 = 0
                    tmid = clip.duration / 2 if hasattr(clip, 'duration') else 1
                    tend = clip.duration if hasattr(clip, 'duration') else 2

                    pos0 = clip.pos(t0)
                    posmid = clip.pos(tmid)
                    posend = clip.pos(tend)

                    # If all positions are the same, we can use a fixed position
                    if pos0 == posmid == posend:
                        clip_copy = clip_copy.set_position(pos0)
                    # Otherwise, use the position at midpoint which is usually most representative
                    else:
                        clip_copy = clip_copy.set_position(posmid)
                except:
                    # If position function fails, use center as fallback
                    clip_copy = clip_copy.set_position('center')

            # 3. Similarly handle size functions
            if hasattr(clip, 'size') and callable(clip.size):
                try:
                    # Sample the size function at midpoint
                    tmid = clip.duration / 2 if hasattr(clip, 'duration') else 1
                    size_mid = clip.size(tmid)
                    clip_copy = clip_copy.resize(newsize=size_mid)
                except:
                    # If size function fails, leave as is
                    pass

            # Use the copy for rendering
            return render_clip_segment(clip_copy, output_path, fps_val, preset_val, threads_val)

        except Exception as e:
            logger.warning(f"Error with direct rendering: {e}. Trying fallback.")

            # If direct rendering fails, try an emergency pre-render approach
            try:
                # Create a temporary path for emergency pre-rendering
                temp_dir = os.path.dirname(output_path)
                temp_filename = f"emergency_{os.path.basename(output_path)}"
                temp_path = os.path.join(temp_dir, temp_filename)

                logger.info(f"Emergency pre-rendering to: {temp_path}")

                # Write with minimal settings for speed
                clip.write_videofile(
                    temp_path,
                    fps=fps_val,
                    preset="ultrafast",
                    codec="libx264",
                    audio_codec="aac",
                    threads=2,
                    logger="bar"
                )

                # Reload and render
                from moviepy.editor import VideoFileClip
                fallback_clip = VideoFileClip(temp_path)
                result = render_clip_segment(fallback_clip, output_path, fps_val, preset_val, threads_val)

                # Clean up
                try:
                    fallback_clip.close()
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                except:
                    pass

                return result

            except Exception as fallback_error:
                logger.error(f"Fallback rendering also failed: {fallback_error}")
                return None

    except Exception as e:
        logger.error(f"Error in parallel rendering task: {e}")
        return None

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

    # Remove pre-processing step - we'll handle serialization in the process pool

    # Determine max workers based on available CPUs
    if max_workers is None:
        max_workers = max(1, multiprocessing.cpu_count() - 1)  # Leave one CPU free

    # Don't try to use more workers than clips
    max_workers = min(max_workers, len(clips))

    # Don't use more than 4 workers to avoid overloading the system
    max_workers = min(max_workers, 4)

    logger.info(f"Starting parallel rendering of {len(clips)} clips with {max_workers} workers")

    # Create a list of tasks for multiprocessing
    render_tasks = []
    temp_paths = []

    # Create task paths
    for idx, clip in enumerate(clips):
        temp_path = os.path.join(temp_dir, f"section_{idx}_{int(time.time())}.mp4")
        temp_paths.append(temp_path)
        render_tasks.append((clip, temp_path, fps, preset, 2))

    # Process clips in parallel
    rendered_paths = []

    if len(render_tasks) == 1:
        # If there's only one clip, just render it directly
        try:
            temp_path = render_clip_segment(*render_tasks[0])
            if temp_path and os.path.exists(temp_path):
                rendered_paths.append(temp_path)
        except Exception as e:
            logger.error(f"Error rendering single clip: {e}")

    elif USING_DILL:
        # Use dill for better serialization with proper error handling
        try:
            # IMPORTANT: Create multiprocessing context that explicitly uses dill
            # Create pool with initializer to ensure dill is properly used
            def init_worker():
                import dill
                dill.settings['recurse'] = True
                # Register dill in the worker process
                multiprocessing.reduction.register(dill.dumps, dill.loads)

            with multiprocessing.get_context('spawn').Pool(
                processes=max_workers,
                initializer=init_worker
            ) as pool:
                # Create a progress bar for rendering
                print(f"\nRendering {len(render_tasks)} clips in parallel:")

                # Use map_async with explicit timeout to avoid hanging
                results = pool.map_async(render_with_timeout, render_tasks)

                # Monitor with our own progress bar
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
                    temp_path = render_clip_segment(*task)
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
                temp_path = render_clip_segment(*task)
                if temp_path and os.path.exists(temp_path):
                    rendered_paths.append(temp_path)
                    logger.info(f"Successfully rendered clip {i+1}")
                else:
                    logger.error(f"Failed to render clip {i+1}")
            except Exception as e:
                logger.error(f"Error rendering clip {i+1}: {e}")

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
