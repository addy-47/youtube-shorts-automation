import nltk
import time
import random
import logging
import numpy as np
import concurrent.futures
import os
from moviepy  import VideoClip, concatenate_videoclips, ColorClip, CompositeVideoClip
from helper.blur import custom_blur, custom_edge_blur
from helper.minor_helper import measure_time
from functools import partial

# Try to import dill for better serialization
try:
    import dill
    HAS_DILL = True
except ImportError:
    HAS_DILL = False

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@measure_time
def _process_background_clip(clip, target_duration, blur_background=False, edge_blur=False):
    """
    Process a background clip to match the required duration

    Args:
        clip (VideoClip): The video clip to process
        target_duration (float): The desired duration in seconds
        blur_background (bool): Whether to apply blur effect
        edge_blur (bool): Whether to apply edge blur effect

    Returns:
        VideoClip: Processed clip with matching duration
    """
    resolution = (1080, 1920) # Assuming a standard resolution for YouTube Shorts
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
                    segment = clip.subclipped(0, min(remaining_needed, clip.duration))
                    looped_clips.append(segment)
            else:
                looped_clips.append(clip.copy())

        clip = concatenate_videoclips(looped_clips)
    else:
        # If longer than needed, take a random segment
        if clip.duration > target_duration + 1:
            max_start = clip.duration - target_duration - 0.5
            start_time = random.uniform(0, max_start)
            clip = clip.subclipped(start_time, start_time + target_duration)
        else:
            # Just take from the beginning if not much longer
            clip = clip.subclipped(0, target_duration)

    # resized to match height
    clip = clip.resized(height=resolution[1])

    # Apply blur effect only if requested
    if blur_background and not edge_blur:
        clip = custom_blur(clip, radius=5)
    elif edge_blur:
        clip = custom_edge_blur(clip, edge_width=75, radius=10)

    # Center the video if it's not wide enough
    if clip.w < resolution[0]:
        bg = ColorClip(size=resolution, color=(0, 0, 0)).with_duration(clip.duration)
        x_pos = (resolution[0] - clip.w) // 2
        clip = CompositeVideoClip([bg, clip.with_position((x_pos, 0))], size=resolution)

    # Crop width if wider than needed
    elif clip.w > resolution[0]:
        x_centering = (clip.w - resolution[0]) // 2
        clip = clip.cropped(x1=x_centering, x2=x_centering + resolution[0])

    # Make sure we have exact duration to prevent timing issues
    clip = clip.with_duration(target_duration)

    return clip

# Move the process_clip function outside of process_background_clips_parallel
def _process_background_clip_wrapper(clip_info, blur_background=False, edge_blur=False):
    """
    Process a single background clip (wrapper for _process_background_clip)

    Args:
        clip_info (dict): Dictionary with 'clip' and 'target_duration'
        blur_background (bool): Whether to apply blur effect
        edge_blur (bool): Whether to apply edge blur effect

    Returns:
        VideoClip: Processed clip
    """
    try:
        return _process_background_clip(
            clip_info['clip'],
            clip_info['target_duration'],
            blur_background,
            edge_blur
        )
    except Exception as e:
        logger.error(f"Error processing background clip: {e}")
        return None

@measure_time
def process_background_clips_parallel(clip_info_list, blur_background=False, edge_blur=False, max_workers=None):
    """
    Process multiple background clips in parallel

    Args:
        clip_info_list (list): List of dictionaries with 'clip' and 'target_duration' keys
        blur_background (bool): Whether to apply blur effect
        edge_blur (bool): Whether to apply edge blur effect
        max_workers (int): Maximum number of concurrent workers

    Returns:
        list: List of processed background clips
    """
    start_time = time.time()
    logger.info(f"Processing {len(clip_info_list)} background clips in parallel")

    # Determine number of workers based on CPU cores
    if not max_workers:
        max_workers = min(len(clip_info_list), os.cpu_count())

    # Choose executor based on dill availability
    if HAS_DILL:
        logger.info("Using ThreadPoolExecutor with dill for background processing")
        executor_class = concurrent.futures.ThreadPoolExecutor
    else:
        logger.info("Using ProcessPoolExecutor for background processing")
        executor_class = concurrent.futures.ProcessPoolExecutor

    # Use the selected executor
    processed_clips = []
    with executor_class(max_workers=max_workers) as executor:
        # Submit all tasks to the executor
        futures = []
        for clip_info in clip_info_list:
            futures.append(
                executor.submit(
                    _process_background_clip_wrapper,
                    clip_info,
                    blur_background=blur_background,
                    edge_blur=edge_blur
                )
            )

        # Process completed futures
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            try:
                processed_clip = future.result()
                if processed_clip:
                    processed_clips.append(processed_clip)
                    logger.info(f"Processed background clip {i+1}/{len(clip_info_list)}")
            except Exception as e:
                logger.error(f"Failed to process background clip: {e}")

    total_time = time.time() - start_time
    logger.info(f"Processed {len(processed_clips)} background clips in {total_time:.2f} seconds")

    return processed_clips
