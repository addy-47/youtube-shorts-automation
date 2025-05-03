import nltk
import time
import random
import logging
import numpy as np
from moviepy.editor import VideoClip, concatenate_videoclips, ColorClip, CompositeVideoClip
from helper.blur import custom_blur, custom_edge_blur
from helper.minor_helper import measure_time

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
                    segment = clip.subclip(0, min(remaining_needed, clip.duration))
                    looped_clips.append(segment)
            else:
                looped_clips.append(clip.copy())

        clip = concatenate_videoclips(looped_clips)
    else:
        # If longer than needed, take a random segment
        if clip.duration > target_duration + 1:
            max_start = clip.duration - target_duration - 0.5
            start_time = random.uniform(0, max_start)
            clip = clip.subclip(start_time, start_time + target_duration)
        else:
            # Just take from the beginning if not much longer
            clip = clip.subclip(0, target_duration)

    # Resize to match height
    clip = clip.resize(height=  resolution[1])

    # Apply blur effect only if requested
    if blur_background and not edge_blur:
        clip = custom_blur(clip, radius=5)
    elif edge_blur:
        clip = custom_edge_blur(clip, edge_width=75, radius=10)

    # Center the video if it's not wide enough
    if clip.w < resolution[0]:
        bg = ColorClip(size=  resolution, color=(0, 0, 0)).set_duration(clip.duration)
        x_pos = (  resolution[0] - clip.w) // 2
        clip = CompositeVideoClip([bg, clip.set_position((x_pos, 0))], size=  resolution)

    # Crop width if wider than needed
    elif clip.w >   resolution[0]:
        x_centering = (clip.w -   resolution[0]) // 2
        clip = clip.crop(x1=x_centering, x2=x_centering +   resolution[0])

    # Make sure we have exact duration to prevent timing issues
    clip = clip.set_duration(target_duration)

    return clip
