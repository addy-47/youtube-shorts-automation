"""
Unified renderer module with optimized parallel processing.
This module provides a clean interface for rendering video clips with maximum efficiency.
"""

import os
import logging
import tempfile
from typing import List, Dict, Any, Optional, Union
from automation.parallel_renderer import render_clips_parallel, render_clips_sequential, DEFAULT_CROSSFADE_DURATION
from helper.memory import optimize_workers_for_rendering

OPTIMIZED_RENDERER_AVAILABLE = True

logger = logging.getLogger(__name__)

def render_video(
    clips: List[Any],
    output_file: str,
    fps: int = 30,
    temp_dir: Optional[str] = None,
    preset: str = "ultrafast",
    parallel: bool = True,
    memory_per_worker_gb: float = 1.0,
    crossfade_duration: float = DEFAULT_CROSSFADE_DURATION,
    options: Optional[Dict[str, Any]] = None
) -> str:
    """
    Unified interface for rendering video clips with automatic selection
    of the most efficient rendering method.

    Args:
        clips: List of MoviePy clips to render
        output_file: Path for the output video file
        fps: Frames per second for the output video
        temp_dir: Directory for temporary files (if None, one will be created)
        preset: FFmpeg preset (slower = better quality, but longer rendering time)
        parallel: Whether to use parallel rendering
        memory_per_worker_gb: Estimated memory usage per worker in GB
        crossfade_duration: Duration of crossfade transitions between clips in seconds
        options: Additional rendering options

    Returns:
        Path to the rendered output file
    """
    if not clips:
        raise ValueError("No clips provided for rendering")

    # Create output directory if needed
    os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)

    # Default values for options
    options = options or {}

    # Allow overriding crossfade duration via options
    if 'crossfade_duration' in options:
        crossfade_duration = options['crossfade_duration']

    # Log clip metadata for debugging
    logger.info(f"Rendering {len(clips)} clips with {fps} fps")
    for i, clip in enumerate(clips):
        section_idx = getattr(clip, '_section_idx', i)
        debug_info = getattr(clip, '_debug_info', f"Clip {i}")
        duration = getattr(clip, 'duration', 0)
        logger.debug(f"Clip {i}: section_idx={section_idx}, duration={duration:.2f}s, info={debug_info}")

    # Log the rendering approach
    if OPTIMIZED_RENDERER_AVAILABLE and parallel:
        logger.info(f"Using optimized parallel FFmpeg-based renderer")

        # Get optimal resource configuration
        resource_config = optimize_workers_for_rendering(
            memory_per_task_gb=memory_per_worker_gb
        )

        # Log resource configuration
        logger.info(f"Resource config: {resource_config.get('worker_count', 1)} workers, "
                    f"{resource_config.get('ffmpeg_threads', 2)} threads per worker")

        # Use parallel rendering
        return render_clips_parallel(
            clips=clips,
            output_file=output_file,
            fps=fps,
            logger=logger,
            temp_dir=temp_dir,
            preset=preset,
            resource_config=resource_config,
            clean_temp=options.get('clean_temp', True),
            crossfade_duration=crossfade_duration
        )
    else:
        # Use sequential rendering
        logger.info(f"Using sequential rendering" +
                   (" (parallel rendering disabled)" if not parallel else ""))
        return render_clips_sequential(
            clips=clips,
            output_file=output_file,
            fps=fps,
            logger=logger,
            temp_dir=temp_dir,
            preset=preset,
            crossfade_duration=crossfade_duration
        )
