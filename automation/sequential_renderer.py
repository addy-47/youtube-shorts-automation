import os
import logging
import tempfile
import threading
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from moviepy import VideoFileClip, CompositeVideoClip, concatenate_videoclips, AudioFileClip
import subprocess
import gc
import shutil
from tqdm import tqdm

logger = logging.getLogger(__name__)

def close_clip(clip):
    """Close a clip and its subcomponents to free memory."""
    try:
        if hasattr(clip, 'close'):
            clip.close()
        if hasattr(clip, 'audio') and clip.audio:
            clip.audio.close()
        if isinstance(clip, CompositeVideoClip) and hasattr(clip, 'clips'):
            for subclip in clip.clips:
                close_clip(subclip)
    except Exception as e:
        logger.debug(f"Error closing clip: {e}")
    finally:
        gc.collect()

def render_clip_segment(clip, output_path, fps=30, preset="veryfast", threads=2):
    """Render a single clip to a file with optimized settings."""
    logger.info(f"Rendering segment to {output_path}")
    temp_audio_path = output_path.replace('.mp4', '_temp_audio.wav')

    try:
        # Pre-process audio to avoid stuttering
        if clip.audio:
            clip.audio.write_audiofile(
                temp_audio_path,
                fps=48000,
                nbytes=4,
                codec='pcm_s32le',
                ffmpeg_params=['-ac', '2', '-ar', '48000', '-sample_fmt', 's32', '-bufsize', '8192k']
            )
            if os.path.exists(temp_audio_path) and os.path.getsize(temp_audio_path) > 0:
                clean_audio = AudioFileClip(temp_audio_path).with_duration(clip.duration)
                clip = clip.with_audio(clean_audio)
                clean_audio.close()

        clip.write_videofile(
            output_path,
            fps=fps,
            codec="libx264",
            audio_codec="aac",
            threads=threads,
            preset=preset,
            ffmpeg_params=['-bufsize', '24M', '-maxrate', '8M', '-b:a', '192k', '-ar', '48000', '-pix_fmt', 'yuv420p'],
            logger=None
        )
        return output_path
    except Exception as e:
        logger.error(f"Error rendering segment {output_path}: {e}")
        raise
    finally:
        if os.path.exists(temp_audio_path):
            try:
                os.remove(temp_audio_path)
            except:
                logger.debug(f"Failed to clean up {temp_audio_path}")
        close_clip(clip)

def render_clip(idx, clip, temp_dir, fps):
    """Render a clip to a file."""
    output_path = os.path.join(temp_dir, f"clip_{idx}.mp4")
    try:
        if isinstance(clip, str) and os.path.exists(clip):
            return idx, clip
        return idx, render_clip_segment(clip, output_path, fps)
    except Exception as e:
        logger.error(f"Error rendering clip {idx}: {e}")
        return idx, None
    finally:
        if not isinstance(clip, str):
            close_clip(clip)

def concatenate_clips(rendered_paths, output_file, codec="libx264", audio_codec="aac", preset="veryfast"):
    """Concatenate rendered clips using FFmpeg, with MoviePy fallback."""
    temp_dir = os.path.dirname(rendered_paths[0][1])
    concat_list_path = os.path.join(temp_dir, "concat_list.txt")

    with open(concat_list_path, "w") as f:
        for _, path in sorted(rendered_paths, key=lambda x: x[0]):
            f.write(f"file '{os.path.abspath(path)}'\n")

    ffmpeg_cmd = [
        "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat_list_path,
        "-c:v", codec, "-preset", preset, "-crf", "23", "-pix_fmt", "yuv420p",
        "-max_muxing_queue_size", "9999", "-c:a", audio_codec, "-b:a", "192k", output_file
    ]

    try:
        result = subprocess.run(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, text=True)
        logger.info(f"Concatenated clips to {output_file}")
        return output_file
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg concatenation failed: {e.stderr[:500]}")

        # Fallback to MoviePy
        logger.info("Falling back to MoviePy concatenation")
        clips = [VideoFileClip(path) for _, path in sorted(rendered_paths, key=lambda x: x[0])]
        final_clip = concatenate_videoclips(clips)
        final_clip.write_videofile(output_file, fps=30, codec=codec, audio_codec=audio_codec, preset="ultrafast")
        for clip in clips:
            close_clip(clip)
        final_clip.close()
        return output_file

def render_clips_with_threads(clips, output_file, fps=30, num_threads=None, logger=None, temp_dir=None, 
                             preset="veryfast", codec="libx264", audio_codec="aac", section_info=None):
    """
    Render clips using threads (not processes) to avoid serialization issues.
    
    This is a safer alternative to render_clips_in_parallel when experiencing 
    serialization issues with multiprocessing.
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    if num_threads is None:
        num_threads = min(len(clips), 4)  # Use a reasonable number of threads

    logger.info(f"Rendering {len(clips)} clips with {num_threads} threads")

    if temp_dir is None:
        temp_dir = tempfile.mkdtemp(prefix="thread_render_")
        logger.info(f"Created temp directory: {temp_dir}")
    else:
        os.makedirs(temp_dir, exist_ok=True)

    os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)

    # Log section info if available
    if section_info:
        logger.info("=== Section Info ===")
        for idx, info in section_info.items():
            logger.info(f"Clip {idx}: Section {info.get('section_idx', '?')} - '{info.get('section_text', 'Unknown')}'")

    # Render all clips using thread pool (no serialization issues)
    rendered_paths = []
    
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        # Submit all rendering tasks
        future_to_idx = {executor.submit(render_clip, i, c, temp_dir, fps): i for i, c in enumerate(clips)}
        
        # Process results as they complete
        for future in tqdm(concurrent.futures.as_completed(future_to_idx), total=len(clips), desc="Rendering clips"):
            idx, path = future.result()
            if path:
                rendered_paths.append((idx, path))
                logger.info(f"Clip {idx} rendered successfully")
            else:
                logger.error(f"Failed to render clip {idx}")

    if not rendered_paths:
        raise ValueError("No clips were successfully rendered")

    logger.info(f"Rendered {len(rendered_paths)} clips")
    return concatenate_clips(rendered_paths, output_file, codec, audio_codec, preset) 