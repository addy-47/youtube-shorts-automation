import os
import logging
import tempfile
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from moviepy import VideoFileClip, CompositeVideoClip, concatenate_videoclips, AudioFileClip
import subprocess
import re
import gc
import shutil
from tqdm import tqdm

logger = logging.getLogger(__name__)

try:
    import dill
    USING_DILL = True
    dill.settings['recurse'] = True
    # Enable byref for better handling of complex objects
    dill.settings['byref'] = True
    logger.info("Using dill for serialization")
except ImportError:
    USING_DILL = False
    logger.info("Dill not available, using standard pickle")

# Configure multiprocessing start method to 'spawn' for better compatibility
# This addresses Windows-specific issues with handle inheritance
def configure_multiprocessing():
    """Configure multiprocessing for maximum compatibility"""
    try:
        # Set start method if it hasn't been set yet
        if not multiprocessing.get_start_method(allow_none=True):
            multiprocessing.set_start_method('spawn', force=True)
            logger.info("Set multiprocessing start method to 'spawn'")
        
        # If dill is available, patch multiprocessing's serialization
        if USING_DILL:
            # Patch multiprocessing's serialization methods
            original_dumps = multiprocessing.reduction.dumps
            
            def patched_dumps(obj):
                try:
                    return dill.dumps(obj)
                except Exception as e:
                    logger.warning(f"Dill serialization failed: {e}, falling back to standard pickle")
                    return original_dumps(obj)
            
            multiprocessing.reduction.dumps = patched_dumps
            logger.info("Patched multiprocessing serialization with dill")
    except Exception as e:
        logger.warning(f"Could not configure multiprocessing: {e}")

# Call this at import time
configure_multiprocessing()

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

def is_complex_clip(clip):
    """Check if a clip has callable attributes or complex subclips requiring pre-rendering."""
    if isinstance(clip, str):
        return False
    # Check for callable pos and size attributes
    if hasattr(clip, 'pos') and callable(clip.pos):
        logger.debug(f"Clip has callable pos attribute")
        return True
    if hasattr(clip, 'size') and callable(clip.size):
        logger.debug(f"Clip has callable size attribute")
        return True
    # Check all attributes for callables
    for attr_name in dir(clip):
        if attr_name.startswith('__') or attr_name.startswith('_'):
            continue
        try:
            attr = getattr(clip, attr_name)
            if callable(attr) and not hasattr(attr, '__self__'):
                logger.debug(f"Clip has callable attribute: {attr_name}")
                return True
        except:
            pass
    # Recursively check subclips in composite clips
    if isinstance(clip, CompositeVideoClip) and hasattr(clip, 'clips'):
        return any(is_complex_clip(subclip) for subclip in clip.clips)
    return False

def sanitize_clip_for_serialization(clip):
    """Make a clip safely serializable by converting callable attributes to static values."""
    if isinstance(clip, str):
        return clip
        
    # Create a copy to avoid modifying the original
    try:
        # Handle problematic attributes that are commonly lambdas
        problematic_attrs = ['pos', 'size', 'mask']
        for attr_name in problematic_attrs:
            if hasattr(clip, attr_name) and callable(getattr(clip, attr_name)):
                # Try to evaluate the callable at midpoint
                try:
                    value = getattr(clip, attr_name)(clip.duration / 2)
                    # Create a new method that returns this fixed value
                    if attr_name == 'pos':
                        clip = clip.with_position(value if value else 'center')
                    elif attr_name == 'size':
                        clip = clip.resized(value)
                    elif attr_name == 'mask' and value is not None:
                        # Handle mask by pre-rendering if needed
                        # For now, we'll just remove it if it's a lambda
                        if callable(value):
                            setattr(clip, 'mask', None)
                except Exception as e:
                    logger.warning(f"Could not evaluate {attr_name}: {e}")
                    # Set to None or a default value
                    if attr_name == 'pos':
                        clip = clip.with_position('center')
                    elif attr_name == 'size':
                        pass  # Keep original size
                    elif attr_name == 'mask':
                        setattr(clip, 'mask', None)
        
        # Handle CompositeVideoClip recursively
        if isinstance(clip, CompositeVideoClip) and hasattr(clip, 'clips'):
            sanitized_clips = [sanitize_clip_for_serialization(subclip) for subclip in clip.clips]
            # Recreate with sanitized clips
            clip = CompositeVideoClip(sanitized_clips, size=clip.size).with_duration(clip.duration)
            if hasattr(clip, 'audio') and clip.audio:
                clip = clip.with_audio(clip.audio)
    
    except Exception as e:
        logger.error(f"Error sanitizing clip: {e}")
    
    return clip

def prerender_clip(idx, clip, temp_dir, fps):
    """Pre-render a complex clip to a file to avoid serialization issues."""
    temp_path = os.path.join(temp_dir, f"prerender_{idx}.mp4")
    try:
        # Always sanitize the clip first, regardless of whether USING_DILL is True
        clip = sanitize_clip_for_serialization(clip)
        
        # For genuinely complex clips or when not using dill, pre-render
        if not USING_DILL or is_complex_clip(clip):
            logger.debug(f"Pre-rendering clip {idx} with attributes: {dir(clip)}")
            
            # Handle composite clips
            if isinstance(clip, CompositeVideoClip) and hasattr(clip, 'clips'):
                fixed_clips = []
                for i, subclip in enumerate(clip.clips):
                    sub_idx = f"{idx}_{i}"
                    _, fixed_subclip = prerender_clip(sub_idx, subclip, temp_dir, fps)
                    fixed_clips.append(VideoFileClip(fixed_subclip) if isinstance(fixed_subclip, str) else fixed_subclip)
                clip = CompositeVideoClip(fixed_clips, size=clip.size).with_duration(clip.duration)
                
            render_clip_segment(clip, temp_path, fps, preset="ultrafast")
            return idx, temp_path
        
        return idx, clip
    except Exception as e:
        logger.error(f"Error pre-rendering clip {idx}: {e}")
        return idx, None
    finally:
        close_clip(clip)

def render_clip(idx, clip, temp_dir, fps):
    """Render a clip (pre-rendered or direct) to a file."""
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

def render_clips_in_parallel(clips, output_file, fps=30, num_processes=None, logger=None, temp_dir=None, preset="veryfast", codec="libx264", audio_codec="aac", section_info=None, prerender_all=False):
    """Render clips in parallel and concatenate them into a final video."""
    if logger is None:
        logger = logging.getLogger(__name__)
    if num_processes is None:
        # Use fewer processes to avoid Windows handle issues
        num_processes = max(1, min(multiprocessing.cpu_count() - 1, 4))  # Reduced from 8 to 4

    logger.info(f"Rendering {len(clips)} clips with {num_processes} processes")

    if temp_dir is None:
        temp_dir = tempfile.mkdtemp(prefix="parallel_render_")
        logger.info(f"Created temp directory: {temp_dir}")
    else:
        os.makedirs(temp_dir, exist_ok=True)

    os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)

    if section_info:
        logger.info("=== Section Info ===")
        for idx, info in section_info.items():
            logger.info(f"Clip {idx}: Section {info.get('section_idx', '?')} - '{info.get('section_text', 'Unknown')}'")

    # Sanitize all clips before preprocessing to address lambda serialization issues
    sanitized_clips = []
    for i, clip in enumerate(clips):
        sanitized_clips.append(sanitize_clip_for_serialization(clip))
    clips = sanitized_clips

    # Pre-render complex clips in parallel if not using dill or if prerender_all is True
    rendered_paths = []
    if not USING_DILL or prerender_all:
        complex_clips = [(i, c) for i, c in enumerate(clips) if prerender_all or is_complex_clip(c)]
        if complex_clips:
            logger.info(f"Pre-rendering {len(complex_clips)} clips in parallel")
            with ProcessPoolExecutor(max_workers=num_processes) as executor:
                futures = [executor.submit(prerender_clip, i, c, temp_dir, fps) for i, c in complex_clips]
                for future in as_completed(futures):
                    idx, path = future.result()
                    if path:
                        clips[idx] = path
                        rendered_paths.append((idx, path))

    # Render all clips in parallel
    logger.info("Rendering clips in parallel")
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        futures = [executor.submit(render_clip, i, c, temp_dir, fps) for i, c in enumerate(clips)]
        for future in tqdm(as_completed(futures), total=len(clips), desc="Rendering clips"):
            idx, path = future.result()
            if path:
                rendered_paths.append((idx, path))

    if not rendered_paths:
        raise ValueError("No clips were successfully rendered")

    logger.info(f"Rendered {len(rendered_paths)} clips")
    return concatenate_clips(rendered_paths, output_file, codec, audio_codec, preset)
