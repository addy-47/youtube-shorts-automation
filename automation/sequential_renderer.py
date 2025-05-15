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
import traceback
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
        logger.error(f"Detailed error: {traceback.format_exc()}")
        raise
    finally:
        if os.path.exists(temp_audio_path):
            try:
                os.remove(temp_audio_path)
            except:
                logger.debug(f"Failed to clean up {temp_audio_path}")
        close_clip(clip)

def sanitize_clip(clip):
    """Prepare a clip for rendering by fixing common issues."""
    if isinstance(clip, str):
        return clip
        
    try:
        # Fix any callable position attributes
        if hasattr(clip, 'pos') and callable(clip.pos):
            try:
                pos = clip.pos(clip.duration / 2)
                clip = clip.with_position(pos if pos else 'center')
            except:
                clip = clip.with_position('center')
                
        # Fix any callable size attributes
        if hasattr(clip, 'size') and callable(clip.size):
            try:
                size = clip.size(clip.duration / 2)
                clip = clip.resized(size)
            except:
                pass  # Keep original size
                
        # Handle any other callable attributes that might cause issues
        for attr_name in dir(clip):
            if attr_name.startswith('__') or attr_name.startswith('_'):
                continue
            try:
                attr = getattr(clip, attr_name)
                if callable(attr) and not hasattr(attr, '__self__'):
                    try:
                        # Try to evaluate the callable and replace it with its value
                        value = attr(clip.duration / 2)
                        setattr(clip, attr_name, value)
                    except:
                        pass  # Ignore if we can't fix it
            except:
                pass
                
    except Exception as e:
        logger.warning(f"Error sanitizing clip: {e}")
        
    return clip

def render_clip(idx, clip, temp_dir, fps):
    """Render a clip to a file."""
    output_path = os.path.join(temp_dir, f"clip_{idx}.mp4")
    try:
        # First sanitize the clip to avoid common issues
        clip = sanitize_clip(clip)
        
        if isinstance(clip, str) and os.path.exists(clip):
            return idx, clip
        return idx, render_clip_segment(clip, output_path, fps)
    except Exception as e:
        logger.error(f"Error rendering clip {idx}: {e}")
        logger.error(f"Detailed error: {traceback.format_exc()}")
        return idx, None
    finally:
        if not isinstance(clip, str):
            close_clip(clip)

def concatenate_clips(rendered_paths, output_file, codec="libx264", audio_codec="aac", preset="veryfast"):
    """Concatenate rendered clips using FFmpeg, with MoviePy fallback."""
    if not rendered_paths:
        raise ValueError("No paths to concatenate")
        
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
        try:
            clips = [VideoFileClip(path) for _, path in sorted(rendered_paths, key=lambda x: x[0])]
            final_clip = concatenate_videoclips(clips)
            final_clip.write_videofile(output_file, fps=30, codec=codec, audio_codec=audio_codec, preset="ultrafast")
            for clip in clips:
                close_clip(clip)
            final_clip.close()
            return output_file
        except Exception as e:
            logger.error(f"MoviePy concatenation also failed: {e}")
            logger.error(f"Detailed error: {traceback.format_exc()}")
            
            # Last resort: try to copy at least the first clip if it exists
            if rendered_paths:
                try:
                    _, first_path = sorted(rendered_paths, key=lambda x: x[0])[0]
                    shutil.copy2(first_path, output_file)
                    logger.info(f"Copied first clip to {output_file} as last resort")
                    return output_file
                except Exception as copy_error:
                    logger.error(f"Even copying first clip failed: {copy_error}")
                    raise ValueError("All concatenation methods failed")
            else:
                raise ValueError("No clips to concatenate")

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

    # Sanitize clips before rendering to fix common issues
    sanitized_clips = []
    for i, clip in enumerate(clips):
        sanitized_clips.append(sanitize_clip(clip))
    clips = sanitized_clips

    # Render all clips using thread pool (no serialization issues)
    rendered_paths = []
    
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        # Submit all rendering tasks
        future_to_idx = {executor.submit(render_clip, i, c, temp_dir, fps): i for i, c in enumerate(clips)}
        
        # Process results as they complete
        for future in tqdm(concurrent.futures.as_completed(future_to_idx), total=len(clips), desc="Rendering clips"):
            try:
                idx, path = future.result()
                if path:
                    rendered_paths.append((idx, path))
                    logger.info(f"Clip {idx} rendered successfully")
                else:
                    logger.error(f"Failed to render clip {idx}")
            except Exception as e:
                idx = future_to_idx[future]
                logger.error(f"Error processing future for clip {idx}: {e}")
                logger.error(f"Detailed error: {traceback.format_exc()}")

    if not rendered_paths:
        raise ValueError("No clips were successfully rendered")

    logger.info(f"Rendered {len(rendered_paths)} clips")
    return concatenate_clips(rendered_paths, output_file, codec, audio_codec, preset) 