"""
Video encoding optimization module for YouTube Shorts.

This module provides consistent encoding settings and optimization
for video processing throughout the codebase.
"""

import os
import logging
import subprocess
from enum import Enum
import shutil

logger = logging.getLogger(__name__)

class EncodingPreset(Enum):
    """Encoding presets for different quality/speed tradeoffs"""
    ULTRAFAST = "ultrafast"  # Fastest, lowest quality
    SUPERFAST = "superfast"
    VERYFAST = "veryfast"    # Good balance for final output
    FASTER = "faster"
    FAST = "fast"
    MEDIUM = "medium"        # Default in FFmpeg
    SLOW = "slow"
    SLOWER = "slower"
    VERYSLOW = "veryslow"    # Slowest, highest quality

class VideoEncoder:
    """
    Static utility class for video encoding operations.
    
    Provides consistent encoding settings and hardware acceleration
    detection across the codebase.
    """
    
    # Class-level hardware acceleration detection
    _hw_accel_codec = None
    _hw_accel_checked = False
    
    @classmethod
    def get_hw_accel_codec(cls):
        """
        Detect available hardware acceleration.
        
        Returns:
            String codec name or None if no hardware acceleration is available
        """
        if cls._hw_accel_checked:
            return cls._hw_accel_codec
            
        # Check for NVIDIA GPU
        try:
            nvidia_check = subprocess.run(
                ["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False
            )
            if nvidia_check.returncode == 0:
                cls._hw_accel_codec = "h264_nvenc"
                logger.info("Hardware acceleration detected: NVIDIA GPU")
        except Exception:
            # NVIDIA tools not installed
            pass
            
        # Check for Intel QuickSync (Linux)
        if cls._hw_accel_codec is None:
            try:
                qsv_check = subprocess.run(
                    ["vainfo"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False
                )
                if qsv_check.returncode == 0 and b"VA-API" in qsv_check.stdout:
                    cls._hw_accel_codec = "h264_qsv"
                    logger.info("Hardware acceleration detected: Intel QuickSync")
            except Exception:
                # vainfo not installed
                pass
                
        # Check for AMD (Linux)
        if cls._hw_accel_codec is None:
            try:
                amd_check = subprocess.run(
                    ["vainfo"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False
                )
                if amd_check.returncode == 0 and b"AMD" in amd_check.stdout:
                    cls._hw_accel_codec = "h264_amf"
                    logger.info("Hardware acceleration detected: AMD GPU")
            except Exception:
                # vainfo not installed
                pass
        
        cls._hw_accel_checked = True
        
        if cls._hw_accel_codec is None:
            logger.info("No hardware acceleration detected, using software encoding")
            
        return cls._hw_accel_codec
    
    @classmethod
    def get_intermediate_codec(cls):
        """
        Get the best codec for intermediate files.
        
        Returns:
            Codec name to use for intermediate files
        """
        # For intermediate files, prefer hardware acceleration
        hw_codec = cls.get_hw_accel_codec()
        return hw_codec if hw_codec else "libx264"
    
    @classmethod
    def get_final_codec(cls):
        """
        Get the best codec for final output files.
        
        Returns:
            Codec name to use for final output files
        """
        # For final output, use hardware acceleration if available
        hw_codec = cls.get_hw_accel_codec()
        return hw_codec if hw_codec else "libx264"
    
    @classmethod
    def get_intermediate_preset(cls):
        """
        Get the best preset for intermediate files.
        
        Returns:
            Preset name to use for intermediate files
        """
        # Always use ultrafast for intermediate files to maximize speed
        return EncodingPreset.ULTRAFAST.value
    
    @classmethod
    def get_final_preset(cls):
        """
        Get the best preset for final output files.
        
        Returns:
            Preset name to use for final output files
        """
        # Use veryfast for final output as a good balance
        return EncodingPreset.VERYFAST.value
    
    @classmethod
    def get_intermediate_params(cls):
        """
        Get complete FFmpeg parameters for intermediate files.
        
        Returns:
            Dictionary with encoding parameters
        """
        codec = cls.get_intermediate_codec()
        preset = cls.get_intermediate_preset()
        
        params = {
            "codec": codec,
            "audio_codec": "aac",
            "preset": preset,
            "threads": 2,  # Lower thread count for intermediate files
            "audio_bufsize": 4096,
            "ffmpeg_params": [
                "-crf", "28",        # Lower quality for intermediate files
                "-maxrate", "4M",    # Lower bitrate for intermediate files
                "-bufsize", "8M",    # Smaller buffer
                "-pix_fmt", "yuv420p",  # Compatible format
            ]
        }
        
        return params
    
    @classmethod
    def get_final_params(cls):
        """
        Get complete FFmpeg parameters for final output files.
        
        Returns:
            Dictionary with encoding parameters
        """
        codec = cls.get_final_codec()
        preset = cls.get_final_preset()
        
        params = {
            "codec": codec,
            "audio_codec": "aac",
            "preset": preset,
            "threads": 4,  # Higher thread count for final output
            "audio_bufsize": 8192,
            "ffmpeg_params": [
                "-crf", "23",        # Higher quality for final output
                "-maxrate", "8M",    # Higher bitrate for final output
                "-bufsize", "16M",   # Larger buffer
                "-pix_fmt", "yuv420p",  # Compatible format
                "-b:a", "192k",      # Higher audio bitrate
                "-ar", "48000",      # Audio sample rate
                "-max_muxing_queue_size", "9999"  # Prevent muxing queue issues
            ]
        }
        
        return params
    
    @classmethod
    def get_concat_command(cls, concat_list_path, output_file, is_final=True):
        """
        Get FFmpeg command for concatenating video files.
        
        Args:
            concat_list_path: Path to file containing list of files to concatenate
            output_file: Path to output file
            is_final: Whether this is a final output file (True) or intermediate (False)
            
        Returns:
            List of command arguments for subprocess
        """
        params = cls.get_final_params() if is_final else cls.get_intermediate_params()
        
        cmd = [
            "ffmpeg", "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", concat_list_path,
            "-c:v", params["codec"],
            "-preset", params["preset"],
        ]
        
        # Add all ffmpeg parameters
        cmd.extend(params["ffmpeg_params"])
        
        # Add output file
        cmd.append(output_file)
        
        return cmd
    
    @classmethod
    def concatenate_videos(cls, input_files, output_file, is_final=True):
        """
        Concatenate video files.
        
        Args:
            input_files: List of input file paths
            output_file: Path to output file
            is_final: Whether this is a final output file (True) or intermediate (False)
            
        Returns:
            True if successful, False otherwise
        """
        import tempfile
        
        # Create temporary file for concat list
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            concat_list_path = f.name
            for file_path in input_files:
                f.write(f"file '{os.path.abspath(file_path)}'\n")
        
        try:
            # Get concat command
            cmd = cls.get_concat_command(concat_list_path, output_file, is_final)
            
            # Run command
            logger.info(f"Running FFmpeg concatenation: {' '.join(cmd)}")
            process = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                text=True
            )
            
            if process.returncode != 0:
                logger.error(f"FFmpeg concatenation failed: {process.stderr}")
                
                # Try alternative approach with copy
                logger.info("Trying alternative FFmpeg approach with copy...")
                
                alt_cmd = [
                    "ffmpeg", "-y",
                    "-f", "concat",
                    "-safe", "0",
                    "-i", concat_list_path,
                    "-c", "copy",
                    "-max_muxing_queue_size", "9999",
                    output_file
                ]
                
                process = subprocess.run(
                    alt_cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=False,
                    text=True
                )
                
                if process.returncode != 0:
                    logger.error(f"Alternative FFmpeg approach failed: {process.stderr}")
                    return False
            
            logger.info(f"Successfully concatenated {len(input_files)} clips to {output_file}")
            return True
            
        finally:
            # Clean up temporary file
            try:
                os.unlink(concat_list_path)
            except Exception as e:
                logger.warning(f"Failed to clean up temporary file: {e}")
    
    @classmethod
    def apply_encoding_params_to_clip(cls, clip, is_final=False):
        """
        Apply encoding parameters to a MoviePy clip.
        
        Args:
            clip: MoviePy clip
            is_final: Whether this is a final output clip (True) or intermediate (False)
            
        Returns:
            Original clip (parameters applied when writing)
        """
        # Store encoding parameters on the clip for later use in write_videofile
        params = cls.get_final_params() if is_final else cls.get_intermediate_params()
        clip._encoding_params = params
        
        return clip
    
    @classmethod
    def write_clip(cls, clip, output_path, fps=30, is_final=False, show_progress=True):
        """
        Write a MoviePy clip to a file with optimized encoding.
        
        Args:
            clip: MoviePy clip
            output_path: Path to output file
            fps: Frames per second
            is_final: Whether this is a final output clip (True) or intermediate (False)
            show_progress: Whether to show a progress bar
            
        Returns:
            Path to output file
        """
        # Get encoding parameters
        params = cls.get_final_params() if is_final else cls.get_intermediate_params()
        
        # Set logger option based on progress preference
        logger_setting = None if show_progress else "bar"
        
        # Write clip to file
        clip.write_videofile(
            output_path,
            fps=fps,
            codec=params["codec"],
            audio_codec=params["audio_codec"],
            preset=params["preset"],
            threads=params["threads"],
            audio_bufsize=params["audio_bufsize"],
            ffmpeg_params=params["ffmpeg_params"],
            logger=logger_setting
        )
        
        return output_path 