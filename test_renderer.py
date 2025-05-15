import os
import logging
import time
from moviepy.editor import ColorClip, TextClip, CompositeVideoClip
from automation import render_video

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("test_renderer")

def create_test_clips(num_clips=5, duration=3.0):
    """Create test clips for rendering."""
    clips = []
    
    for i in range(num_clips):
        # Create a color clip
        color = (i * 40 % 256, (i * 70) % 256, (i * 100) % 256)
        color_clip = ColorClip(size=(1080, 1920), color=color).with_duration(duration)
        
        # Add text
        text = TextClip(
            f"Clip {i+1}", 
            fontsize=100, 
            color='white', 
            font='Arial-Bold',
            stroke_color='black',
            stroke_width=2
        ).with_duration(duration).with_position('center')
        
        # Composite the clips
        composite = CompositeVideoClip([color_clip, text])
        clips.append(composite)
    
    return clips

def test_renderer():
    """Test our optimized renderer."""
    # Create output directory
    output_dir = os.path.join(os.getcwd(), "test_output")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create output filename
    output_file = os.path.join(output_dir, f"test_render_{int(time.time())}.mp4")
    
    # Create test clips
    clips = create_test_clips(num_clips=3, duration=2.0)
    
    # Render the clips
    logger.info(f"Rendering {len(clips)} clips to {output_file}")
    start_time = time.time()
    
    output_path = render_video(
        clips=clips,
        output_file=output_file,
        fps=30,
        preset="veryfast",
        parallel=True,
        memory_per_worker_gb=1.5
    )
    
    elapsed_time = time.time() - start_time
    logger.info(f"Rendering completed in {elapsed_time:.2f} seconds")
    
    # Check the output
    if os.path.exists(output_path):
        file_size = os.path.getsize(output_path) / (1024 * 1024)  # Size in MB
        logger.info(f"Output file exists: {output_path} (Size: {file_size:.2f} MB)")
        return True
    else:
        logger.error(f"Output file does not exist: {output_path}")
        return False

if __name__ == "__main__":
    success = test_renderer()
    exit(0 if success else 1) 