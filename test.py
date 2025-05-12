import os
import logging
import shutil
from moviepy import ColorClip, TextClip, CompositeVideoClip
from automation.parallel_renderer import render_clips_in_parallel

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def moving_text_position(t):
    """Named function for text position animation."""
    return (100 * t, 100 * t)

def create_test_clips():
    """Create sample clips for testing (simple and complex)."""
    resolution = (1080, 1920)
    clips = []

    font = r"D:\youtube-shorts-automation\packages\fonts\default_font.ttf"

    # Simple clip: Solid color with text
    simple_clip = ColorClip(size=resolution, color=(255, 0, 0), duration=2)
    text = TextClip(font,"Simple Clip", font_size=70, color='white', size=resolution).with_duration(2)
    simple_clip = CompositeVideoClip([simple_clip, text.with_position('center')])
    simple_clip._section_idx = 0
    simple_clip._section_text = "Simple Clip"
    simple_clip._debug_info = "Simple Clip with static text"
    clips.append(simple_clip)

    # Complex clip: With named function for position
    complex_clip = ColorClip(size=resolution, color=(0, 255, 0), duration=2)
    text = TextClip(font,"Complex Clip", font_size=70, color='white', size=resolution).with_duration(2)
    complex_clip = CompositeVideoClip([complex_clip, text.with_position(moving_text_position)])
    complex_clip._section_idx = 1
    complex_clip._section_text = "Complex Clip"
    complex_clip._debug_info = "Complex Clip with moving text"
    clips.append(complex_clip)

    return clips

def test_parallel_renderer(prerender_all=False):
    """Test the parallel renderer with sample clips."""
    output_file = f"test_output{'_prerender_all' if prerender_all else ''}.mp4"
    temp_dir = "test_temp"
    clips = create_test_clips()

    logger.info("Starting parallel rendering test")
    section_info = {
        0: {'section_idx': 0, 'section_text': "Simple Clip"},
        1: {'section_idx': 1, 'section_text': "Complex Clip"}
    }

    try:
        result = render_clips_in_parallel(
            clips=clips,
            output_file=output_file,
            fps=30,
            num_processes=2,
            temp_dir=temp_dir,
            section_info=section_info,
            prerender_all=prerender_all
        )
        if os.path.exists(result):
            logger.info(f"Test succeeded: Output created at {result}")
        else:
            logger.error("Test failed: Output file not created")
    except Exception as e:
        logger.error(f"Test failed: {e}")
    finally:
        for clip in clips:
            try:
                clip.close()
            except:
                pass
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)
        if os.path.exists(output_file):
            logger.info(f"Output file {output_file} retained for inspection")

if __name__ == "__main__":
    # Test without prerender_all
    logger.info("Testing without prerender_all")
    test_parallel_renderer(prerender_all=False)
    # Test with prerender_all
    logger.info("Testing with prerender_all")
    test_parallel_renderer(prerender_all=True)
