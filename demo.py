from moviepy.editor import ImageClip, CompositeVideoClip
from video_maker import YTShortsCreator  # Import your existing VideoMaker class
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_word_animation():
    # Create instance of your VideoMaker class
    video_maker = YTShortsCreator()

    # Sample text to animate
    sample_text = "This is a test of the word by word animation with curved backgrounds"

    # Duration of the animation in seconds
    duration = 5.0

    # Load the background image
    try:
        background = ImageClip("sample_img.jpg")
        # Resize if needed to match your resolution
        background = background.resize(video_maker.resolution)
        background = background.set_duration(duration)
    except Exception as e:
        logger.error(f"Error loading background image: {e}")
        return

    # Create word-by-word animation
    text_clip = video_maker.create_word_by_word_clip(
        text=sample_text,
        duration=duration,
        font_size=60,
        font_path=video_maker.body_font_path,
        text_color=(255, 255, 255, 255),
        pill_color=(0, 0, 0, 160),
        position=('center', 'center'),
    )

    # Combine background and text
    final_clip = CompositeVideoClip([background, text_clip])

    # Write the output video
    output_path = "word_animation_test.mp4"
    final_clip.write_videofile(
        output_path,
        fps=30,
        codec='libx264',
        audio=False
    )

    logger.info(f"Animation saved to {output_path}")

if __name__ == "__main__":
    test_word_animation()
