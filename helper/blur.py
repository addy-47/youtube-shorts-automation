import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from helper.minor_helper import measure_time
from dotenv import load_dotenv

load_dotenv()

@measure_time
def custom_blur(clip, radius=5):
    """
    Apply a Gaussian blur effect to video clips

    Args:
        clip (VideoClip): Video clip to blur
        radius (int): Blur radius

    Returns:
        VideoClip: Blurred video clip
    """
    def blur_frame(get_frame, t):
        frame = get_frame(t)
        img = Image.fromarray(frame)
        blurred = img.filter(ImageFilter.GaussianBlur(radius=radius))
        return np.array(blurred)

    def apply_blur(get_frame, t):
        return blur_frame(get_frame, t)

    return clip.fl(apply_blur)

@measure_time
def custom_edge_blur(clip, edge_width=50, radius=10):
    """
    Apply blur only to the edges of a video clip

    Args:
        clip (VideoClip): Video clip to blur edges of
        edge_width (int): Width of the edge to blur
        radius (int): Blur radius

    Returns:
        VideoClip: Video clip with blurred edges
    """
    def blur_frame(get_frame, t):
        frame = get_frame(t)
        img = Image.fromarray(frame)
        width, height = img.size

        # Create a mask for the unblurred center
        mask = Image.new('L', (width, height), 0)
        draw = ImageDraw.Draw(mask)
        draw.rectangle(
            [(edge_width, edge_width), (width - edge_width, height - edge_width)],
            fill=255
        )

        # Blur the entire image
        blurred = img.filter(ImageFilter.GaussianBlur(radius=radius))

        # Composite the blurred image with the original using the mask
        composite = Image.composite(img, blurred, mask)

        return np.array(composite)

    def apply_edge_blur(get_frame, t):
        return blur_frame(get_frame, t)

    return clip.fl(apply_edge_blur)
