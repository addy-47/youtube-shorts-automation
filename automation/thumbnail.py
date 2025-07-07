import os
import time
import random
import logging
import requests
import tempfile
import textwrap
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from dotenv import load_dotenv
from datetime import datetime
import numpy as np
import shutil

# Custom helpers
from helper.minor_helper import measure_time, cleanup_temp_directories

# Configure logging
logger = logging.getLogger(__name__)

# Timer function for performance monitoring
def measure_time(func):
    """Decorator to measure the execution time of functions"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        start_datetime = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        logger.info(f"STARTING {func.__name__} at {start_datetime}")
        result = func(*args, **kwargs)
        end_time = time.time()
        duration = end_time - start_time
        logger.info(f"COMPLETED {func.__name__} in {duration:.2f} seconds")
        return result
    return wrapper

# Get temp directory from environment variable or use default
TEMP_DIR = os.getenv("TEMP_DIR", os.path.join(os.path.dirname(os.path.dirname(__file__)), "temp"))
# Ensure temp directory exists
os.makedirs(TEMP_DIR, exist_ok=True)

class ThumbnailGenerator:
    def __init__(self, output_dir="output"):
        """
        Initialize the thumbnail generator with necessary settings

        Args:
            output_dir (str): Directory to save output thumbnails
        """
        # Load environment variables
        load_dotenv()

        # Setup directories
        self.output_dir = output_dir
        self.temp_dir = os.path.join(TEMP_DIR, f"thumbnail_{int(time.time())}")
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(self.temp_dir, exist_ok=True)

        # Font settings
        self.fonts_dir = os.path.join(os.path.dirname(__file__), 'fonts')
        self.title_font_path = r"/home/addy/projects/youtube-shorts-automation/packages/fonts/default_font.ttf"

        # Setup API credentials
        self.huggingface_api_key = os.getenv("HUGGINGFACE_API_KEY")
        self.hf_model = os.getenv("HF_MODEL", "stabilityai/stable-diffusion-2-1")
        self.hf_api_url = f"https://api-inference.huggingface.co/models/{self.hf_model}"
        self.hf_headers = {"Authorization": f"Bearer {self.huggingface_api_key}"}

        # Unsplash API (for fallback)
        self.unsplash_api_key = os.getenv("UNSPLASH_API_KEY")
        self.unsplash_api_url = "https://api.unsplash.com/search/photos"

        # Thumbnail settings
        self.thumbnail_size = (1080, 1920)  # YouTube Shorts recommended size (9:16 aspect ratio)

    @measure_time
    def generate_image_huggingface(self, prompt, style="photorealistic", file_path=None):
        """
        Generate an image using Hugging Face Diffusion API based on prompt

        Args:
            prompt (str): Image generation prompt
            style (str): Style to apply to the image
            file_path (str): Path to save the image, if None a path will be generated

        Returns:
            str: Path to the generated image or None if failed
        """
        if not file_path:
            file_path = os.path.join(self.temp_dir, f"thumbnail_{int(time.time())}_{random.randint(1000, 9999)}.png")

        # Remove any existing style descriptors from the prompt
        style_keywords = ["digital art", "photorealistic", "oil painting", "realistic", "anime",
                         "concept art", "cinematic", "cartoon", "3d render", "watercolor",
                         "sketch", "illustration", "painting"]

        # Clean the prompt of any existing style descriptors
        clean_prompt = prompt
        for keyword in style_keywords:
            clean_prompt = clean_prompt.replace(f", {keyword}", "")
            clean_prompt = clean_prompt.replace(f" {keyword}", "")
            clean_prompt = clean_prompt.replace(f"{keyword} ", "")
            clean_prompt = clean_prompt.replace(f"{keyword},", "")

        # Clean up any double commas or spaces
        while ",," in clean_prompt:
            clean_prompt = clean_prompt.replace(",,", ",")
        while "  " in clean_prompt:
            clean_prompt = clean_prompt.replace("  ", " ")
        clean_prompt = clean_prompt.strip(" ,")

        # Add the desired style and quality enhancements tailored for thumbnails
        enhanced_prompt = f"{clean_prompt}, {style}, eye-catching thumbnail, vibrant, professional, highly detailed, 4K, high resolution, perfect for YouTube Shorts"

        logger.info(f"Hugging Face prompt: {enhanced_prompt[:75]}...")

        # Check if Hugging Face API key is available
        if not self.huggingface_api_key:
            logger.error("No Hugging Face API key provided. Will fall back to Unsplash.")
            return None

        retry_count = 0
        max_retries = 2
        success = False
        initial_wait_time = 15  # Starting wait time in seconds

        while not success and retry_count < max_retries:
            try:
                # Make request to Hugging Face API
                response = requests.post(
                    self.hf_api_url,
                    headers=self.hf_headers,
                    json={"inputs": enhanced_prompt},
                    timeout=30  # Timeout to prevent hanging
                )

                if response.status_code == 200:
                    # Save the image
                    with open(file_path, "wb") as f:
                        f.write(response.content)
                    logger.info(f"Thumbnail image saved to {file_path}")
                    success = True
                else:
                    # Error handling
                    logger.error(f"Error from Hugging Face API: {response.status_code}")
                    time.sleep(initial_wait_time * (retry_count + 1))
                    retry_count += 1
            except Exception as e:
                logger.error(f"Exception during Hugging Face API call: {e}")
                retry_count += 1
                time.sleep(initial_wait_time)

        # If all retries failed, return None to signal fallback to Unsplash
        if not success:
            logger.error("Failed to generate image with Hugging Face API after multiple attempts")
            return None

        return file_path

    @measure_time
    def fetch_image_unsplash(self, query, file_path=None):
        """
        Fetch an image from Unsplash API based on query

        Args:
            query (str): Search query for Unsplash
            file_path (str): Path to save the image, if None a path will be generated

        Returns:
            str: Path to the downloaded image or None if failed
        """
        if not file_path:
            file_path = os.path.join(self.temp_dir, f"thumbnail_unsplash_{int(time.time())}_{random.randint(1000, 9999)}.jpg")

        # Check if Unsplash API key is available
        if not self.unsplash_api_key:
            logger.error("No Unsplash API key provided.")
            return None

        try:
            # Clean query for Unsplash search
            clean_query = query.replace("eye-catching", "").replace("thumbnail", "").replace("YouTube Shorts", "")
            # Remove any double spaces
            while "  " in clean_query:
                clean_query = clean_query.replace("  ", " ")
            clean_query = clean_query.strip(" ,")

            logger.info(f"Searching Unsplash with query: {clean_query}")

            # Make request to Unsplash API
            params = {
                "query": clean_query,
                "orientation": "landscape",
                "per_page": 30,
                "client_id": self.unsplash_api_key
            }

            response = requests.get(self.unsplash_api_url, params=params, timeout=10)

            if response.status_code == 200:
                data = response.json()

                # Check if we have results
                if data["results"] and len(data["results"]) > 0:
                    # Pick a random image from top results for variety
                    max_index = min(10, len(data["results"]))
                    image_data = random.choice(data["results"][:max_index])
                    image_url = image_data["urls"]["regular"]

                    # Download the image
                    img_response = requests.get(image_url, timeout=10)
                    if img_response.status_code == 200:
                        with open(file_path, "wb") as f:
                            f.write(img_response.content)
                        logger.info(f"Unsplash image downloaded to {file_path}")

                        # Add attribution as required by Unsplash API guidelines
                        attribution = f"Photo by {image_data['user']['name']} on Unsplash"
                        logger.info(f"Image attribution: {attribution}")

                        return file_path
                    else:
                        logger.error(f"Failed to download image from Unsplash: {img_response.status_code}")
                else:
                    logger.error("No results found on Unsplash")
            else:
                logger.error(f"Unsplash API error: {response.status_code} - {response.text}")

        except Exception as e:
            logger.error(f"Error fetching image from Unsplash: {e}")

        return None

    @measure_time
    def create_thumbnail(self, title, image_path, output_path=None):
        """
        Create a thumbnail with text overlay using the given image

        Args:
            title (str): Title text to overlay on the thumbnail
            image_path (str): Path to the base image
            output_path (str): Path to save the final thumbnail

        Returns:
            str: Path to the created thumbnail
        """
        if not output_path:
            output_path = os.path.join(self.output_dir, f"thumbnail_{int(time.time())}.jpg")

        logger.info(f"Creating thumbnail for title: '{title}'")
        logger.info(f"Using base image: {image_path}")
        logger.info(f"Output path: {output_path}")

        try:
            # Open the image
            img = Image.open(image_path)
            logger.info(f"Base image size: {img.size}")

            # resize to YouTube thumbnail dimensions
            img = img.resize(self.thumbnail_size, Image.LANCZOS)
            logger.info(f"resized image to YouTube thumbnail dimensions: {self.thumbnail_size}")

            # Convert to RGBA to support transparency for overlay
            img = img.convert("RGBA")

            # Create a semi-transparent dark overlay for better text visibility
            overlay = Image.new('RGBA', self.thumbnail_size, (0, 0, 0, 0))
            overlay_draw = ImageDraw.Draw(overlay)

            # Draw a gradient overlay (darker at bottom)
            for y in range(self.thumbnail_size[1] // 2, self.thumbnail_size[1]):
                # Calculate alpha based on y position (more transparent at top, more opaque at bottom)
                alpha = int(180 * (y - self.thumbnail_size[1] // 2) / (self.thumbnail_size[1] // 2))
                overlay_draw.line([(0, y), (self.thumbnail_size[0], y)], fill=(0, 0, 0, alpha))

            # Composite the image with the overlay
            img = Image.alpha_composite(img, overlay)
            logger.info("Added gradient overlay to thumbnail")

            # Add title text
            draw = ImageDraw.Draw(img)

            # Try to load the font, use default if fails
            try:
                # Calculate appropriate font size based on title length
                font_size = 70 if len(title) < 30 else 60 if len(title) < 50 else 50
                logger.info(f"Selected font size: {font_size} based on title length: {len(title)}")
                font = ImageFont.truetype(self.title_font_path, font_size)
                logger.info(f"Loaded custom font from: {self.title_font_path}")
            except Exception as e:
                # Fallback to default font
                font = ImageFont.load_default()
                logger.warning(f"Using default font as custom font could not be loaded: {e}")

            # Wrap text to fit thumbnail width
            wrapped_text = textwrap.fill(title, width=30)
            logger.info(f"Wrapped text: '{wrapped_text}'")

            # Calculate text position (centered horizontally, near bottom vertically)
            text_bbox = draw.textbbox((0, 0), wrapped_text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]

            text_x = (self.thumbnail_size[0] - text_width) // 2
            text_y = self.thumbnail_size[1] - text_height - 50  # 50px from bottom
            logger.info(f"Text position calculated: x={text_x}, y={text_y}, width={text_width}, height={text_height}")

            # Draw text shadow/outline for better visibility
            outline_width = 3
            for dx, dy in [(dx, dy) for dx in range(-outline_width, outline_width+1, 2)
                                     for dy in range(-outline_width, outline_width+1, 2)]:
                draw.text((text_x + dx, text_y + dy), wrapped_text, font=font, fill=(0, 0, 0, 255))
            logger.info("Added text shadow/outline for visibility")

            # Draw the main text in white
            draw.text((text_x, text_y), wrapped_text, font=font, fill=(255, 255, 255, 255))
            logger.info("Added main title text")

            # Add a small "SHORTS" label in the corner
            shorts_label = "SHORTS"
            try:
                shorts_font = ImageFont.truetype(self.title_font_path, 30)
            except Exception as e:
                shorts_font = ImageFont.load_default()
                logger.warning(f"Using default font for SHORTS label due to error: {e}")

            shorts_bbox = draw.textbbox((0, 0), shorts_label, font=shorts_font)
            shorts_width = shorts_bbox[2] - shorts_bbox[0]
            shorts_height = shorts_bbox[3] - shorts_bbox[1]
            logger.info(f"SHORTS label dimensions: width={shorts_width}, height={shorts_height}")

            # Draw rounded rectangle background for SHORTS label
            label_padding = 10
            label_x = self.thumbnail_size[0] - shorts_width - label_padding * 2 - 20
            label_y = 20
            label_height = shorts_height + 10
            logger.info(f"SHORTS label position: x={label_x}, y={label_y}, height={label_height}")

            # Draw pill background for "SHORTS" text
            draw.rectangle(
                [(label_x, label_y), (label_x + shorts_width + label_padding * 2, label_y + label_height)],
                fill=(255, 0, 0, 200),
                outline=(255, 255, 255, 200),
                width=2  # Make outline more visible
            )
            logger.info("Added rectangle background for SHORTS label")

            # Draw SHORTS text - properly centered in the rectangle
            text_y_offset = (label_height - shorts_height) // 2
            text_position = (label_x + label_padding, label_y + text_y_offset)
            logger.info(f"SHORTS text position: {text_position}, y_offset={text_y_offset}")
            draw.text(
                text_position,
                shorts_label,
                font=shorts_font,
                fill=(255, 255, 255, 255)
            )
            logger.info("Added SHORTS text label")

            # Convert back to RGB for saving as JPG
            img = img.convert("RGB")

            # Save the final thumbnail with high quality
            img.save(output_path, quality=95)
            logger.info(f"Thumbnail created and saved to {output_path}")

            return output_path

        except Exception as e:
            logger.error(f"Error creating thumbnail: {e}")
            import traceback
            logger.error(f"Thumbnail creation traceback: {traceback.format_exc()}")
            return None

    @measure_time
    def generate_thumbnail(self, title, script_sections=None, prompt=None, style="photorealistic", output_path=None):
        """
        Main function to generate a thumbnail for a YouTube Short

        Args:
            title (str): Title of the short
            script_sections (list): List of script sections for context
            prompt (str): Custom prompt for image generation (optional)
            style (str): Style of image to generate
            output_path (str): Path to save the final thumbnail

        Returns:
            str: Path to the generated thumbnail
        """
        # Create a default output path if none provided
        if not output_path:
            timestamp = int(time.time())
            output_filename = f"thumbnail_{timestamp}.jpg"
            output_path = os.path.join(self.output_dir, output_filename)

        # Generate or use the image prompt
        if not prompt:
            if script_sections:
                prompt = self.generate_thumbnail_query(title, script_sections)
            else:
                prompt = f"{title}, eye-catching, vibrant colors, compelling visual, engaging thumbnail for YouTube Shorts"

        logger.info(f"Using thumbnail prompt: {prompt}")

        # First try with Hugging Face
        image_path = self.generate_image_huggingface(prompt, style=style)

        # Fallback to Unsplash if Hugging Face fails
        if not image_path:
            logger.info("Falling back to Unsplash for thumbnail image")
            image_path = self.fetch_image_unsplash(prompt)

            # If both methods fail, create a basic text-based thumbnail
            if not image_path:
                logger.warning("Both Hugging Face and Unsplash failed. Creating text-only thumbnail.")
                # Create solid color background with text
                img = Image.new('RGB', self.thumbnail_size, color=(33, 33, 33))
                draw = ImageDraw.Draw(img)

                try:
                    font = ImageFont.truetype(self.title_font_path, 70)
                except:
                    font = ImageFont.load_default()

                wrapped_text = textwrap.fill(title, width=25)

                # Calculate text position
                text_bbox = draw.textbbox((0, 0), wrapped_text, font=font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]

                text_x = (self.thumbnail_size[0] - text_width) // 2
                text_y = (self.thumbnail_size[1] - text_height) // 2

                # Draw text with simple shadow
                draw.text((text_x+2, text_y+2), wrapped_text, font=font, fill=(0, 0, 0))
                draw.text((text_x, text_y), wrapped_text, font=font, fill=(255, 255, 255))

                # Add SHORTS text
                draw.text((10, 10), "SHORTS", font=font, fill=(255, 0, 0))

                # Save the image
                temp_path = os.path.join(self.temp_dir, f"text_thumbnail_{int(time.time())}.jpg")
                img.save(temp_path, quality=95)
                image_path = temp_path

        # Create the final thumbnail with text overlay
        thumbnail_path = self.create_thumbnail(title, image_path, output_path)

        if thumbnail_path:
            logger.info(f"Successfully generated thumbnail at: {thumbnail_path}")
            return thumbnail_path
        else:
            logger.error("Failed to generate thumbnail")
            return None

    def cleanup(self):
        """Clean up temporary files"""
        from helper.minor_helper import cleanup_temp_directories

        if hasattr(self, 'temp_dir') and self.temp_dir:
            logger.info(f"Cleaning up thumbnail temporary directory: {self.temp_dir}")
            cleanup_temp_directories(specific_dir=self.temp_dir)


# Simple test function
def test_thumbnail_generator():
    generator = ThumbnailGenerator(output_dir="output/thumbnails")
    title = "How AI is Revolutionizing Healthcare"
    script_sections = [
        {"text": "AI is transforming how doctors diagnose diseases with unprecedented accuracy.", "duration": 5},
        {"text": "Machine learning algorithms can now detect patterns that human doctors might miss.", "duration": 5},
        {"text": "This technology is already saving lives in hospitals around the world.", "duration": 5}
    ]

    thumbnail_path = generator.generate_thumbnail(
        title=title,
        script_sections=script_sections,
        style="photorealistic"
    )

    print(f"Thumbnail generated at: {thumbnail_path}")
    generator.cleanup()


if __name__ == "__main__":
    # Set up basic logging for stand-alone testing
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
    )

    # Run test
    test_thumbnail_generator()
