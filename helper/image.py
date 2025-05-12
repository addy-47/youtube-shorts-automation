import time
import random
import os
import requests
import logging
import concurrent.futures
from moviepy  import VideoClip, concatenate_videoclips, ColorClip, CompositeVideoClip, ImageClip, TextClip
from helper.blur import custom_blur, custom_edge_blur
from helper.minor_helper import measure_time
from helper.text import TextHelper
from dotenv import load_dotenv

load_dotenv()

huggingface_api_key = os.getenv("HUGGINGFACE_API_KEY")
hf_api_url = os.getenv("HUGGINGFACE_API_URL")
hf_headers = {"Authorization": f"Bearer {huggingface_api_key}"}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

resolution = (1080, 1920)  # Assuming a standard resolution for YouTube Shorts

# Get temp directory from environment variable or use default
TEMP_DIR = os.getenv("TEMP_DIR", "D:\\youtube-shorts-automation\\temp")
# Create images subdirectory
temp_dir = os.path.join(TEMP_DIR, "generated_images")
os.makedirs(temp_dir, exist_ok=True)  # Create temp directory if it doesn't exist

@measure_time
def generate_images_parallel(prompts, style="photorealistic", max_workers=None):
    """
    Generate multiple images in parallel based on prompts

    Args:
        prompts (list): List of image generation prompts
        style (str): Style to apply to the images
        max_workers (int): Maximum number of concurrent workers

    Returns:
        list: List of paths to generated images
    """
    start_time = time.time()
    logger.info(f"Generating {len(prompts)} images in parallel")

    def generate_single_image(prompt):
        try:
            return _generate_image_from_prompt(prompt, style)
        except Exception as e:
            logger.error(f"Error generating image: {e}")
            return None

    if not max_workers:
        # Use fewer workers for API calls to avoid rate limiting
        max_workers = min(len(prompts), 4)

    # Image generation is I/O bound (API calls), so use ThreadPoolExecutor
    image_paths = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(generate_single_image, prompt) for prompt in prompts]

        for future in concurrent.futures.as_completed(futures):
            try:
                image_path = future.result()
                if image_path:
                    image_paths.append(image_path)
            except Exception as e:
                logger.error(f"Failed to get result from image generation: {e}")

    total_time = time.time() - start_time
    logger.info(f"Generated {len(image_paths)} images in {total_time:.2f} seconds")

    return image_paths

@measure_time
def _generate_image_from_prompt(prompt, style="photorealistic", file_path=None):
  """
  Generate an image using Hugging Face Diffusion API based on prompt

  Args:
      prompt (str): Image generation prompt
      style (str): Style to apply to the image (e.g., "digital art", "realistic", "photorealistic")
      file_path (str): Path to save the image, if None a path will be generated

  Returns:
      str: Path to the generated image or None if failed
  """
  if not file_path:
      file_path = os.path.join(temp_dir, f"gen_img_{int(time.time())}_{random.randint(1000, 9999)}.png")

  # Remove any existing style descriptors from the prompt
  style_keywords = ["digital art", "photorealistic", "oil painting", "realistic", "anime",
                    "concept art", "cinematic", "cartoon", "3d render", "watercolor",
                    "sketch", "illustration", "painting"]

  # First, clean the prompt of any existing style descriptors
  clean_prompt = prompt
  for keyword in style_keywords:
      clean_prompt = clean_prompt.replace(f", {keyword}", "")
      clean_prompt = clean_prompt.replace(f" {keyword}", "")
      clean_prompt = clean_prompt.replace(f"{keyword} ", "")
      clean_prompt = clean_prompt.replace(f"{keyword},", "")

  # Clean up any double commas or spaces that might have been created
  while ",," in clean_prompt:
      clean_prompt = clean_prompt.replace(",,", ",")
  while "  " in clean_prompt:
      clean_prompt = clean_prompt.replace("  ", " ")
  clean_prompt = clean_prompt.strip(" ,")

  # Now add the desired style and quality enhancements
  enhanced_prompt = f"{clean_prompt}, {style}, highly detailed, crisp focus, 4K, high resolution"

  logger.info(f"Original prompt: {prompt[:50]}...")
  logger.info(f"Using style: {style}")
  logger.info(f"Enhanced prompt: {enhanced_prompt[:50]}...")

  retry_count = 0
  max_retries = 3
  success = False
  initial_wait_time = 20  # Starting wait time in seconds

  # Check if Hugging Face API key is available
  if not huggingface_api_key:
      logger.error("No Hugging Face API key provided. Will fall back to shorts_maker_V.")
      return None

  while not success and retry_count < max_retries:
      try:
          # Make request to Hugging Face API
          response = requests.post(
              hf_api_url,
              headers=hf_headers,
              json={"inputs": enhanced_prompt},
              timeout=30  # Add timeout to prevent hanging indefinitely
          )

          if response.status_code == 200:
              # Save the image
              with open(file_path, "wb") as f:
                  f.write(response.content)
              logger.info(f"Image saved to {file_path}")
              success = True
          else:
              # If model is loading, wait and retry
              try:
                  if "application/json" in response.headers.get("Content-Type", ""):
                      response_json = response.json()
                      if response.status_code == 503 and "estimated_time" in response_json:
                          wait_time = response_json.get("estimated_time", initial_wait_time)
                          logger.info(f"Model is loading. Waiting {wait_time} seconds...")
                          time.sleep(wait_time)
                      else:
                          # Other error
                          logger.error(f"Error generating image: {response.status_code} - {response.text}")
                          time.sleep(initial_wait_time)  # Wait before retrying
                  else:
                      # Non-JSON response (HTML error page)
                      logger.error(f"Non-JSON error response: {response.status_code}")
                      # For 503 errors, wait longer before retry
                      if response.status_code == 503:
                          wait_time = initial_wait_time * (retry_count + 1)  # Gradually increase wait time
                          logger.info(f"Service unavailable (503). Waiting {wait_time} seconds before retry...")
                          time.sleep(wait_time)
                      else:
                          time.sleep(initial_wait_time)  # Wait before retrying
              except ValueError:
                  # Non-JSON response
                  logger.error(f"Could not parse response: {response.status_code}")
                  time.sleep(initial_wait_time)  # Wait before retrying

              # Check if we should fall back before trying more retries
              if response.status_code == 503 and retry_count >= 1:
                  logger.warning("Multiple 503 errors from Hugging Face API. Falling back to shorts_maker_V.")
                  return None

              retry_count += 1
      except requests.exceptions.RequestException as e:
          logger.error(f"Network error during image generation: {e}")
          retry_count += 1
          time.sleep(initial_wait_time)
      except Exception as e:
          logger.error(f"Unexpected exception during image generation: {e}")
          retry_count += 1
          time.sleep(initial_wait_time)

  # If all retries failed, return None to signal fallback to shorts_maker_V
  if not success:
      logger.error("Failed to generate image with Hugging Face API after multiple attempts")
      return None

  return file_path

@measure_time
def create_image_clips_parallel(image_paths, durations, texts=None, with_zoom=True, max_workers=None):
    """
    Create still image clips in parallel

    Args:
        image_paths (list): List of paths to images
        durations (list): List of durations for each clip
        texts (list): Optional list of text overlays
        with_zoom (bool): Whether to add zoom effect
        max_workers (int): Maximum number of concurrent workers

    Returns:
        list: List of video clips
    """
    start_time = time.time()
    logger.info(f"Creating {len(image_paths)} image clips in parallel")

    if not texts:
        texts = [None] * len(image_paths)

    # Make sure all lists have the same length
    if len(durations) != len(image_paths):
        logger.warning(f"Duration list length {len(durations)} doesn't match image paths length {len(image_paths)}")
        # Pad or truncate durations list
        if len(durations) < len(image_paths):
            durations.extend([5.0] * (len(image_paths) - len(durations)))
        else:
            durations = durations[:len(image_paths)]

    if len(texts) != len(image_paths):
        texts = [None] * len(image_paths)

    def create_clip(args):
        image_path, duration, text = args
        try:
            return _create_still_image_clip(image_path, duration, text, with_zoom=with_zoom)
        except Exception as e:
            logger.error(f"Error creating image clip: {e}")
            return None

    if not max_workers:
        max_workers = min(len(image_paths), os.cpu_count())

    # Image clip creation is CPU bound, so use ProcessPoolExecutor
    clips = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(create_clip, (img, dur, txt))
                  for img, dur, txt in zip(image_paths, durations, texts)]

        for future in concurrent.futures.as_completed(futures):
            try:
                clip = future.result()
                if clip:
                    clips.append(clip)
            except Exception as e:
                logger.error(f"Failed to get result from image clip creation: {e}")

    total_time = time.time() - start_time
    logger.info(f"Created {len(clips)} image clips in {total_time:.2f} seconds")

    return clips

@measure_time
def _create_still_image_clip(image_path, duration, text=None, text_position=('center','center'),
                          font_size=60, with_zoom=True, zoom_factor=0.05):
  """
  Create a still image clip with optional text and zoom effect

  Args:
      image_path (str): Path to the image
      duration (float): Duration of the clip in seconds
      text (str): Optional text overlay
      text_position (str): Position of text ('top', 'center', ('center','center'))
      font_size (int): Font size for text
      with_zoom (bool): Whether to add a subtle zoom effect
      zoom_factor (float): Rate of zoom (higher = faster zoom)

  Returns:
      VideoClip: MoviePy clip containing the image and effects
  """
  # Load image
  image = ImageClip(image_path)

  # resized to fill screen while maintaining aspect ratio
  img_ratio = image.size[0] / image.size[1]
  target_ratio = resolution[0] / resolution[1]

  if img_ratio > target_ratio:  # Image is wider
      new_height = resolution[1]
      new_width = int(new_height * img_ratio)
  else:  # Image is taller
      new_width = resolution[0]
      new_height = int(new_width / img_ratio)

  image = image.resized(newsize=(new_width, new_height))

  # Center crop if needed
  if new_width > resolution[0] or new_height > resolution[1]:
      x_center = new_width // 2
      y_center = new_height // 2
      x1 = max(0, x_center - resolution[0] // 2)
      y1 = max(0, y_center - resolution[1] // 2)
      image = image.crop(x1=x1, y1=y1, width=resolution[0], height=resolution[1])

  # Add zoom effect if requested
  if with_zoom:
      def zoom(t):
          # Start at 1.0 zoom and gradually increase
          zoom_level = 1 + (t / duration) * zoom_factor
          return zoom_level

      # Replace lambda with named function
      def zoom_func(t):
          return zoom(t)

      image = image.resized(zoom_func)

  # Make sure the image is the right duration
  image = image.with_duration(duration)

  # Add text if provided
  if text:
      txt = TextClip(
          text=text,
          font_size=font_size,
          color='white',
          font=r"D:\youtube-shorts-automation\packages\fonts\default_font.ttf",
          stroke_color='black',
          stroke_width=1,
          method='caption',
          size=(resolution[0] - 100, None)
      ).with_duration(duration)

      # Add shadow for text
      txt_shadow = TextClip(
          text=text,
          font_size=font_size,
          color='black',
          font=r"D:\youtube-shorts-automation\packages\fonts\default_font.ttf",
          method='caption',
          size=(resolution[0] - 100, None)
      ).with_position((2, 2), relative=True).with_opacity(0.6).with_duration(duration)

      # Position the text
      txt = txt.with_position(text_position)
      txt_shadow = txt_shadow.with_position(text_position)

      # Composite all together
      return CompositeVideoClip([image, txt_shadow, txt], size=resolution)
  else:
      return image
