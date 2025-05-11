import time
import random
import os
import requests
import logging
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
temp_dir = "temp_images"  # Temporary directory for generated images

@measure_time
def _generate_image_from_prompt(self, prompt, style="photorealistic", file_path=None):
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
      file_path = os.path.join( temp_dir, f"gen_img_{int(time.time())}_{random.randint(1000, 9999)}.png")

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
  if not  huggingface_api_key:
      logger.error("No Hugging Face API key provided. Will fall back to shorts_maker_V.")
      return None

  while not success and retry_count < max_retries:
      try:
          # Make request to Hugging Face API
          response = requests.post(
               hf_api_url,
              headers= hf_headers,
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
def _create_still_image_clip(self, image_path, duration, text=None, text_position=('center','center'),
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
  target_ratio =  resolution[0] /  resolution[1]

  if img_ratio > target_ratio:  # Image is wider
      new_height =  resolution[1]
      new_width = int(new_height * img_ratio)
  else:  # Image is taller
      new_width =  resolution[0]
      new_height = int(new_width / img_ratio)

  image = image.resized(newsize=(new_width, new_height))

  # Center crop if needed
  if new_width >  resolution[0] or new_height >  resolution[1]:
      x_center = new_width // 2
      y_center = new_height // 2
      x1 = max(0, x_center -  resolution[0] // 2)
      y1 = max(0, y_center -  resolution[1] // 2)
      image = image.crop(x1=x1, y1=y1, width= resolution[0], height= resolution[1])

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

  # Set the duration
  image = image.with_duration(duration)

  # Add text if provided
  if text:
      try:
          # Try using the text clip function from YTShortsCreator_V
          txt_clip = TextHelper._create_text_clip(
              text,
              duration=duration,
              font_size=font_size,
              position=text_position,
              with_pill=True
          )
          # Combine image and text
          return CompositeVideoClip([image, txt_clip], size= resolution)
      except Exception as e:
          logger.error(f"Error creating text clip using V creator: {e}")
          # Fallback to a simple text implementation if the V creator fails
          try:
              # Use the simpler built-in MoviePy TextClip without fancy effects
              simple_text_clip = TextClip(
                  text=text,
                  font_size=font_size,
                  font=r"D:\youtube-shorts-automation\packages\fonts\default_font.ttf",
                  color='white',
                  method='caption',
                  size=(int( resolution[0] * 0.9), None)
              ).with_position(('center', int( resolution[1] * 0.85))).with_duration(duration)

              # Create a semi-transparent background for better readability
              text_w, text_h = simple_text_clip.size
              bg_width = text_w + 40
              bg_height = text_h + 40
              bg_clip = ColorClip(size=(bg_width, bg_height), color=(0, 0, 0, 128))
              bg_clip = bg_clip.with_position(('center', int( resolution[1] * 0.85) - 20)).with_duration(duration).with_opacity(0.7)

              # Combine all elements
              return CompositeVideoClip([image, bg_clip, simple_text_clip], size= resolution)
          except Exception as e2:
              logger.error(f"Fallback text clip also failed: {e2}")
              # If all text methods fail, just return the image without text
              logger.warning("Returning image without text overlay due to text rendering failures")
              return image
  return image
