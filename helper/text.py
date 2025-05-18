import numpy as np
import concurrent.futures
import os
import time
import logging
from moviepy import *
from moviepy.video.fx import FadeIn
from moviepy.video.fx import FadeOut
from PIL import Image, ImageDraw, ImageFont
from helper.minor_helper import measure_time
from functools import partial

# Try to import dill for better serialization
try:
    import dill
    HAS_DILL = True
except ImportError:
    HAS_DILL = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define a standalone function for process_section that will work with multiprocessing
def _process_text_section_standalone(section, helper_resolution, helper_body_font_path, create_text_clip_func,
                                    animation="fade", font_size=60, font_path=None, with_pill=True, position='center'):
    """
    Standalone function to process a text section, used for parallel processing.

    This function is defined outside of any class to make it properly serializable.
    """
    try:
        text = section.get('text', '')
        duration = section.get('duration', 5)
        section_position = section.get('position', position)
        section_font_size = section.get('font_size', font_size)

        if not font_path:
            font_path = helper_body_font_path

        # Create text clip with the provided parameters
        return create_text_clip_func(
            text=text,
            duration=duration,
            font_size=section_font_size,
            font_path=font_path,
            position=section_position,
            animation=animation,
            with_pill=with_pill
        )
    except Exception as e:
        logger.error(f"Error creating text clip: {e}")
        return None

class TextHelper:

  def __init__(self):

      # Video resolution
      self.resolution = (1080, 1920)  # Portrait mode for shorts (width, height)

      # Font settings
      self.fonts_dir = os.path.join(os.path.dirname(__file__), 'fonts')
      os.makedirs(self.fonts_dir, exist_ok=True)
      self.title_font_path = r"D:\youtube-shorts-automation\packages\fonts\default_font.ttf"
      self.body_font_path = r"D:\youtube-shorts-automation\packages\fonts\default_font.ttf"

      # Define transitions with proper effects
      self.transitions = {
          "fade": lambda clip, duration: clip.with_effects([FadeIn(duration)]),
          "fade_out": lambda clip, duration: clip.with_effects([FadeOut(duration)]),
          "slide": lambda clip, duration: clip.with_position(lambda t: (0, 0 + t * (self.resolution[1] / duration))),
          "slide_out": lambda clip, duration: clip.with_position(lambda t: (0, self.resolution[1] - t * (self.resolution[1] / duration))),
          "zoom": lambda clip, duration: clip.resized(lambda t: 1 + t * (0.5 / duration)),
          "zoom_out": lambda clip, duration: clip.resized(lambda t: 1 - t * (0.5 / duration)),
      }

  @measure_time
  def _create_pill_image(self, size, color=(0, 0, 0, 160), radius=30):
      """
      Create a pill-shaped background image with rounded corners.

      Args:
          size (tuple): Size of the image (width, height)
          color (tuple): Color of the pill background (RGBA)
          radius (int): Radius of the rounded corners

      Returns:
          Image: PIL Image with the pill-shaped background
      """
      width, height = size
      
      # Ensure minimum dimensions to prevent drawing errors
      if width <= 0 or height <= 0:
          logger.warning(f"Invalid pill dimensions: {width}x{height}, using minimum size")
          width = max(width, 10)
          height = max(height, 10)
      
      # Ensure radius isn't too large for the image dimensions
      radius = min(radius, width // 2, height // 2)
      
      img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
      draw = ImageDraw.Draw(img)

      # Draw the rounded rectangle only if dimensions are valid
      if width > 2*radius and height > 2*radius:
          # Draw center rectangle
          draw.rectangle([(radius, 0), (width - radius, height)], fill=color)
          # Draw horizontal rectangles
          draw.rectangle([(0, radius), (width, height - radius)], fill=color)
          # Draw corner circles
          draw.ellipse([(0, 0), (radius * 2, radius * 2)], fill=color)
          draw.ellipse([(width - radius * 2, 0), (width, radius * 2)], fill=color)
          draw.ellipse([(0, height - radius * 2), (radius * 2, height)], fill=color)
          draw.ellipse([(width - radius * 2, height - radius * 2), (width, height)], fill=color)
      else:
          # Fallback to simple rectangle if dimensions are too small for rounded corners
          draw.rectangle([(0, 0), (width, height)], fill=color)

      return img

  @measure_time
  def _create_text_clip(self, text, duration=5, font_size=60, font_path=None, color='white',
                        position='center', animation="fade", animation_duration=1.0, shadow=True,
                        outline=True, with_pill=False, pill_color=(0, 0, 0, 160), pill_radius=30):
      """
      Create a text clip with various effects and animations.

      Args:
          text (str): Text content
          duration (float): Duration in seconds
          font_size (int): Font size
          font_path (str): Path to font file
          color (str): Text color
          position (str): Position of text (top, center, bottom)
          animation (str): Animation type
          animation_duration (float): Duration of animation effects
          shadow (bool): Whether to add shadow
          outline (bool): Whether to add outline
          with_pill (bool): Whether to add pill background
          pill_color (tuple): RGBA color for pill background
          pill_radius (int): Radius for pill corners

      Returns:
          TextClip: MoviePy text clip with effects
      """
      if not font_path:
          font_path = self.body_font_path

      try:
          text_clip = TextClip(
              text=text,
              font=font_path,
              font_size=font_size,
              color=color,
              method='caption',
              size=(self.resolution[0] - 100, None)
          )
      except Exception as e:
          logger.warning(f"Text rendering error with custom font: {e}. Using default.")
          text_clip = TextClip(
              text=text,
              font_size=font_size,
              font="",  # Empty string as fallback font (using system default)
              color=color,
              method='caption',
              size=(self.resolution[0] - 100, None)
          )

      text_clip = text_clip.with_duration(duration)
      clips = []

      # Add pill-shaped background if requested
      if with_pill:
          pill_image = self._create_pill_image(text_clip.size, color=pill_color, radius=pill_radius)
          pill_clip = ImageClip(np.array(pill_image), duration=duration)
          clips.append(pill_clip)

      # Add shadow effect
      if shadow:
          shadow_clip = TextClip(
              text=text,
              font=font_path,
              font_size=font_size,
              color='black',
              method='caption',
              size=(self.resolution[0] - 100, None)
          ).with_position((5, 5), relative=True).with_opacity(0.7).with_duration(duration)
          clips.append(shadow_clip)

      # Add outline effect
      if outline:
          outline_clips = []
          for dx, dy in [(-1,-1), (-1,1), (1,-1), (1,1)]:
              oc = TextClip(
                  text=text,
                  font=font_path,
                  font_size=font_size,
                  color='black',
                  method='caption',
                  size=(self.resolution[0] - 100, None)
              ).with_position((dx, dy), relative=True).with_opacity(0.5).with_duration(duration)
              outline_clips.append(oc)
          clips.extend(outline_clips)

      clips.append(text_clip)
      text_composite = CompositeVideoClip(clips)

      # Set the position of the entire composite
      text_composite = text_composite.with_position(position)

      # Apply animation
      if animation in self.transitions:
          anim_func = self.transitions[animation]
          text_composite = anim_func(text_composite, animation_duration)

      # Create transparent background for the text
      bg = ColorClip(size=self.resolution, color=(0,0,0,0)).with_duration(duration)
      final_clip = CompositeVideoClip([bg, text_composite], size=self.resolution)

      return final_clip

  # Move process_section outside of the method
  def _process_text_section(self, section, animation="fade", font_size=60, font_path=None, with_pill=True, position='center'):
      try:
          text = section.get('text', '')
          duration = section.get('duration', 5)
          section_position = section.get('position', position)
          section_font_size = section.get('font_size', font_size)

          return self._create_text_clip(
              text=text,
              duration=duration,
              font_size=section_font_size,
              font_path=font_path,
              position=section_position,
              animation=animation,
              with_pill=with_pill
          )
      except Exception as e:
          logger.error(f"Error creating text clip: {e}")
          return None

  @measure_time
  def generate_text_clips_parallel(self, script_sections, max_workers=None,
                                 animation="fade", font_size=60, font_path=None,
                                 with_pill=True, position='center'):
      """
      Generate text clips for all script sections in parallel

      Args:
          script_sections (list): List of script sections with 'text' and 'duration' keys
          max_workers (int): Maximum number of concurrent workers
          animation (str): Animation type (fade, slide, etc.)
          font_size (int): Font size
          font_path (str): Path to font file
          with_pill (bool): Whether to add pill background
          position (str): Position of text

      Returns:
          list: List of text clips
      """
      start_time = time.time()
      logger.info(f"Generating {len(script_sections)} text clips in parallel")

      if not max_workers:
          max_workers = min(len(script_sections), os.cpu_count())

      # Check if we have dill for advanced serialization
      if HAS_DILL:
          # Use ThreadPoolExecutor instead of ProcessPoolExecutor with dill
          logger.info("Using ThreadPoolExecutor with dill for text generation")
          with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
              # Process in threads with direct method calls
              futures = []
              for section in script_sections:
                  futures.append(
                      executor.submit(
                          self._process_text_section,
                          section,
                          animation=animation,
                          font_size=font_size,
                          font_path=font_path,
                          with_pill=with_pill,
                          position=position
                      )
                  )

              # Collect results
              text_clips = []
              for future in concurrent.futures.as_completed(futures):
                  try:
                      clip = future.result()
                      if clip is not None:
                          text_clips.append(clip)
                  except Exception as e:
                      logger.error(f"Error in parallel text clip generation: {e}")
      else:
          # Use ProcessPoolExecutor with the standalone function for better serialization
          logger.info("Using ProcessPoolExecutor with standalone function for text generation")
          # Create a serializable function for the creation
          def create_text_clip_wrapper(text, duration, font_size, font_path, position, animation, with_pill):
              return self._create_text_clip(
                  text=text, duration=duration, font_size=font_size,
                  font_path=font_path, position=position,
                  animation=animation, with_pill=with_pill
              )

          text_clips = []
          with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
              futures = []
              for section in script_sections:
                  futures.append(
                      executor.submit(
                          _process_text_section_standalone,
                          section,
                          self.resolution,
                          self.body_font_path,
                          create_text_clip_wrapper,
                          animation=animation,
                          font_size=font_size,
                          font_path=font_path,
                          with_pill=with_pill,
                          position=position
                      )
                  )

              # Collect results
              for future in concurrent.futures.as_completed(futures):
                  try:
                      clip = future.result()
                      if clip is not None:
                          text_clips.append(clip)
                  except Exception as e:
                      logger.error(f"Error in parallel text clip generation: {e}")

      total_time = time.time() - start_time
      logger.info(f"Generated {len(text_clips)} text clips in {total_time:.2f} seconds")

      return text_clips

  @measure_time
  def _create_word_by_word_clip(self, text, duration, font_size=60, font_path=None,
                            text_color=(255, 255, 255, 255),
                            pill_color=(0, 0, 0, 160),  # Semi-transparent black
                            position=('center', 'center')):
      """
      Create a word-by-word animation clip with pill-shaped backgrounds

          text: text to be animated
          duration: duration of the animation
          font_size: size of the font
          font_path: path to the font file
          text_color: color of the text
          pill_color: color of the pill background (with transparency)
          position: position of the text

      Returns:
          VideoClip: Word-by-word animation clip
      """
      if not font_path:
          font_path = self.body_font_path

      # Split text into words and calculate durations
      words = text.split()
      char_counts = [len(word) for word in words]
      total_chars = sum(char_counts)
      transition_duration = 0.02  # Faster transitions for better sync
      total_transition_time = transition_duration * (len(words) - 1)
      speech_duration = duration * 0.98  # Use more of the time for speech
      effective_duration = speech_duration - total_transition_time

      word_durations = []
      min_word_time = 0.2  # Slightly faster minimum word display time
      for word in words:
          char_ratio = len(word) / max(1, total_chars)
          word_time = min_word_time + (effective_duration - min_word_time * len(words)) * char_ratio
          word_durations.append(word_time)

      def make_frame_with_pill(word, font_size=font_size, font_path=font_path,
                              text_color=text_color, pill_color=pill_color):
          # Load font
          try:
              font = ImageFont.truetype(font_path, font_size)
          except Exception:
              # Fallback to default font
              font = ImageFont.load_default()

          # Get text size using modern methods compatible with newer Pillow versions
          img = Image.new('RGBA', (1, 1), (0, 0, 0, 0))
          draw = ImageDraw.Draw(img)

          # Try different methods to get text dimensions based on Pillow version
          try:
              # Use bbox for Pillow >= 8.0.0 (compatible with Python 3.13)
              bbox = draw.textbbox((0, 0), word, font=font)
              text_width = bbox[2] - bbox[0]
              text_height = bbox[3] - bbox[1]
          except AttributeError:
              # Fallback calculation for older versions
              text_width = len(word) * font_size * 0.6
              text_height = font_size * 1.2

          # Add more padding for better visual appearance
          padding_w = int(text_width * 0.3)
          padding_h = int(text_height * 0.4)
          width = text_width + padding_w * 2
          height = text_height + padding_h * 2

          # Create pill background with a better-looking radius
          radius = min(height // 2, 30)  # Limit radius for better appearance, but no more than half height
          pill_img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
          draw = ImageDraw.Draw(pill_img)

          # Draw the rounded rectangle (pill) with proper rounding
          # Left and right semi-circles
          draw.ellipse([(0, 0), (2 * radius, height)], fill=pill_color)
          draw.ellipse([(width - 2 * radius, 0), (width, height)], fill=pill_color)
          
          # Center rectangle
          if width > 2 * radius:
              draw.rectangle([(radius, 0), (width - radius, height)], fill=pill_color)

          # Draw text in the center of the pill - properly positioned
          text_x = (width - text_width) // 2  # Center horizontally
          text_y = (height - text_height) // 2  # Center vertically
          
          # Use newer Pillow method for drawing text if available
          try:
              # For Pillow >= 8.0.0
              draw.text((text_x, text_y), word, font=font, fill=text_color)
          except TypeError:
              # Fallback for older Pillow versions
              draw.text((padding_w, padding_h), word, font=font, fill=text_color)

          # Convert PIL Image to numpy array for MoviePy
          return np.array(pill_img)

      # Create a video clip for each word with its own pill background
      word_clips = []
      current_time = 0

      for i, word in enumerate(words):
          # Make a function to generate frames for this specific word
          make_frame = lambda t, word=word: make_frame_with_pill(word)

          # Create a clip for this word
          clip = VideoClip(make_frame, duration=word_durations[i])

          # Set start time
          clip = clip.with_start(current_time)

          # Update current time for next word
          current_time += word_durations[i] + transition_duration

          word_clips.append(clip)

      # Combine all word clips
      combined_clip = CompositeVideoClip(word_clips)

      # Calculate center position for the clip in the frame
      def get_position(t):
          """
          Function to position the word-by-word clip centrally in the frame.
          This ensures the text stays perfectly centered regardless of word length.
          """
          # Center the text but add a slight offset for better visual appearance
          # Default position is center, but use the user's position if provided
          if position == ('center', 'center'):
              # For center position, ensure perfect centering
              return ('center', 'center')
          elif isinstance(position, tuple) and len(position) == 2:
              # Use the user's custom position
              return position
          else:
              # Default fallback to center
              return ('center', 'center')

      # Apply center positioning with a transparent background of the full resolution size
      word_clip_width, word_clip_height = combined_clip.size
      bg = ColorClip(size=self.resolution, color=(0, 0, 0, 0)).with_duration(duration)

      # Create final clip with proper full-frame compositing to prevent cutoff
      final_clip = CompositeVideoClip(
          [bg, combined_clip.with_position(get_position)], 
          size=self.resolution
      )

      return final_clip

  # Also apply the same fix to word_by_word generation
  def _process_word_by_word_section(self, section, font_size=60, font_path=None):
      try:
          text = section.get('text', '')
          duration = section.get('duration', 5)
          position = section.get('position', ('center', 'center'))
          section_font_size = section.get('font_size', font_size)

          return self._create_word_by_word_clip(
              text=text,
              duration=duration,
              font_size=section_font_size,
              font_path=font_path,
              position=position
          )
      except Exception as e:
          logger.error(f"Error creating word-by-word clip: {e}")
          return None

  @measure_time
  def generate_word_by_word_clips_parallel(self, script_sections, max_workers=None,
                                           font_size=60, font_path=None):
      """
      Generate word-by-word text clips for all script sections in parallel

      Args:
          script_sections (list): List of script sections with 'text' and 'duration' keys
          max_workers (int): Maximum number of concurrent workers
          font_size (int): Font size
          font_path (str): Path to font file

      Returns:
          list: List of text clips
      """
      start_time = time.time()
      logger.info(f"Generating {len(script_sections)} word-by-word text clips in parallel")

      if not max_workers:
          max_workers = min(len(script_sections), os.cpu_count())

      # Use ThreadPoolExecutor for simpler serialization
      with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
          futures = []
          for section in script_sections:
              futures.append(
                  executor.submit(
                      self._process_word_by_word_section,
                      section,
                      font_size=font_size,
                      font_path=font_path
                  )
              )

          # Collect results
          text_clips = []
          for future in concurrent.futures.as_completed(futures):
              try:
                  clip = future.result()
                  if clip is not None:
                      text_clips.append(clip)
              except Exception as e:
                  logger.error(f"Error in parallel word-by-word clip generation: {e}")

      total_time = time.time() - start_time
      logger.info(f"Generated {len(text_clips)} word-by-word text clips in {total_time:.2f} seconds")

      return text_clips

  @measure_time
  def add_watermark(self, clip, watermark_text="Lazycreator", position=("right", "top"), opacity=0.7, font_size=30):
      """Add a text watermark to a video clip

      Args:
          clip (VideoClip): The video clip to add watermark to
          watermark_text (str): Text to use as watermark
          position (tuple): Position of watermark (combination of "left"/"right"/"center" and "top"/"bottom"/"center")
          opacity (float): Opacity of the watermark
          font_size (int): Font size for the watermark

      Returns:
          VideoClip: Video clip with watermark
      """
      # Create a text clip for the watermark
      txt_clip = TextClip(watermark_text, font=self.body_font_path, font_size=font_size, color='white',
                        stroke_color='gray', stroke_width=1)
      txt_clip = txt_clip.with_duration(clip.duration).with_opacity(opacity)

      # Calculate the margin (relative to resolution)
      margin_x = int(self.resolution[0] * 0.03)
      margin_y = int(self.resolution[1] * 0.02)

      # Determine the position
      if position[0] == "left":
          x_pos = margin_x
      elif position[0] == "right":
          x_pos = self.resolution[0] - txt_clip.w - margin_x
      else:  # center
          x_pos = (self.resolution[0] - txt_clip.w) // 2

      if position[1] == "top":
          y_pos = margin_y
      elif position[1] == "bottom":
          y_pos = self.resolution[1] - txt_clip.h - margin_y
      else:  # center
          y_pos = (self.resolution[1] - txt_clip.h) // 2

      # Add the text clip to the main clip
      return CompositeVideoClip([clip, txt_clip.with_position((x_pos, y_pos))])
