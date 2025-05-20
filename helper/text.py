import numpy as np
import concurrent.futures
import os
import time
import logging
from moviepy import *
from moviepy.video.fx import FadeIn
from moviepy.video.fx import FadeOut
from moviepy.video.fx import CrossFadeOut,CrossFadeIn
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
      img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
      draw = ImageDraw.Draw(img)

      # Draw the rounded rectangle
      draw.rectangle([(radius, 0), (width - radius, height)], fill=color)
      draw.rectangle([(0, radius), (width, height - radius)], fill=color)
      draw.ellipse([(0, 0), (radius * 2, radius * 2)], fill=color)
      draw.ellipse([(width - radius * 2, 0), (width, radius * 2)], fill=color)
      draw.ellipse([(0, height - radius * 2), (radius * 2, height)], fill=color)
      draw.ellipse([(width - radius * 2, height - radius * 2), (width, height)], fill=color)

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
      logger.info(f"Creating text clip: '{text[:30]}...' with duration {duration:.2f}s")
      
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
      
      # Ensure the duration is correct after composition
      final_clip = final_clip.with_duration(duration)
      logger.info(f"Created text clip with final duration: {final_clip.duration:.2f}s")

      return final_clip

  # Move process_section outside of the method
  def _process_text_section(self, section, animation="fade", font_size=60, font_path=None, with_pill=True, position='center'):
      try:
          text = section.get('text', '')
          duration = section.get('duration', 5)
          section_idx = section.get('section_idx', -1)
          section_position = section.get('position', position)
          section_font_size = section.get('font_size', font_size)

          logger.info(f"Processing text section {section_idx}: '{text[:30]}...' duration={duration:.2f}s")

          result = self._create_text_clip(
              text=text,
              duration=duration,
              font_size=section_font_size,
              font_path=font_path,
              position=section_position,
              animation=animation,
              with_pill=with_pill
          )
          
          # Add section index for proper ordering
          if result:
              result._section_idx = section_idx
              result._debug_info = f"Text section {section_idx}"
              logger.info(f"Created text clip for section {section_idx} with duration {result.duration:.2f}s")
          
          return result
      except Exception as e:
          logger.error(f"Error creating text clip for section {section.get('section_idx', -1)}: {e}")
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
      logger.info(f"Creating word-by-word clip: '{text[:30]}...' with duration {duration:.2f}s")
      
      if not font_path:
          font_path = self.body_font_path

      # Handle empty text case
      if not text.strip():
          logger.warning("Empty text provided for word-by-word clip, creating empty clip")
          bg = ColorClip(size=self.resolution, color=(0,0,0,0)).with_duration(duration)
          return bg

      # Split text into words and calculate durations
      words = text.split()
      word_count = len(words)
      char_counts = [len(word) for word in words]
      total_chars = sum(char_counts)
      
      # Calculate timing - adjust for longer words to display longer
      # Minimum word display time increased for better readability
      min_word_time = 0.4  # Minimum time to display each word
      
      # Calculate base duration per character, ensuring minimum display times
      if total_chars > 0:
          # Reserve 15% of duration for transitions
          effective_duration = duration * 0.85
          # Calculate base time per character
          time_per_char = effective_duration / total_chars
          
          # Calculate initial word durations based on character count
          word_durations = []
          for word in words:
              # Base duration on word length, but ensure minimum display time
              word_time = max(min_word_time, len(word) * time_per_char)
              word_durations.append(word_time)
      else:
          # Fallback for empty text
          word_durations = [duration]
      
      # Calculate transition time based on remaining duration
      total_word_duration = sum(word_durations)
      remaining_time = duration - total_word_duration
      
      # Ensure positive transition time
      transition_duration = max(0.15, remaining_time / max(1, word_count - 1)) if word_count > 1 else 0
      
      # Adjust word durations to ensure we fill exactly the requested duration
      adjusted_total = sum(word_durations) + transition_duration * max(0, word_count - 1)
      if abs(adjusted_total - duration) > 0.01 and word_count > 0:
          adjustment_factor = (duration - transition_duration * max(0, word_count - 1)) / sum(word_durations)
          word_durations = [d * adjustment_factor for d in word_durations]
      
      logger.info(f"Word-by-word timing: {word_count} words, transition: {transition_duration:.2f}s")
      for i, (word, word_duration) in enumerate(zip(words, word_durations)):
          logger.info(f"  Word {i+1}: '{word}' - {word_duration:.2f}s")

      clips = []
      for i, (word, word_duration) in enumerate(zip(words, word_durations)):
          # Create a PIL image with the word and pill background
          def create_word_pill():
              # Load font
              try:
                  font = ImageFont.truetype(font_path, font_size)
              except Exception as e:
                  logger.warning(f"Failed to load font: {e}, using default")
                  # Use default font if custom fails
                  font = ImageFont.load_default()

              # Calculate text size
              dummy_img = Image.new('RGBA', (1, 1))
              dummy_draw = ImageDraw.Draw(dummy_img)
              text_bbox = dummy_draw.textbbox((0, 0), word, font=font)
              text_width = text_bbox[2] - text_bbox[0]
              text_height = text_bbox[3] - text_bbox[1]

              # Get font metrics for precise positioning
              ascent, descent = font.getmetrics()

              # Add padding for the pill
              padding_x = int(font_size * 0.7)  # Horizontal padding
              padding_y = int(font_size * 0.35)  # Vertical padding

              # Calculate image dimensions
              img_width = text_width + padding_x * 2
              img_height = text_height + padding_y * 2

              # Create a transparent image
              img = Image.new('RGBA', (img_width, img_height), (0, 0, 0, 0))
              draw = ImageDraw.Draw(img)

              # Create the pill shape with rounded corners
              radius = min(img_height // 2, padding_y + padding_y // 2)  # Ensure radius isn't too large

              # Draw the pill shape
              # Draw the center rectangle
              draw.rectangle([(radius, 0), (img_width - radius, img_height)], fill=pill_color)
              # Draw the left and right edges
              draw.rectangle([(0, radius), (img_width, img_height - radius)], fill=pill_color)
              # Draw the corner circles
              draw.ellipse([(0, 0), (radius * 2, radius * 2)], fill=pill_color)
              draw.ellipse([(img_width - radius * 2, 0), (img_width, radius * 2)], fill=pill_color)
              draw.ellipse([(0, img_height - radius * 2), (radius * 2, img_height)], fill=pill_color)
              draw.ellipse([(img_width - radius * 2, img_height - radius * 2), (img_width, img_height)], fill=pill_color)

              # Calculate text position for perfect centering
              text_x = (img_width - text_width) // 2
              vertical_offset = (descent - ascent) // 4  # Small adjustment for better vertical centering
              text_y = (img_height - text_height) // 2 + vertical_offset

              # Draw the text
              draw.text((text_x, text_y), word, font=font, fill=text_color)

              return img

          # Create the word pill image
          word_image = create_word_pill()

          # Convert to clip with proper duration
          word_clip = ImageClip(np.array(word_image), duration=word_duration)

          # Add to clips list
          clips.append(word_clip)

      # Handle single word case
      if len(clips) == 1:
          word_sequence = clips[0]
      else:
          # Create transitions between words
          try:
              # Method 1: Use concatenate_videoclips with crossfadein
              concatenated_clips = []
              for i, clip in enumerate(clips):
                  # Apply appropriate effects based on position
                  if i > 0:  # Not the first clip
                      clip = clip.with_effects((CrossFadeIn(transition_duration/2)))
                  if i < len(clips) - 1:  # Not the last clip
                      clip = clip.with_effects((CrossFadeOut(transition_duration/2)))
                  concatenated_clips.append(clip)
              
              word_sequence = concatenate_videoclips(concatenated_clips, method="compose")
              logger.info(f"Successfully created word sequence with crossfades, duration: {word_sequence.duration:.2f}s")
          except Exception as e:
              # Fallback Method: Use simple fade in/out effects
              logger.warning(f"Crossfade failed: {e}. Using fallback fade method.")
              concatenated_clips = []
              for i, clip in enumerate(clips):
                  if i > 0:  # Not the first clip
                      clip = clip.with_effects([FadeIn(transition_duration/2)])
                  if i < len(clips) - 1:  # Not the last clip
                      clip = clip.with_effects([FadeOut(transition_duration/2)])
                  concatenated_clips.append(clip)
                  
              word_sequence = concatenate_videoclips(concatenated_clips, method="compose")
              logger.info(f"Created word sequence with fade effects, duration: {word_sequence.duration:.2f}s")

      # Create a transparent background for the entire video
      bg = ColorClip(size=self.resolution, color=(0,0,0,0)).with_duration(word_sequence.duration)

      # Position the word sequence
      positioned_sequence = word_sequence.with_position(position)

      # Combine the background and positioned sequence
      final_clip = CompositeVideoClip([bg, positioned_sequence], size=self.resolution)
      
      # Ensure the clip has exactly the requested duration
      final_clip = final_clip.with_duration(duration)
      
      logger.info(f"Created word-by-word clip with final duration: {final_clip.duration:.2f}s")
      
      return final_clip

  # Also apply the same fix to word_by_word generation
  def _process_word_by_word_section(self, section, font_size=60, font_path=None):
      try:
          text = section.get('text', '')
          duration = section.get('duration', 5)
          section_idx = section.get('section_idx', -1)
          position = section.get('position', ('center', 'center'))
          section_font_size = section.get('font_size', font_size)
          
          logger.info(f"Processing word-by-word section {section_idx}: '{text[:30]}...' duration={duration:.2f}s")

          result = self._create_word_by_word_clip(
              text=text,
              duration=duration,
              font_size=section_font_size,
              font_path=font_path,
              position=position
          )
          
          # Add section index for proper ordering
          if result:
              result._section_idx = section_idx
              result._debug_info = f"Word-by-word section {section_idx}"
              logger.info(f"Created word-by-word clip for section {section_idx} with duration {result.duration:.2f}s")
          
          return result
      except Exception as e:
          logger.error(f"Error creating word-by-word clip for section {section.get('section_idx', -1)}: {e}")
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
          list: List of text clips in the same order as script_sections
      """
      start_time = time.time()
      logger.info(f"Generating {len(script_sections)} word-by-word text clips in parallel")

      if not max_workers:
          max_workers = min(len(script_sections), os.cpu_count())

      # Store results with their section index to ensure correct ordering
      results_with_index = []

      # Use ThreadPoolExecutor for simpler serialization
      with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
          futures = {}
          # Submit tasks with section index
          for i, section in enumerate(script_sections):
              futures[executor.submit(
                  self._process_word_by_word_section,
                  section,
                  font_size=font_size,
                  font_path=font_path
              )] = i

          # Collect results with their indices
          for future in concurrent.futures.as_completed(futures):
              section_idx = futures[future]
              try:
                  clip = future.result()
                  if clip is not None:
                      # Store section index with the clip for proper ordering
                      clip._section_idx = section_idx
                      clip._debug_info = f"Word-by-word clip {section_idx}"
                      results_with_index.append((section_idx, clip))
              except Exception as e:
                  logger.error(f"Error in parallel word-by-word clip generation for section {section_idx}: {e}")

      # Sort the results by section index
      results_with_index.sort(key=lambda x: x[0])

      # Extract just the clips in correct order
      text_clips = [clip for _, clip in results_with_index]

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
