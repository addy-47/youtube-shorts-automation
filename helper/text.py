import numpy as np
from moviepy import *
from moviepy.video.fx import FadeIn
from moviepy.video.fx import FadeOut
from PIL import Image, ImageDraw, ImageFont
import logging
import os
from helper.minor_helper import measure_time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

      # Adjust durations to match total duration
      actual_sum = sum(word_durations) + total_transition_time
      if abs(actual_sum - duration) > 0.01:
          adjust_factor = (duration - total_transition_time) / sum(word_durations)
          word_durations = [d * adjust_factor for d in word_durations]

      clips = []
      current_time = 0

      for i, (word, word_duration) in enumerate(zip(words, word_durations)):
          # Create a function to draw the frame with the word on a pill background
          def make_frame_with_pill(word=word, font_size=font_size, font_path=font_path,
                                  text_color=text_color, pill_color=pill_color):
              # Load font
              font = ImageFont.truetype(font_path, font_size)

              # Calculate text size
              dummy_img = Image.new('RGBA', (1, 1))
              dummy_draw = ImageDraw.Draw(dummy_img)
              text_bbox = dummy_draw.textbbox((0, 0), word, font=font)
              text_width = text_bbox[2] - text_bbox[0]
              text_height = text_bbox[3] - text_bbox[1]

              # Get ascent and descent for more precise vertical positioning
              ascent, descent = font.getmetrics()

              # Add padding for the pill
              padding_x = int(font_size * 0.7)  # Horizontal padding
              padding_y = int(font_size * 0.35)  # Vertical padding

              # Create image
              img_width = text_width + padding_x * 2
              img_height = text_height + padding_y * 2

              # Create a transparent image
              img = Image.new('RGBA', (img_width, img_height), (0, 0, 0, 0))
              draw = ImageDraw.Draw(img)

              # Create the pill shape (rounded rectangle)
              radius = img_height // 2

              # Draw the pill
              # Draw the center rectangle
              draw.rectangle([(radius, 0), (img_width - radius, img_height)], fill=pill_color)
              # Draw the left semicircle
              draw.ellipse([(0, 0), (radius * 2, img_height)], fill=pill_color)
              # Draw the right semicircle
              draw.ellipse([(img_width - radius * 2, 0), (img_width, img_height)], fill=pill_color)

              # For horizontal centering:
              text_x = (img_width - text_width) // 2
              # For vertical centering:
              offwith_y = (descent - ascent) // 4 # This small adjustment often helps
              text_y = (img_height - text_height) // 2 + offwith_y

              draw.text((text_x, text_y), word, font=font, fill=text_color)

              return img

          # Create the frame with the word on a pill
          word_image = make_frame_with_pill()

          # Convert to clip
          word_clip = ImageClip(np.array(word_image), duration=word_duration)

          # Add to clips list
          clips.append(word_clip)

          # Update current time
          current_time += word_duration + transition_duration

      # Concatenate clips
      clips_with_transitions = []
      for i, clip in enumerate(clips):
          if i < len(clips) - 1:  # Not the last clip
              clip = clip.with_effects([FadeIn(transition_duration)])
          clips_with_transitions.append(clip)

      word_sequence = concatenate_videoclips(clips_with_transitions, method="compose")

      # Create a transparent background the size of the entire clip
      bg = ColorClip(size=self.resolution, color=(0,0,0,0)).with_duration(word_sequence.duration)

      # Position the word sequence in the center of the background
      positioned_sequence = word_sequence.with_position(position)

      # Combine the background and positioned sequence
      final_clip = CompositeVideoClip([bg, positioned_sequence], size=self.resolution)

      return final_clip

  @measure_time
  def add_watermark(self, clip, watermark_text="Lazycreator", position=("right", "top"), opacity=0.7, font_size=30):
      """
      Add a watermark to a video clip

      Args:
          clip (VideoClip): Video clip to add watermark to
          watermark_text (str): Text to display as watermark
          position (tuple): Position of watermark ('left'/'right', 'top'/'bottom')
          opacity (float): Opacity of watermark (0-1)
          font_size (int): Font size for watermark

      Returns:
          VideoClip: Clip with watermark added
      """
      # Create text clip for watermark
      watermark = TextClip(
          text=watermark_text,
          font_size=font_size,
          font=self.body_font_path,
          color='white'
      ).with_duration(clip.duration).with_opacity(opacity)

      # Calculate position
      if position[0] == "right":
          x_pos = clip.w - watermark.w - 20
      else:
          x_pos = 20

      if position[1] == "bottom":
          y_pos = clip.h - watermark.h - 20
      else:
          y_pos = 20

      watermark = watermark.with_position((x_pos, y_pos))

      # Add watermark to video
      return CompositeVideoClip([clip, watermark], size=self.resolution)
