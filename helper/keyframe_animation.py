"""
Keyframe-based animation system for YouTube Shorts.

This module provides tools for creating and managing keyframe-based animations
rather than using callable functions, which are difficult to serialize for
parallel processing.
"""

import numpy as np
import logging
from enum import Enum
import json

logger = logging.getLogger(__name__)

class InterpolationType(Enum):
    """Types of interpolation between keyframes"""
    LINEAR = "linear"
    EASE_IN = "ease_in"
    EASE_OUT = "ease_out"
    EASE_IN_OUT = "ease_in_out"
    STEP = "step"

class KeyframeTrack:
    """
    A track of keyframes for a single property.
    
    Stores a collection of keyframes and provides methods for
    interpolation between them.
    """
    
    def __init__(self, property_name, interpolation=InterpolationType.LINEAR):
        """
        Initialize a keyframe track.
        
        Args:
            property_name: Name of the property this track controls
            interpolation: Type of interpolation between keyframes
        """
        self.property_name = property_name
        self.interpolation = interpolation
        self.keyframes = []  # List of (time, value) tuples
        
    def add_keyframe(self, time, value):
        """
        Add a keyframe to the track.
        
        Args:
            time: Time point for the keyframe
            value: Value at this keyframe
        """
        self.keyframes.append((time, value))
        # Sort keyframes by time for easier interpolation
        self.keyframes.sort(key=lambda k: k[0])
        
    def get_value_at_time(self, time):
        """
        Get the interpolated value at a specific time.
        
        Args:
            time: Time to evaluate
            
        Returns:
            Interpolated value at the specified time
        """
        if not self.keyframes:
            logger.warning(f"No keyframes in track {self.property_name}")
            return None
            
        # If time is before first keyframe, return first value
        if time <= self.keyframes[0][0]:
            return self.keyframes[0][1]
            
        # If time is after last keyframe, return last value
        if time >= self.keyframes[-1][0]:
            return self.keyframes[-1][1]
            
        # Find surrounding keyframes
        for i in range(len(self.keyframes) - 1):
            t1, v1 = self.keyframes[i]
            t2, v2 = self.keyframes[i + 1]
            
            if t1 <= time <= t2:
                # Calculate normalized time between keyframes
                if t2 == t1:  # Avoid division by zero
                    alpha = 0
                else:
                    alpha = (time - t1) / (t2 - t1)
                
                # Apply different interpolation types
                if self.interpolation == InterpolationType.LINEAR:
                    return self._interpolate_linear(v1, v2, alpha)
                elif self.interpolation == InterpolationType.EASE_IN:
                    alpha = alpha * alpha  # Squared for ease-in
                    return self._interpolate_linear(v1, v2, alpha)
                elif self.interpolation == InterpolationType.EASE_OUT:
                    alpha = 1 - (1 - alpha) * (1 - alpha)  # Inverted square for ease-out
                    return self._interpolate_linear(v1, v2, alpha)
                elif self.interpolation == InterpolationType.EASE_IN_OUT:
                    if alpha < 0.5:
                        alpha = 2 * alpha * alpha  # First half: ease-in
                    else:
                        alpha = 1 - 2 * (1 - alpha) * (1 - alpha)  # Second half: ease-out
                    return self._interpolate_linear(v1, v2, alpha)
                elif self.interpolation == InterpolationType.STEP:
                    return v1  # Step function: no interpolation
                else:
                    # Default to linear
                    return self._interpolate_linear(v1, v2, alpha)
                    
        # Fallback to last value (shouldn't reach here given earlier checks)
        return self.keyframes[-1][1]
        
    def _interpolate_linear(self, v1, v2, alpha):
        """
        Perform linear interpolation between two values.
        
        Args:
            v1: First value
            v2: Second value
            alpha: Interpolation factor (0-1)
            
        Returns:
            Interpolated value
        """
        # Handle different value types
        if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
            return v1 + alpha * (v2 - v1)
        elif isinstance(v1, (list, tuple)) and isinstance(v2, (list, tuple)) and len(v1) == len(v2):
            return tuple(v1[i] + alpha * (v2[i] - v1[i]) for i in range(len(v1)))
        elif isinstance(v1, np.ndarray) and isinstance(v2, np.ndarray) and v1.shape == v2.shape:
            return v1 + alpha * (v2 - v1)
        else:
            # For other types, just return nearest value
            return v1 if alpha < 0.5 else v2
    
    def to_callable(self):
        """
        Convert this keyframe track to a callable function.
        
        Returns:
            A function that takes a time argument and returns
            the interpolated value at that time.
        """
        def keyframe_func(t):
            return self.get_value_at_time(t)
        
        return keyframe_func
    
    def to_dict(self):
        """Convert to serializable dictionary"""
        return {
            "property_name": self.property_name,
            "interpolation": self.interpolation.value if isinstance(self.interpolation, InterpolationType) else self.interpolation,
            "keyframes": [(float(t), v) for t, v in self.keyframes]
        }
    
    @classmethod
    def from_dict(cls, data):
        """Create from dictionary"""
        track = cls(data["property_name"])
        
        # Set interpolation
        if isinstance(data["interpolation"], str):
            try:
                track.interpolation = InterpolationType(data["interpolation"])
            except (ValueError, KeyError):
                track.interpolation = InterpolationType.LINEAR
        else:
            track.interpolation = InterpolationType.LINEAR
            
        # Add keyframes
        for t, v in data["keyframes"]:
            track.add_keyframe(float(t), v)
            
        return track
    
    @classmethod
    def from_callable(cls, func, property_name, duration, num_samples=20):
        """
        Create a keyframe track by sampling a callable function.
        
        Args:
            func: Callable function that takes a time value
            property_name: Name of the property this track controls
            duration: Duration to sample over
            num_samples: Number of keyframes to generate
            
        Returns:
            KeyframeTrack instance
        """
        track = cls(property_name)
        
        for i in range(num_samples):
            t = i * duration / (num_samples - 1) if num_samples > 1 else 0
            try:
                v = func(t)
                track.add_keyframe(t, v)
            except Exception as e:
                logger.warning(f"Error sampling function at time {t}: {e}")
                # Use default values based on common property types
                if "position" in property_name:
                    v = (0, 0)
                elif "size" in property_name:
                    v = (100, 100)
                elif "rotation" in property_name:
                    v = 0
                elif "opacity" in property_name:
                    v = 1.0
                else:
                    v = 0
                track.add_keyframe(t, v)
                
        return track

class Animation:
    """
    Container for multiple keyframe tracks making up a complete animation.
    
    This class can be used to create complex animations with multiple
    properties changing simultaneously.
    """
    
    def __init__(self, duration):
        """
        Initialize an animation.
        
        Args:
            duration: Duration of the animation in seconds
        """
        self.duration = duration
        self.tracks = {}  # Dictionary mapping property names to KeyframeTracks
        
    def add_track(self, track):
        """
        Add a keyframe track to the animation.
        
        Args:
            track: KeyframeTrack instance
        """
        self.tracks[track.property_name] = track
        
    def create_track(self, property_name, interpolation=InterpolationType.LINEAR):
        """
        Create and add a new keyframe track.
        
        Args:
            property_name: Name of the property this track controls
            interpolation: Type of interpolation between keyframes
            
        Returns:
            The created KeyframeTrack
        """
        track = KeyframeTrack(property_name, interpolation)
        self.add_track(track)
        return track
    
    def get_value(self, property_name, time):
        """
        Get the value of a property at a specific time.
        
        Args:
            property_name: Name of the property
            time: Time to evaluate
            
        Returns:
            Interpolated value at the specified time
        """
        if property_name in self.tracks:
            return self.tracks[property_name].get_value_at_time(time)
        else:
            logger.warning(f"No track found for property: {property_name}")
            return None
    
    def get_callable(self, property_name):
        """
        Get a callable function for a property.
        
        Args:
            property_name: Name of the property
            
        Returns:
            A function that takes a time argument and returns
            the interpolated value at that time.
        """
        if property_name in self.tracks:
            return self.tracks[property_name].to_callable()
        else:
            logger.warning(f"No track found for property: {property_name}")
            # Return a default function that always returns None
            return lambda t: None
    
    def to_dict(self):
        """Convert to serializable dictionary"""
        return {
            "duration": self.duration,
            "tracks": {name: track.to_dict() for name, track in self.tracks.items()}
        }
    
    def to_json(self):
        """Convert to JSON string"""
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_dict(cls, data):
        """Create from dictionary"""
        animation = cls(data["duration"])
        
        for name, track_data in data["tracks"].items():
            track = KeyframeTrack.from_dict(track_data)
            animation.add_track(track)
            
        return animation
    
    @classmethod
    def from_json(cls, json_str):
        """Create from JSON string"""
        data = json.loads(json_str)
        return cls.from_dict(data)

def create_position_animation(start_pos, end_pos, duration, interpolation=InterpolationType.EASE_IN_OUT):
    """
    Create a simple position animation between two points.
    
    Args:
        start_pos: Starting position (x, y)
        end_pos: Ending position (x, y)
        duration: Duration of the animation in seconds
        interpolation: Type of interpolation
        
    Returns:
        KeyframeTrack for the position animation
    """
    track = KeyframeTrack("position", interpolation)
    track.add_keyframe(0, start_pos)
    track.add_keyframe(duration, end_pos)
    return track

def create_size_animation(start_size, end_size, duration, interpolation=InterpolationType.EASE_IN_OUT):
    """
    Create a simple size animation between two sizes.
    
    Args:
        start_size: Starting size (width, height)
        end_size: Ending size (width, height)
        duration: Duration of the animation in seconds
        interpolation: Type of interpolation
        
    Returns:
        KeyframeTrack for the size animation
    """
    track = KeyframeTrack("size", interpolation)
    track.add_keyframe(0, start_size)
    track.add_keyframe(duration, end_size)
    return track

def create_fade_animation(duration, fade_in=0.5, fade_out=0.5):
    """
    Create a fade in/out opacity animation.
    
    Args:
        duration: Duration of the animation in seconds
        fade_in: Duration of the fade in in seconds
        fade_out: Duration of the fade out in seconds
        
    Returns:
        KeyframeTrack for the opacity animation
    """
    track = KeyframeTrack("opacity", InterpolationType.LINEAR)
    
    # Fade in
    if fade_in > 0:
        track.add_keyframe(0, 0)
        track.add_keyframe(fade_in, 1)
    else:
        track.add_keyframe(0, 1)
    
    # Hold in middle if needed
    if duration > fade_in + fade_out:
        track.add_keyframe(duration - fade_out, 1)
    
    # Fade out
    if fade_out > 0:
        track.add_keyframe(duration, 0)
        
    return track

def convert_callable_to_keyframes(func, property_name, duration, num_samples=20):
    """
    Convert a callable function to a keyframe track.
    
    Args:
        func: Callable function that takes a time value
        property_name: Name of the property this track controls
        duration: Duration to sample over
        num_samples: Number of keyframes to generate
        
    Returns:
        KeyframeTrack instance
    """
    return KeyframeTrack.from_callable(func, property_name, duration, num_samples)

def apply_keyframes_to_clip(clip, property_name, keyframe_track):
    """
    Apply a keyframe track to a clip.
    
    This function adds keyframe animation to a clip by setting the
    appropriate attribute of the clip to a callable function.
    
    Args:
        clip: MoviePy clip to apply animation to
        property_name: Name of the property to animate
        keyframe_track: KeyframeTrack instance
        
    Returns:
        The modified clip
    """
    if property_name == "position":
        return clip.set_position(keyframe_track.to_callable())
    elif property_name == "size":
        return clip.resize(keyframe_track.to_callable())
    elif property_name == "opacity":
        opacity_func = keyframe_track.to_callable()
        opacity_transform = lambda t, frame: frame * opacity_func(t)
        return clip.transform(opacity_transform)
    else:
        logger.warning(f"Property {property_name} not supported for direct application")
        return clip 