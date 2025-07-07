import re
import logging
from collections import Counter
from pathlib import Path
import nltk
from nltk.corpus import stopwords
import re
import time
import os
import shutil
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get temp directory from environment variable or use default
TEMP_DIR = os.getenv("TEMP_DIR", os.path.join(os.path.dirname(os.path.dirname(__file__)), "temp"))

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Timer function for performance monitoring
def measure_time(func):
    """Decorator to measure the execution time of functions"""
    def wrapper(*args, **kwargs):
        # Only log timing for major functions (create_youtube_short)
        if func.__name__ == "create_youtube_short":
            start_time = time.time()
            logger.info(f"Starting YouTube short creation")
            result = func(*args, **kwargs)
            end_time = time.time()
            duration = end_time - start_time
            logger.info(f"Completed YouTube short creation in {duration:.2f} seconds")
        else:
            # For all other functions, just run without detailed logging
            result = func(*args, **kwargs)
        return result
    return wrapper

def cleanup_temp_directories(max_age_hours=24, specific_dir=None, force_all=False):
    """
    Clean up temporary directories in the main TEMP_DIR or a specific temp directory

    Args:
        max_age_hours (int): Maximum age in hours for directories to keep
        specific_dir (str): Specific directory to clean (None for all temp dirs)
        force_all (bool): If True, remove all temp directories regardless of age

    Returns:
        bool: True if cleanup was successful, False otherwise
    """
    # Create the main temp directory if it doesn't exist
    if not os.path.exists(TEMP_DIR):
        logger.info(f"Main temp directory {TEMP_DIR} does not exist, creating it")
        os.makedirs(TEMP_DIR, exist_ok=True)
        return True

    # If cleaning a specific directory
    if specific_dir:
        logger.info(f"Cleaning specific temporary directory: {specific_dir}")
        if not os.path.exists(specific_dir):
            logger.info(f"Specified directory {specific_dir} does not exist, nothing to clean")
            return True

        try:
            # Try to remove everything as a directory tree first
            try:
                shutil.rmtree(specific_dir)
                logger.info(f"Successfully removed directory tree: {specific_dir}")
                return True
            except Exception as tree_error:
                logger.warning(f"Could not remove entire directory tree: {tree_error}")

            # Fallback: try to remove files one by one
            for root, dirs, files in os.walk(specific_dir, topdown=False):
                for file in files:
                    try:
                        file_path = os.path.join(root, file)
                        if os.path.exists(file_path):
                            os.unlink(file_path)
                            logger.debug(f"Removed file: {file_path}")
                    except Exception as e:
                        logger.warning(f"Failed to remove file {file}: {e}")

                # Try to remove empty directories
                for dir in dirs:
                    try:
                        dir_path = os.path.join(root, dir)
                        if os.path.exists(dir_path) and not os.listdir(dir_path):
                            os.rmdir(dir_path)
                            logger.debug(f"Removed directory: {dir_path}")
                    except Exception as e:
                        logger.warning(f"Failed to remove directory {dir}: {e}")

            # Finally try to remove the specified directory itself
            try:
                if os.path.exists(specific_dir) and not os.listdir(specific_dir):
                    os.rmdir(specific_dir)
                    logger.info(f"Removed empty directory: {specific_dir}")
            except Exception as e:
                logger.warning(f"Failed to remove directory {specific_dir}: {e}")

            return True
        except Exception as e:
            logger.error(f"Error during cleanup of {specific_dir}: {e}")
            import traceback
            logger.debug(f"Cleanup error details: {traceback.format_exc()}")
            return False

    # Clean all temporary directories in TEMP_DIR
    logger.info(f"Cleaning up temporary directories in {TEMP_DIR}")
    current_time = time.time()
    max_age_seconds = max_age_hours * 3600
    success = True

    try:
        # Check all immediate subdirectories in the main TEMP_DIR
        for item in os.listdir(TEMP_DIR):
            item_path = os.path.join(TEMP_DIR, item)

            # Only process directories
            if os.path.isdir(item_path):
                # Check if this is a temp directory we should clean
                is_temp_dir = any(prefix in item for prefix in ["shorts_", "thumbnail_", "temp_", "parallel_render"])

                # Skip directories that don't match our temp directory patterns
                # unless force_all is True
                if not force_all and not is_temp_dir:
                    continue

                # Check directory age if we're not forcing removal of all
                if not force_all:
                    try:
                        dir_mtime = os.path.getmtime(item_path)
                        dir_age = current_time - dir_mtime

                        # Skip directories newer than max_age_hours
                        if dir_age <= max_age_seconds:
                            logger.debug(f"Skipping directory {item_path} - age: {dir_age/3600:.1f} hours")
                            continue
                    except Exception as e:
                        logger.warning(f"Error checking age of directory {item_path}: {e}")

                # Remove the directory
                logger.info(f"Removing temporary directory: {item_path}")
                try:
                    shutil.rmtree(item_path)
                except Exception as e:
                    logger.warning(f"Failed to remove directory {item_path}: {e}")
                    success = False
    except Exception as e:
        logger.error(f"Error during temp directory cleanup: {e}")
        import traceback
        logger.debug(f"Cleanup error details: {traceback.format_exc()}")
        success = False

    # Make sure required subdirectories exist
    os.makedirs(os.path.join(TEMP_DIR, "video_downloads"), exist_ok=True)
    os.makedirs(os.path.join(TEMP_DIR, "generated_images"), exist_ok=True)

    return success

def ensure_output_directory(directory="ai_shorts_output"):
    """Ensure the output directory exists."""
    Path(directory).mkdir(parents=True, exist_ok=True)
    return directory

def parse_script_to_cards(script):
    """Parse the raw script into a list of cards with text and duration."""
    cards = []
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!)\s', script)

    logger.info(f"Parsed script into {len(sentences)} sentences")

    for i, sentence in enumerate(sentences):
        if not sentence:
            logger.debug(f"Skipping empty sentence at position {i}")
            continue
        duration = 5 if len(sentence) > 30 else 3
        voice_style = "excited" if i == 0 or i == len(sentences) - 1 else "normal"
        cards.append({"text": sentence, "duration": duration, "voice_style": voice_style})
        logger.info(f"Added sentence {i} to cards: '{sentence[:30]}...' (duration: {duration}s)")

    logger.info(f"Created {len(cards)} script cards")
    return cards

def get_keywords(script, max_keywords=3):
    """Extract keywords from text using NLTK (Now potentially unused)."""
    # Ensure NLTK resources are downloaded
    nltk.download('stopwords', quiet=True) #quiet=True to suppress output
    nltk.download('punkt', quiet=True)

    stop_words = set(stopwords.words('english'))

    # Extract words from script, ignoring stopwords
    words = re.findall(r'\b\w+\b', script.lower())
    filtered_words = [word for word in words if word not in stop_words and len(word) > 3]

    # Count word frequency
    word_counts = Counter(filtered_words)

    # Get the most common words
    top_keywords = [word for word, count in word_counts.most_common(max_keywords)]

    return top_keywords


