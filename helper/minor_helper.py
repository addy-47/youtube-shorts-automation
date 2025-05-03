import re
import logging
from collections import Counter
from pathlib import Path
import nltk
from nltk.corpus import stopwords
import re
import time


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


