import logging # for logging events
import logging.handlers # Import handlers
import os # for environment variables and file paths
import sys # for stdout encoding
from pathlib import Path # for file paths and directory creation
from dotenv import load_dotenv # for loading environment variables
from script_generator import generate_script, generate_batch_video_queries,generate_batch_image_prompts
from shorts_maker_V import YTShortsCreator_V
from shorts_maker_I import YTShortsCreator_I
from youtube_upload import upload_video, get_authenticated_service
from nltk.corpus import stopwords
import datetime # for timestamp
import re # for regular expressions
import nltk # for natural language processing
from collections import Counter # for counting elements in a list
import requests # for making HTTP requests
import random # for generating random numbers

load_dotenv()
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
YOUTUBE_TOPIC = os.getenv("YOUTUBE_TOPIC", "Artificial Intelligence")

# Configure logging with daily rotation
LOG_DIR = 'logs'  # Define log directory
LOG_FILENAME = os.path.join(LOG_DIR, 'youtube_shorts_daily.log') # Create full path
LOG_LEVEL = logging.INFO

# Ensure log directory exists
Path(LOG_DIR).mkdir(parents=True, exist_ok=True)

# Set up a specific logger with our desired output level
logger = logging.getLogger(__name__)
logger.setLevel(LOG_LEVEL)

# Define log format - simpler format without emojis to avoid encoding issues
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')

# Force stdout to use utf-8 encoding (helps with emoji on Windows)
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# Remove any existing handlers to avoid duplicates
for handler in logger.handlers[:]:
    logger.removeHandler(handler)

# Add the log message handler to the logger
# Rotate logs daily at midnight, keep 7 backups
handler = logging.handlers.TimedRotatingFileHandler(
    LOG_FILENAME, when='midnight', interval=1, backupCount=7,
    encoding='utf-8'  # Force UTF-8 encoding for log files
)
handler.setFormatter(formatter)
logger.addHandler(handler)

# Add a handler to also output to console (like the original setup)
stream_handler = logging.StreamHandler(sys.stdout)  # Use explicit stdout with proper encoding
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

# Configure root logger - first remove any existing handlers
root_logger = logging.getLogger()
for handler in root_logger.handlers[:]:
    root_logger.removeHandler(handler)

# Configure root logger similarly if other modules use logging.getLogger() without a name
# This ensures consistency if other modules just call logging.info etc.
root_logger.setLevel(LOG_LEVEL)
root_handler = logging.handlers.TimedRotatingFileHandler(
    LOG_FILENAME, when='midnight', interval=1, backupCount=7,
    encoding='utf-8'  # Force UTF-8 encoding for log files
)
root_handler.setFormatter(formatter)
root_logger.addHandler(root_handler)

# Add console output for root logger too
root_stream_handler = logging.StreamHandler(sys.stdout)  # Use explicit stdout with proper encoding
root_stream_handler.setFormatter(formatter)
root_logger.addHandler(root_stream_handler)

def ensure_output_directory(directory="ai_shorts_output"):
    """Ensure the output directory exists."""
    Path(directory).mkdir(parents=True, exist_ok=True)
    return directory

def get_latest_ai_news():
    """Get the latest technology or AI news."""
    if not NEWS_API_KEY:
        raise ValueError("NewsAPI key is missing. Set NEWS_API_KEY in .env.")

    # Calculate the date two weeks ago
    two_weeks_ago = datetime.datetime.now() - datetime.timedelta(weeks=2)
    # Format the date as YYYY-MM-DD for the News API
    from_date = two_weeks_ago.strftime("%Y-%m-%d")

    # Specify technology and AI focus with multiple topics
    topics = ["artificial intelligence", "technology", "tech innovation", "AI", "machine learning"]

    # Try each topic until we find a suitable article
    for topic in topics:
        url = f"https://newsapi.org/v2/top-headlines?q={topic}&category=technology&from={from_date}&sortBy=popularity&pageSize=10&apiKey={NEWS_API_KEY}"
        response = requests.get(url)

        if response.status_code == 200:
            articles = response.json().get('articles', [])
            if articles:
                chosen_article = random.choice(articles) # Choose a random article from top 10
                return chosen_article['title']


    # Fallback to a general technology search if no AI-specific news
    url = f"https://newsapi.org/v2/top-headlines?category=technology&apiKey={NEWS_API_KEY}"
    response = requests.get(url)

    if response.status_code == 200:
        articles = response.json().get('articles', [])
        if articles:
            chosen_article = random.choice(articles)
            return chosen_article['title']

    return "Latest Technology Innovation News"


def parse_script_to_cards(script):
    """Parse the raw script into a list of cards with text and duration."""
    cards = []
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!)\s', script)
    for i, sentence in enumerate(sentences):
        if not sentence:
            continue
        duration = 5 if len(sentence) > 30 else 3
        voice_style = "excited" if i == 0 or i == len(sentences) - 1 else "normal"
        cards.append({"text": sentence, "duration": duration, "voice_style": voice_style})
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

def get_creator_for_day():
    """Alternate between video and image creators based on day"""
    today = datetime.datetime.now()
    day_of_year = today.timetuple().tm_yday  # 1-366
    use_images = day_of_year % 2 == 0  # Even days use images, odd days use videos

    if use_images:
        logger.info(f"Day {day_of_year}: Using image-based creator (YTShortsCreator_I)")
        return YTShortsCreator_I()
    else:
        logger.info(f"Day {day_of_year}: Using video-based creator (YTShortsCreator_V)")
        return YTShortsCreator_V()

def generate_youtube_short(topic, style="photorealistic", max_duration=25, creator_type=None):
    """
    Generate a YouTube Short.

    Args:
        topic (str): Topic for the YouTube Short
        style (str): Style for the content ("photorealistic", "digital art", etc.)
        max_duration (int): Maximum video duration in seconds
        creator_type: Optional creator instance to use (if None, will create a new one)
    """
    try:
        output_dir = ensure_output_directory()

        # Generate unique filename with timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"yt_shorts_{topic.replace(' ', '_')}_{timestamp}.mp4"
        output_path = os.path.join(output_dir, output_filename)

        # Script Generation
        latest_ai_news = get_latest_ai_news()
        logger.info(f"Generating script for topic: {latest_ai_news}")
        max_tokens = 200
        # user_input = input("Prompt for the user: ")
        # topic = user_input,
        prompt = f"""
        Generate a YouTube Shorts script focused entirely on the latest AI news: '{latest_ai_news}
        for the date {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.
        The script should not exceed 25 secs and should follow this structure:
        1. Start with an attention-grabbing opening (0–3 seconds).
        2. Highlight 1–2 key points about this specific AI news (4–22 seconds).
        3. End with a clear call to action (23–25 seconds).
        Use short, concise sentences and suggest 3-4 trending hashtags (e.g., #AI, #TechNews).
        Keep it under {max_tokens} tokens.
        """
        # prompt = user_input

        script = generate_script(prompt, max_tokens=max_tokens)
        logger.info("Raw script generated successfully")

        script_cards = parse_script_to_cards(script)
        logger.info(f"Script parsed into {len(script_cards)} sections")
        for i, card in enumerate(script_cards):
            logger.info(f"Section {i+1}: {card['text'][:30]}... (duration: {card['duration']}s)")

        # Get or create creator type
        if creator_type is None:
            creator_type = get_creator_for_day()

        # Generate section-specific queries using the LLM in a single batch call
        card_texts = [card['text'] for card in script_cards]
        if isinstance(creator_type, YTShortsCreator_V):
           logger.info("Generating video search queries for each section using AI...")
           batch_query_results = generate_batch_video_queries(card_texts, overall_topic=topic, model="gpt-4o-mini-2024-07-18")
        else:
           logger.info("Generating image search prompts for each section using AI...")
           batch_query_results = generate_batch_image_prompts(card_texts, overall_topic=topic, model="gpt-4o-mini-2024-07-18")

        # Extract queries in order, using a fallback if needed
        default_query = f"abstract {topic}"

        section_queries = []
        for i in range(len(script_cards)):
            query = batch_query_results.get(i, default_query) # Get query by index, fallback to default
            if not query: # Ensure query is not empty string
                 query = default_query
                 logger.warning(f"Query for section {i} was empty, using fallback: '{default_query}'")
            section_queries.append(query)
            logger.info(f"Section {i+1} query: {query}")

        # Generate a fallback query for the whole script if needed
        fallback_query = section_queries[0] if section_queries else default_query

        # Video Creation - always pass through the style parameter
        logger.info(f"Creating YouTube Short with style: {style}")
        video_path = creator_type.create_youtube_short(
            title=topic,
            script_sections=script_cards,
            background_query=fallback_query,
            output_filename=output_path,
            style=style,  # Always use the style parameter passed to this function
            voice_style="none",
            max_duration=max_duration,
            background_queries=section_queries,
            blur_background=False,
            edge_blur=True
        )

        # Optional: YouTube Upload
        if os.getenv("ENABLE_YOUTUBE_UPLOAD", "false").lower() == "true":
            logger.info("Uploading to YouTube")
            youtube = get_authenticated_service()
            upload_video(
                youtube,
                video_path,
                f"AI Short: {topic}",
                f"Explore {topic} in this quick AI-generated Short!", # Removed keywords from description for now
                ["shorts", "ai", "technology"]
            )

        return video_path

    except Exception as e:
        logger.error(f"Error generating YouTube Short: {e}")
        raise

def main():
    try:
        topic = os.getenv("YOUTUBE_TOPIC", "Artificial Intelligence")

        # Get creator type - this will already log the creator type
        creator = get_creator_for_day()

        # We'll always use "photorealistic" as the default style unless overridden at function call
        style = "photorealistic"
        logger.info(f"Using style: {style}")

        try:
            video_path = generate_youtube_short(
                topic,
                style=style,
                max_duration=25,
                creator_type=creator  # Pass the creator we already initialized
            )
            logger.info(f"YouTube Short created successfully: {video_path}")
        except Exception as e:
            logger.error(f"Error generating YouTube Short: {str(e)}")
            import traceback
            logger.error(f"Detailed error: {traceback.format_exc()}")
            raise
    except Exception as e:
        logger.error(f"Process failed: {str(e)}")
        import traceback
        logger.error(f"Detailed error trace: {traceback.format_exc()}")

if __name__ == "__main__":
    main()
