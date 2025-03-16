import logging # for logging events
import os # for environment variables and file paths
from pathlib import Path # for file paths and directory creation
from dotenv import load_dotenv # for loading environment variables
from script_generator import generate_script
from video_maker import YTShortsCreator
from youtube_upload import upload_video, get_authenticated_service
from nltk.corpus import stopwords
import datetime # for timestamp
import re # for regular expressions
import nltk # for natural language processing
from collections import Counter # for counting elements in a list
import requests # for making HTTP requests

load_dotenv()
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
YOUTUBE_TOPIC = os.getenv("YOUTUBE_TOPIC", "Artificial Intelligence")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('youtube_shorts.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def ensure_output_directory(directory="ai_shorts_output"):
    """Ensure the output directory exists."""
    Path(directory).mkdir(parents=True, exist_ok=True)
    return directory

def get_latest_ai_news():
    if not NEWS_API_KEY:
        raise ValueError("NewsAPI key is missing. Set NEWS_API_KEY in .env.")
    url = f"https://newsapi.org/v2/top-headlines?q=artificial intelligence&apiKey={NEWS_API_KEY}"
    response = requests.get(url)
    if response.status_code == 200:
        articles = response.json().get('articles', [])
        if articles:
            return articles[0]['title']  # Return the latest article’s title
    return "No recent AI news available."


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

def get_keywords(script, max_keywords=5):
    """
    Extract the top most relevant keywords from the script.

    Args:
        script (str): The script text.
        max_keywords (int): Maximum number of keywords to return.

    Returns:
        list: A list of the top keywords.
    """
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


def clean_script_text(text):
    """Clean the script text to remove instructional labels for TTS."""
    text = text.strip()

    # Remove labels like "1. Hook (0–3 seconds):" or "hook 3 sec"
    text = re.sub(r'\d*\.?\s*\w+\s*\(?\d*–?\d*\s*sec(onds)?\)?\s*:?\s*', '', text)

    # Additional cleanup
    text = re.sub(r'\n\s*\n', '\n', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'[^\w\s.,!?\'\"]', '', text)
    return text.strip()

def generate_youtube_short(topic, style="video", max_duration=25):
    """
    Generate a YouTube Short.

    Args:
        topic (str): Topic for the YouTube Short
        style (str): Type of background ("video" or "animation")
        max_duration (int): Maximum video duration in seconds
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
        prompt = f"""
        Generate a YouTube Shorts script focused entirely on the latest AI news: '{latest_ai_news}
        for the date {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.
        The script should not exceed 25 secs and should follow this structure:
        1. Start with an attention-grabbing opening (0–3 seconds).
        2. Highlight 1–2 key points about this specific AI news (4–22 seconds).
        3. End with a clear call to action (23–25 seconds).
        Use short, concise sentences, include 2–3 relevant emojis (e.g., 🤖, 🚀), and suggest 3-4 trending hashtags (e.g., #AI, #TechNews).
        Keep it under {max_tokens} tokens.
        """

        script = generate_script(prompt, max_tokens=max_tokens)
        logger.info("Raw script generated successfully")

        # Clean the script
        cleaned_script = clean_script_text(script)
        logger.info("Script cleaned")

        script_cards = parse_script_to_cards(cleaned_script)
        logger.info(f"Script parsed into {len(script_cards)} sections")
        for i, card in enumerate(script_cards):
            logger.info(f"Section {i+1}: {card['text'][:30]}... (duration: {card['duration']}s)")

        # Extract keywords
        keywords = get_keywords(cleaned_script)
        logger.info(f"Extracted keywords: {keywords}")

        # Video Creation
        logger.info("Creating YouTube Short")
        creator = YTShortsCreator(output_dir=output_dir)
        video_path = creator.create_youtube_short(
            title=topic,
            script_sections=script_cards,
            background_query=keywords,
            output_filename=output_path,
            style=style,
            voice_style = "none",
            max_duration=max_duration
            )

        # Optional: YouTube Upload
        if os.getenv("ENABLE_YOUTUBE_UPLOAD", "false").lower() == "true":
            logger.info("Uploading to YouTube")
            youtube = get_authenticated_service()
            upload_video(
                youtube,
                video_path,
                f"AI Short: {topic}",
                f"Explore {topic} in this quick AI-generated Short! {' '.join(['#' + kw for kw in keywords[:5]])}",
                ["shorts", "ai", "technology"]
            )

        return video_path

    except Exception as e:
        logger.error(f"Error generating YouTube Short: {e}")
        raise

def main():
    try:
        topic = os.getenv("YOUTUBE_TOPIC", "Artificial Intelligence")
        style = os.getenv("BACKGROUND_STYLE", "video")
        video_path = generate_youtube_short(topic, style=style,max_duration=25)
        logger.info(f"YouTube Short created successfully: {video_path}")
    except Exception as e:
        logger.error(f"Process failed: {e}")

if __name__ == "__main__":
    main()
