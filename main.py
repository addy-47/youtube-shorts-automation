import os # for environment variables and file paths
import sys # for stdout encoding
import logging # for logging events
import logging.handlers # Import handlers
from pathlib import Path # for file paths and directory creation
from dotenv import load_dotenv # for loading environment variables
import datetime # for timestamp
from automation.content_generator import generate_batch_video_queries, generate_batch_image_prompts, generate_comprehensive_content
from automation.shorts_maker_V import YTShortsCreator_V
from automation.shorts_maker_I import YTShortsCreator_I
from automation.youtube_upload import upload_video, get_authenticated_service
from automation.thumbnail import ThumbnailGenerator
from helper.news import get_latest_news
from helper.minor_helper import ensure_output_directory, parse_script_to_cards

load_dotenv()
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
YOUTUBE_TOPIC = os.getenv("YOUTUBE_TOPIC", "Artificial Intelligence")

# Configure logging with daily rotation
LOG_DIR = 'logs'  # Define log directory
LOG_FILENAME = os.path.join(LOG_DIR, 'youtube_shorts_daily.log') # Create full path
LOG_LEVEL = logging.INFO

# Add a debug flag to enable more verbose logging when needed
DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"
if DEBUG_MODE:
    LOG_LEVEL = logging.DEBUG
    print("DEBUG MODE ENABLED: More verbose logging activated")

# Ensure log directory exists
Path(LOG_DIR).mkdir(parents=True, exist_ok=True)

# First, disable any existing loggers to avoid duplicate outputs
logging.getLogger().handlers = []

# Configure a single root logger
root_logger = logging.getLogger()
root_logger.setLevel(LOG_LEVEL)
# Suppress MoviePy logs to avoid excessive output
logging.getLogger('moviepy').setLevel(logging.ERROR)
logging.getLogger('imageio').setLevel(logging.ERROR)
logging.getLogger('imageio_ffmpeg').setLevel(logging.ERROR)
# Also suppress PIL warnings (which are common with MoviePy 2.1.2)
logging.getLogger('PIL').setLevel(logging.ERROR)

# Define log format - simpler format without emojis to avoid encoding issues
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')

# Add file handler with rotation
file_handler = logging.handlers.TimedRotatingFileHandler(
    LOG_FILENAME, when='midnight', interval=1, backupCount=7,
    encoding='utf-8'  # Force UTF-8 encoding for log files
)
file_handler.setFormatter(formatter)
root_logger.addHandler(file_handler)

# Add console handler
console_handler = logging.StreamHandler(sys.stdout)  # Use explicit stdout with proper encoding
console_handler.setFormatter(formatter)
root_logger.addHandler(console_handler)

# Use the root logger for this module
logger = logging.getLogger(__name__)

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

    Returns:
        tuple: (video_path, thumbnail_path)
    """
    try:
        output_dir = ensure_output_directory()

        # Generate unique filename with timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        if topic is None:
            topic = get_latest_news()
        logger.info(f"Generating comprehensive content for : {topic}")

        # Generate all content in a single API call
        content_package = generate_comprehensive_content(topic, max_tokens=800)

        # Extract content elements
        script = content_package["script"]
        title = f"LazyCreator presents: {content_package['title']}"
        description = content_package["description"]
        thumbnail_hf_prompt = content_package["thumbnail_hf_prompt"]
        thumbnail_unsplash_query = content_package["thumbnail_unsplash_query"]

        logger.info("Content package generated successfully:")
        logger.info(f"Title: {title}")
        logger.info(f"Description length: {len(description)} characters")
        logger.info("Raw script generated successfully")

        # Create output filename using the title instead of raw topic
        safe_title = title.replace(' ', '_').replace(':', '').replace('?', '').replace('!', '')[:30]
        output_filename = f"yt_shorts_{safe_title}_{timestamp}.mp4"
        output_path = os.path.join(output_dir, output_filename)

        # Parse script into cards as before
        script_cards = parse_script_to_cards(script)
        logger.info(f"Script parsed into {len(script_cards)} sections")
        for i, card in enumerate(script_cards):
            logger.info(f"Section {i+1}: {card['text'][:30]}... (duration: {card['duration']}s)")

        # Add intro card to display the title at the beginning
        intro_card = {
            "text": f"LazyCreator presents: {content_package['title']}",
            "duration": 3,
            "voice_style": "excited"
        }
        script_cards.insert(0, intro_card)
        logger.info(f"Added intro card with title: '{intro_card['text']}'")

        # Log all sections after insertion to confirm proper order
        logger.info("=== FINAL SCRIPT CARDS ORDER ===")
        for i, card in enumerate(script_cards):
            logger.info(f"Section {i}: '{card['text'][:30]}...' (duration: {card['duration']}s)")
        logger.info("=== END SCRIPT CARDS ORDER ===")

        if creator_type is None:
            creator_type = get_creator_for_day()

        # Generate section-specific queries based on creator type
        card_texts = [card['text'] for card in script_cards]

        # We still need to generate section-specific queries for each section
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

        # Log all section queries at once to avoid duplication
        logger.info(f"Section queries: {', '.join([f'{i+1}: {q}' for i, q in enumerate(section_queries)])}")

        # Generate a fallback query for the whole script if needed
        fallback_query = section_queries[0] if section_queries else default_query

        # Video Creation - only log style for image-based creators
        if isinstance(creator_type, YTShortsCreator_I):
            logger.info(f"Creating YouTube Short with style: {style}")
        else:
            logger.info(f"Creating YouTube Short")

        video_path = creator_type.create_youtube_short(
            title=title,  # Use the generated title
            script_sections=script_cards,
            background_query=fallback_query,
            output_filename=output_path,
            style=style,
            voice_style="none",
            max_duration=max_duration,
            background_queries=section_queries,
            blur_background=False,
            edge_blur=False
        )

        # Generate thumbnail for the short
        thumbnail_path = None
        try:
            logger.info("Generating thumbnail for the short")
            thumbnail_dir = os.path.join(output_dir, "thumbnails")
            os.makedirs(thumbnail_dir, exist_ok=True)

            # Initialize thumbnail generator
            thumbnail_generator = ThumbnailGenerator(output_dir=thumbnail_dir)

            # Generate thumbnail using the prompts from the content package
            safe_title_thumbnail = safe_title[:20]  # Shorter version for thumbnail filename
            thumbnail_output_path = os.path.join(
                thumbnail_dir,
                f"thumbnail_{safe_title_thumbnail}_{timestamp}.jpg"
            )

            # Use the specialized thumbnail prompts from content package
            thumbnail_path = thumbnail_generator.generate_thumbnail(
                title=title,  # Use the generated title
                script_sections=script_cards,
                prompt=thumbnail_hf_prompt,  # Use the specialized HF prompt
                style=style,
                output_path=thumbnail_output_path
            )

            # If Hugging Face generation fails, it will use Unsplash with our query
            if not thumbnail_path:
                logger.info(f"Attempting with Unsplash query: {thumbnail_unsplash_query}")
                thumbnail_path = thumbnail_generator.fetch_image_unsplash(thumbnail_unsplash_query)

                if thumbnail_path:
                    # Create thumbnail with the downloaded image
                    thumbnail_path = thumbnail_generator.create_thumbnail(
                        title=title,
                        image_path=thumbnail_path,
                        output_path=thumbnail_output_path
                    )

            logger.info(f"Thumbnail generated at: {thumbnail_path}")
            thumbnail_generator.cleanup()

        except Exception as thumbnail_error:
            logger.error(f"Failed to generate thumbnail: {thumbnail_error}")
            # Continue without thumbnail if generation fails

        # Optional: YouTube Upload
        if os.getenv("ENABLE_YOUTUBE_UPLOAD", "false").lower() == "true":
            logger.info("Uploading to YouTube")
            youtube = get_authenticated_service()

            # Remove "LazyCreator presents: " prefix from title for upload
            upload_title = title
            if title.startswith("LazyCreator presents: "):
                upload_title = title.replace("LazyCreator presents: ", "")

            upload_video(
                youtube,
                video_path,
                upload_title,  # Use the cleaned title without prefix
                description,  # Use the generated description
                ["shorts", "ai", "technology"],  # Still include default tags
                thumbnail_path=thumbnail_path
            )

        return video_path, thumbnail_path

    except Exception as e:
        logger.error(f"Error generating YouTube Short: {e}")
        raise

def main(creator_type=None):

    try:
        topic = None
        # Only get creator for day if no creator_type is provided
        if creator_type is None:
            creator_type = get_creator_for_day()

        # Set style based on creator type
        style = "photorealistic"
        # Only log style for image-based creators
        if isinstance(creator_type, YTShortsCreator_I):
            logger.info(f"Using style: {style}")

        try:
            # Set max_duration to 25 seconds as requested
            max_duration = 25  # Full duration for shorts

            result = generate_youtube_short(
                topic,
                style=style,
                max_duration=max_duration,
                creator_type=creator_type
            )

            # Unpack the result (video_path, thumbnail_path)
            if isinstance(result, tuple) and len(result) == 2:
                video_path, thumbnail_path = result
                logger.info(f"Process completed successfully!")
                logger.info(f"Video path: {video_path}")
                if thumbnail_path:
                    logger.info(f"Thumbnail path: {thumbnail_path}")
            else:
                # For backward compatibility
                video_path = result
                logger.info(f"Process completed successfully! Video path: {video_path}")

            return video_path

        except Exception as e:
            logger.error(f"Error generating YouTube Short: {str(e)}")
            import traceback
            logger.error(f"Detailed error: {traceback.format_exc()}")
            raise
    except Exception as e:
        logger.error(f"Process failed: {str(e)}")
        import traceback
        logger.error(f"Detailed error trace: {traceback.format_exc()}")
        return None

if __name__ == "__main__":
    # Add ability to override creator type from command line
    import sys

    creator_type = None
    if len(sys.argv) > 1 and sys.argv[1] in ["video", "image"]:
        # Allow command-line specification of creator type
        if sys.argv[1] == "video":
            logger.info("Manually selected video-based creator (YTShortsCreator_V)")
            creator_type = YTShortsCreator_V()
        else:  # image
            logger.info("Manually selected image-based creator (YTShortsCreator_I)")
            creator_type = YTShortsCreator_I()

    # Call main with the selected creator type (or None for day-based selection)
    main(creator_type)
