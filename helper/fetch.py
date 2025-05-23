import os
import random
import requests
import concurrent.futures
from moviepy  import VideoFileClip
import logging
import time
from helper.minor_helper import measure_time

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create temp directories if they don't exist
video_temp_dir = os.path.join("automation", "temp", "video_downloads")
os.makedirs(video_temp_dir, exist_ok=True)

# Try to load API keys from environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    logger.info("python-dotenv not installed, loading API keys directly")

# Load API keys
pexels_api_key = os.environ.get("PEXELS_API_KEY", "YOUR_API_KEY")
pixabay_api_key = os.environ.get("PIXABAY_API_KEY", "YOUR_API_KEY")

# Constants
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds

@measure_time
def fetch_videos_parallel(queries, count_per_query=1, min_duration=5):
    """
    Fetch videos for multiple queries in parallel

    Args:
        queries (list): List of search terms
        count_per_query (int): Number of videos to fetch per query
        min_duration (int): Minimum video duration in seconds

    Returns:
        dict: Dictionary mapping queries to lists of video paths
    """
    start_time = time.time()
    logger.info(f"Fetching videos for {len(queries)} queries in parallel")

    def fetch_for_query(query):
        """Helper function to fetch videos for a single query"""
        try:
            videos = _fetch_videos(query, count=count_per_query, min_duration=min_duration)
            return query, videos
        except Exception as e:
            logger.error(f"Error fetching videos for query '{query}': {e}")
            return query, []

    # Use ThreadPoolExecutor for I/O-bound operations
    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(queries), 10)) as executor:
        future_to_query = {executor.submit(fetch_for_query, query): query for query in queries}

        for future in concurrent.futures.as_completed(future_to_query):
            query = future_to_query[future]
            try:
                query, videos = future.result()
                results[query] = videos
                logger.info(f"Fetched {len(videos)} videos for query '{query}'")
            except Exception as e:
                logger.error(f"Error processing query '{query}': {e}")
                results[query] = []

    total_time = time.time() - start_time
    logger.info(f"Fetched videos for {len(queries)} queries in {total_time:.2f} seconds")

    return results

@measure_time
def _fetch_videos(query, count=5, min_duration=5):
    """
    Fetch background videos from different APIs

    Args:
        query (str): Search term for videos
        count (int): Number of videos to fetch
        min_duration (int): Minimum video duration in seconds

    Returns:
        list: Paths to downloaded video files
    """
    # Try different video APIs in order of preference
    videos = []

    # Define APIs to try (with different counts to increase chances)
    apis = [
        {"name": "pexels", "func": _fetch_from_pexels, "count": min(count * 2, 10)},
        {"name": "pixabay", "func": _fetch_from_pixabay, "count": min(count * 2, 10)}
    ]

    # Shuffle APIs to distribute load
    random.shuffle(apis)

    # Try each API until we get enough videos
    for api in apis:
        if len(videos) >= count:
            break

        try:
            logger.info(f"Fetching {api['count']} videos using {api['name']} API")
            api_videos = api["func"](query, api["count"], min_duration)
            videos.extend(api_videos)
        except Exception as e:
            logger.error(f"Error fetching videos from {api['name']}: {e}")
            # Continue to next API

    # If we have fewer videos than requested, but at least one, return what we have
    if not videos:
        logger.warning(f"Could not fetch any videos for query: {query}")

    return videos[:count]

def _download_with_retry(url, output_path, headers=None, max_retries=MAX_RETRIES, chunk_size=8192):
    """
    Download a file with retry logic

    Args:
        url (str): URL to download from
        output_path (str): Path to save the file
        headers (dict): Headers for the request
        max_retries (int): Maximum number of retries
        chunk_size (int): Size of chunks to download

    Returns:
        bool: True if download successful, False otherwise
    """
    for attempt in range(max_retries):
        try:
            with requests.get(url, headers=headers, stream=True) as r:
                r.raise_for_status()
                with open(output_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=chunk_size):
                        f.write(chunk)
            return True
        except (requests.exceptions.RequestException, requests.exceptions.HTTPError) as e:
            logger.warning(f"Download attempt {attempt+1}/{max_retries} failed: {e}")
            if attempt < max_retries - 1:
                # Exponential backoff
                sleep_time = RETRY_DELAY * (2 ** attempt)
                time.sleep(sleep_time)
            else:
                logger.error(f"Failed to download after {max_retries} attempts: {url}")
                return False
        except Exception as e:
            logger.error(f"Unexpected error downloading {url}: {e}")
            return False

@measure_time
def _fetch_from_pixabay(query, count, min_duration):
    """
    Fetch background videos from Pixabay API

    Args:
        query (str): Search term for videos
        count (int): Number of videos to fetch
        min_duration (int): Minimum video duration in seconds

    Returns:
        list: Paths to downloaded video files
    """
    try:
        url = f"https://pixabay.com/api/videos/?key={pixabay_api_key}&q={query}&min_width=1080&min_height=1920&per_page=20"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            videos = data.get("hits", [])
            video_paths = []
            # Randomly select videos from the top 10
            top_videos = videos[:10]
            # Then randomly select 'count' videos from the top 10
            if len(top_videos) > count:
                selected_videos = random.sample(top_videos, count)
            else:
                selected_videos = top_videos

            def download_and_check_video(video):
                try:
                    video_url = video["videos"]["large"]["url"]
                    video_path = os.path.join(video_temp_dir, f"pixabay_{video['id']}.mp4")

                    # Skip download if the file already exists and is valid
                    if os.path.exists(video_path):
                        # Check if the file is valid and meets duration requirement
                        try:
                            clip = VideoFileClip(video_path)
                            if clip.duration >= min_duration:
                                clip.close()
                                return video_path
                            clip.close()
                            # Remove the video if it's too short
                            os.remove(video_path)
                        except Exception:
                            # Remove the file if it's corrupted
                            if os.path.exists(video_path):
                                os.remove(video_path)

                    # Download the video with retry
                    if _download_with_retry(video_url, video_path):
                        # Check duration
                        try:
                            clip = VideoFileClip(video_path)
                            if clip.duration >= min_duration:
                                clip.close()
                                return video_path
                            clip.close()
                            # Remove the video if it's too short
                            if os.path.exists(video_path):
                                os.remove(video_path)
                        except Exception as e:
                            logger.error(f"Error checking video duration: {e}")
                            if os.path.exists(video_path):
                                os.remove(video_path)
                    return None
                except Exception as e:
                    logger.error(f"Error downloading video from Pixabay: {e}")
                    return None

            # Use ThreadPoolExecutor to download videos in parallel
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(selected_videos), 5)) as executor:
                # Submit all download tasks and collect futures
                future_to_video = {executor.submit(download_and_check_video, video): video for video in selected_videos}

                # Process completed futures
                for future in concurrent.futures.as_completed(future_to_video):
                    video_path = future.result()
                    if video_path:
                        video_paths.append(video_path)

            return video_paths

        # If response wasn't 200, return empty list
        logger.warning(f"Pixabay API returned status code {response.status_code}")
        return []
    except Exception as e:
        logger.error(f"Error fetching videos from Pixabay: {e}")
        return []

@measure_time
def _fetch_from_pexels(query, count=5, min_duration=15):
    """
    Fetch background videos from Pexels API

    Args:
        query (str): Search term for videos
        count (int): Number of videos to fetch
        min_duration (int): Minimum video duration in seconds

    Returns:
        list: Paths to downloaded video files
    """
    try:
        url = f"https://api.pexels.com/videos/search?query={query}&per_page=20&orientation=portrait"
        headers = {"Authorization": pexels_api_key}
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            videos = data.get("videos", [])
            video_paths = []
            # Randomly select videos from the top 10
            top_videos = videos[:10]
            # Then randomly select 'count' videos from those top 10
            if len(top_videos) > count:
                selected_videos = random.sample(top_videos, count)
            else:
                selected_videos = top_videos

            def download_and_check_video(video):
                try:
                    video_files = video.get("video_files", [])
                    if not video_files:
                        return None

                    # Find HD or higher quality video
                    hd_videos = [v for v in video_files if v.get("quality") in ["hd", "sd", "fhd"]]
                    if hd_videos:
                        video_file = hd_videos[0]
                    else:
                        video_file = video_files[0]

                    video_url = video_file.get("link")
                    if not video_url:
                        return None

                    video_path = os.path.join(video_temp_dir, f"pexels_{video['id']}.mp4")

                    # Skip download if the file already exists and is valid
                    if os.path.exists(video_path):
                        # Check if the file is valid and meets duration requirement
                        try:
                            clip = VideoFileClip(video_path)
                            if clip.duration >= min_duration:
                                clip.close()
                                return video_path
                            clip.close()
                            # Remove the video if it's too short
                            os.remove(video_path)
                        except Exception:
                            # Remove the file if it's corrupted
                            if os.path.exists(video_path):
                                os.remove(video_path)

                    # Download the video with retry
                    if _download_with_retry(video_url, video_path, headers=headers):
                        # Check duration
                        try:
                            clip = VideoFileClip(video_path)
                            if clip.duration >= min_duration:
                                clip.close()
                                return video_path
                            clip.close()
                            # Remove the video if it's too short
                            if os.path.exists(video_path):
                                os.remove(video_path)
                        except Exception as e:
                            logger.error(f"Error checking video duration: {e}")
                            if os.path.exists(video_path):
                                os.remove(video_path)
                    return None
                except Exception as e:
                    logger.error(f"Error downloading video from Pexels: {e}")
                    return None

            # Use ThreadPoolExecutor to download videos in parallel
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(selected_videos), 5)) as executor:
                # Submit all download tasks
                future_to_video = {executor.submit(download_and_check_video, video): video for video in selected_videos}

                # Process completed futures
                for future in concurrent.futures.as_completed(future_to_video):
                    video_path = future.result()
                    if video_path:
                        video_paths.append(video_path)

            return video_paths

        # If response wasn't 200, return empty list
        logger.warning(f"Pexels API returned status code {response.status_code}")
        return []
    except Exception as e:
        logger.error(f"Error fetching videos from Pexels: {e}")
        return []

@measure_time
def fetch_image_unsplash(self, query, file_path=None):
    """
    Fetch an image from Unsplash API based on query

    Args:
        query (str): Search query for Unsplash
        file_path (str): Path to save the image, if None a path will be generated

    Returns:
        str: Path to the downloaded image or None if failed
    """
    if not file_path:
        file_path = os.path.join(self.temp_dir, f"thumbnail_unsplash_{int(time.time())}_{random.randint(1000, 9999)}.jpg")

    # Check if Unsplash API key is available
    if not self.unsplash_api_key:
        logger.error("No Unsplash API key provided.")
        return None

    try:
        # Clean query for Unsplash search
        clean_query = query.replace("eye-catching", "").replace("thumbnail", "").replace("YouTube Shorts", "")
        # Remove any double spaces
        while "  " in clean_query:
            clean_query = clean_query.replace("  ", " ")
        clean_query = clean_query.strip(" ,")

        logger.info(f"Searching Unsplash with query: {clean_query}")

        # Make request to Unsplash API
        params = {
            "query": clean_query,
            "orientation": "landscape",
            "per_page": 30,
            "client_id": self.unsplash_api_key
        }

        response = requests.get(self.unsplash_api_url, params=params, timeout=10)

        if response.status_code == 200:
            data = response.json()

            # Check if we have results
            if data["results"] and len(data["results"]) > 0:
                # Pick a random image from top results for variety
                max_index = min(10, len(data["results"]))
                image_data = random.choice(data["results"][:max_index])
                image_url = image_data["urls"]["regular"]

                # Download the image
                img_response = requests.get(image_url, timeout=10)
                if img_response.status_code == 200:
                    with open(file_path, "wb") as f:
                        f.write(img_response.content)
                    logger.info(f"Unsplash image downloaded to {file_path}")

                    # Add attribution as required by Unsplash API guidelines
                    attribution = f"Photo by {image_data['user']['name']} on Unsplash"
                    logger.info(f"Image attribution: {attribution}")

                    return file_path
                else:
                    logger.error(f"Failed to download image from Unsplash: {img_response.status_code}")
            else:
                logger.error("No results found on Unsplash")
        else:
            logger.error(f"Unsplash API error: {response.status_code} - {response.text}")

    except Exception as e:
        logger.error(f"Error fetching image from Unsplash: {e}")

    return None