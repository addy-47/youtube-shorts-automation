import os
import random
import requests
import concurrent.futures
from moviepy.editor import VideoFileClip
import logging
import time
from helper.minor_helper import measure_time
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load API keys from environment variables
pexels_api_key = os.getenv("PEXELS_API_KEY")
pixabay_api_key = os.getenv("PIXABAY_API_KEY")
temp_dir = os.path.join(os.path.dirname(__file__), "temp")
os.makedirs(temp_dir, exist_ok=True)  # Create temp directory if it doesn't exist

@measure_time
def _fetch_videos(query, count=5, min_duration=5):
    """
    Fetch background videos from multiple sources with randomized API selection

    Args:
        query (str): Search term for videos
        count (int): Number of videos to fetch
        min_duration (int): Minimum video duration in seconds

    Returns:
        list: Paths to downloaded video files
    """
    # Determine how many videos to fetch from each source
    videos = []

    # Randomly decide which API to try first
    apis = ["pexels", "pixabay"]
    random.shuffle(apis)  # Shuffle the list to try APIs in random order

    # Try each API in sequence
    for api in apis:
        logger.info(f"Fetching {count} videos using {api} API")

        if api == "pexels":
            try:
                api_videos = _fetch_from_pexels(query, count, min_duration)
                if api_videos:  # If we got videos, add them and stop trying
                    videos.extend(api_videos)
                    break
            except Exception as e:
                logger.error(f"Error fetching videos from Pexels: {e}")
                # Continue to next API
        else:  # pixabay
            try:
                api_videos = _fetch_from_pixabay(query, count, min_duration)
                if api_videos:  # If we got videos, add them and stop trying
                    videos.extend(api_videos)
                    break
            except Exception as e:
                logger.error(f"Error fetching videos from Pixabay: {e}")
                # Continue to next API

    # If we have fewer videos than requested, but at least one, return what we have
    if not videos:
        logger.warning(f"Could not fetch any videos for query: {query}")

    return videos[:count]

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
        response = requests.get(url) # make a request to the API
        if response.status_code == 200: # if the request is successful
            data = response.json()  # changes the response in json to py dict data
            videos = data.get("hits", []) # get the videos from the data
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
                    video_path = os.path.join(temp_dir, f"pixabay_{video['id']}.mp4") # create a path for the video
                    with requests.get(video_url, stream=True) as r:  # get the video from the url
                        r.raise_for_status() # raise an error if the request is not successful
                        with open(video_path, 'wb') as f: # open the video file in write binary mode
                            for chunk in r.iter_content(chunk_size=8192): # iterate over the content of the video
                                f.write(chunk)
                    clip = VideoFileClip(video_path)
                    if clip.duration >= min_duration:
                        clip.close()
                        return video_path
                    clip.close()
                    # Remove the video if it's too short
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
                    video_files = video.get("video_files", []) # get the video files
                    if not video_files:
                        return None

                    video_url = video_files[0].get("link") # get the video link
                    if not video_url:
                        return None

                    video_path = os.path.join(temp_dir, f"pexels_{video['id']}.mp4")
                    with requests.get(video_url, stream=True) as r:
                        r.raise_for_status()
                        with open(video_path, 'wb') as f:
                            for chunk in r.iter_content(chunk_size=8192):
                                f.write(chunk)

                    clip = VideoFileClip(video_path)
                    if clip.duration >= min_duration:
                        clip.close()
                        return video_path
                    clip.close()
                    # Remove the video if it's too short
                    if os.path.exists(video_path):
                        os.remove(video_path)
                    return None
                except Exception as e:
                    logger.error(f"Error downloading video from Pexels: {e}")
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
        logger.warning(f"Pexels API returned status code {response.status_code}")
        return []
    except Exception as e:
        logger.error(f"Error fetching videos from Pexels: {e}")
        return []
