import os  # for file operations
import googleapiclient.discovery # for interacting with the YouTube API
import googleapiclient.errors # for handling API errors
from automation.youtube_auth import authenticate_youtube
import logging
from google.auth.transport.requests import Request
from dotenv import load_dotenv
import io

# Configure logging - don't use basicConfig since main.py handles this
logger = logging.getLogger(__name__)

load_dotenv()

def get_authenticated_service():
    """Load YouTube API credentials."""
    credentials = authenticate_youtube()
    return googleapiclient.discovery.build("youtube", "v3", credentials=credentials) #

def upload_video(youtube, file_path, title, description, tags, thumbnail_path=None, privacy="public"):
    """
    Upload a video to YouTube with optional thumbnail.

    Args:
        youtube: Authenticated YouTube API service
        file_path (str): Path to the video file
        title (str): Title for the video
        description (str): Video description
        tags (list): List of tags
        thumbnail_path (str): Optional path to thumbnail image
        privacy (str): Privacy status ('public', 'private', 'unlisted')

    Returns:
        str: Video ID of the uploaded video or None if failed
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Video file '{file_path}' not found.")

    body = {
        "snippet": {
            "title": title,
            "description": description,
            "tags": tags,
            "categoryId": "22"  # People & Blogs
        },
        "status": {
            "privacyStatus": privacy
        }
    }

    media_body = googleapiclient.http.MediaFileUpload(file_path, chunksize=-1, resumable=True)
    try:
        # Upload the video first
        logger.info(f"Uploading video: {title}")
        request = youtube.videos().insert(
            part="snippet,status",
            body=body,
            media_body=media_body
        )
        response = request.execute()
        video_id = response.get('id')
        logger.info(f"✅ Video upload successful! Video ID: {video_id}")

        # Upload thumbnail if provided
        if thumbnail_path and os.path.exists(thumbnail_path):
            try:
                logger.info(f"Uploading thumbnail for video ID: {video_id}")
                # Use a more robust approach for thumbnail upload
                media = googleapiclient.http.MediaFileUpload(
                    thumbnail_path,
                    mimetype='image/jpeg',
                    resumable=True
                )
                youtube.thumbnails().set(
                    videoId=video_id,
                    media_body=media
                ).execute()
                logger.info(f"✅ Thumbnail upload successful!")
            except googleapiclient.errors.HttpError as e:
                logger.error(f"Thumbnail upload failed: {e}")
                # Try alternative approach if the first one fails
                try:
                    logger.info("Attempting alternative thumbnail upload method...")
                    with open(thumbnail_path, 'rb') as image_file:
                        image_data = image_file.read()
                        youtube.thumbnails().set(
                            videoId=video_id,
                            media_body=googleapiclient.http.MediaIoBaseUpload(
                                io.BytesIO(image_data),
                                mimetype='image/jpeg',
                                resumable=True
                            )
                        ).execute()
                    logger.info(f"✅ Thumbnail upload successful with alternative method!")
                except Exception as alt_error:
                    logger.error(f"Alternative thumbnail upload also failed: {alt_error}")
                    # Continue even if thumbnail upload fails

        return video_id
    except googleapiclient.errors.HttpError as e:
        logger.error(f"Video upload failed: {e}")
        raise

if __name__ == "__main__":
    youtube = get_authenticated_service()
    upload_video(youtube, "short_output.mp4", "Test Short", "A test video.", ["shorts", "test"])
