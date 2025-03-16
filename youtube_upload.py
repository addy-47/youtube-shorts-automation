import os  # for file operations
import googleapiclient.discovery # for interacting with the YouTube API
import googleapiclient.errors # for handling API errors
from youtube_auth import authenticate_youtube
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_authenticated_service():
    """Load YouTube API credentials."""
    credentials = authenticate_youtube()
    return googleapiclient.discovery.build("youtube", "v3", credentials=credentials) #

def upload_video(youtube, file_path, title, description, tags, privacy="public"):
    """Upload a video to YouTube."""
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

    media_body = googleapiclient.http.MediaFileUpload(file_path, chunksize=-1, resumable=True) # creates a object of the file to be uploaded
    try:
        request = youtube.videos().insert(  # creates a request object to upload the video
            part="snippet,status",
            body=body,
            media_body=media_body
        )
        response = request.execute() # executes the request
        logger.info(f"✅ Upload successful! Video ID: {response.get('id')}")
    except googleapiclient.errors.HttpError as e:
        logger.error(f"Upload failed: {e}")
        raise

if __name__ == "__main__":
    youtube = get_authenticated_service()
    upload_video(youtube, "short_output.mp4", "Test Short", "A test video.", ["shorts", "test"])
