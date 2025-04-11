import os  # for interacting with the file system and operating system
import pickle  # for saving objects to a file and loading them
from dotenv import load_dotenv
from google.oauth2.credentials import Credentials  # for storing and using an access token to authenticate with Google APIs
from google_auth_oauthlib.flow import InstalledAppFlow  # for handling the OAuth 2.0 flow with Google APIs
from google.auth.transport.requests import Request  # for making HTTP requests

# Load environment variables
load_dotenv()
CLIENT_SECRETS_FILE = os.getenv("YOUTUBE_CLIENT_SECRETS", "client_secret.json")
SCOPES = [
    "https://www.googleapis.com/auth/youtube.upload",
    "https://www.googleapis.com/auth/youtube",
    "https://www.googleapis.com/auth/youtube.force-ssl",
    "https://www.googleapis.com/auth/youtube.readonly"
]  # Added full YouTube scope for thumbnail uploads

def authenticate_youtube():
    credentials = None
    token_file = "token.pickle"

    # Load existing credentials if available
    if os.path.exists(token_file):
        with open(token_file, "rb") as token:
            credentials = pickle.load(token)

    # Check if credentials are invalid or expired
    if not credentials or not credentials.valid:
        try:
            if credentials and credentials.expired and credentials.refresh_token:
                # Refresh the token if possible
                credentials.refresh(Request())
                print("🔄 Token refreshed successfully!")
            else:
                # Prompt user for re-authentication if no valid refresh token
                print("🔑 Token expired or invalid. Re-authenticating...")
                flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRETS_FILE, SCOPES)
                credentials = flow.run_local_server(port=0)
        except Exception as e:
            print(f"❌ Error during token refresh or authentication: {e}")
            print("⚠️ Re-authenticating...")
            flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRETS_FILE, SCOPES)
            credentials = flow.run_local_server(port=0)

        # Save the new or refreshed credentials
        with open(token_file, "wb") as token:
            pickle.dump(credentials, token)
            print("💾 Token saved successfully!")

    print("✅ Authentication successful!")
    return credentials
