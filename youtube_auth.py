import os  # for interacting with the file system and operating system
import pickle  # for  saving objects to a file and loading them
from dotenv import load_dotenv
from google.oauth2.credentials import Credentials # for storing and using an access token to authenticate with Google APIs
# even though the class is imported, it is not used directly in the code as it is used internally by google_auth_oauthlib.flow
from google_auth_oauthlib.flow import InstalledAppFlow # for handling the OAuth 2.0 flow with Google APIs
from google.auth.transport.requests import Request # for making HTTP requests

# Load environment variables
load_dotenv()
CLIENT_SECRETS_FILE = os.getenv("YOUTUBE_CLIENT_SECRETS", "client_secret.json")
SCOPES = ["https://www.googleapis.com/auth/youtube.upload"] # YouTube Data API v3 scope

def authenticate_youtube():
    credentials = None
    if os.path.exists("token.pickle"):
        with open("token.pickle", "rb") as token: # Open the file in binary mode -rb (read binary)
            credentials = pickle.load(token)
    if not credentials or not credentials.valid:
        if credentials and credentials.expired and credentials.refresh_token:
            credentials.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRETS_FILE, SCOPES)
            credentials = flow.run_local_server(port=0)
        with open("token.pickle", "wb") as token:
            pickle.dump(credentials, token)

    print("✅ Authentication successful!")
    return credentials
