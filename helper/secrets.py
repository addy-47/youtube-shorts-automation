import os
import json
from pathlib import Path
import logging
from google.cloud import secretmanager
from google.oauth2 import service_account
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

class SecretManager:
    def __init__(self):
        load_dotenv()
        self.project_id = os.getenv("GCP_PROJECT_ID")
        self.credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        self._client = None
        self._init_client()

    def _init_client(self):
        """Initialize the Secret Manager client with credentials."""
        try:
            if not self.credentials_path or not os.path.exists(self.credentials_path):
                raise ValueError("Master service account key file not found")

            credentials = service_account.Credentials.from_service_account_file(
                self.credentials_path,
                scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )
            self._client = secretmanager.SecretManagerServiceClient(credentials=credentials)
        except Exception as e:
            logger.error(f"Failed to initialize Secret Manager client: {e}")
            raise

    def _get_secret_version_path(self, secret_id: str, version: str = "latest") -> str:
        """Construct the path for the secret version."""
        return f"projects/{self.project_id}/secrets/{secret_id}/versions/{version}"

    def get_secret(self, secret_id: str, version: str = "latest") -> str:
        """Retrieve a secret from Secret Manager."""
        try:
            name = self._get_secret_version_path(secret_id, version)
            response = self._client.access_secret_version(name=name)
            return response.payload.data.decode("UTF-8")
        except Exception as e:
            logger.error(f"Failed to retrieve secret {secret_id}: {e}")
            return None

    def get_json_secret(self, secret_id: str, version: str = "latest") -> dict:
        """Retrieve and parse a JSON secret from Secret Manager."""
        secret_value = self.get_secret(secret_id, version)
        if secret_value:
            try:
                return json.loads(secret_value)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON secret {secret_id}: {e}")
                return None
        return None

class Secrets:
    """Singleton class to manage application secrets."""
    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Secrets, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self._secret_manager = SecretManager()
            self._cache = {}
            self._initialized = True

    def get_youtube_credentials(self) -> dict:
        """Get YouTube credentials from Secret Manager."""
        if "youtube_credentials" not in self._cache:
            self._cache["youtube_credentials"] = self._secret_manager.get_json_secret("youtube-client")
        return self._cache["youtube_credentials"]

    def write_temp_credentials(self, base_path: str = None) -> str:
        """
        Write YouTube client secrets to a temporary file and return its path.
        
        Args:
            base_path: Base directory for temporary credential files
            
        Returns:
            str: Path to the temporary youtube_client_secrets.json file, or None on failure.
        """
        if base_path is None:
            base_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "automation", "credentials")

        # Ensure the credentials directory exists
        os.makedirs(base_path, exist_ok=True)

        youtube_creds = self.get_youtube_credentials()

        youtube_path = os.path.join(base_path, "client_secret.json")

        try:
            # Write YouTube client secrets
            if youtube_creds:
                with open(youtube_path, 'w') as f:
                    json.dump(youtube_creds, f)
            else:
                logger.warning("YouTube credentials secret ('youtube-client') not found or is empty. `client_secret.json` will not be created.")
                return None

            return youtube_path
        except Exception as e:
            logger.error(f"Failed to write credentials to temp files: {e}")
            return None

    def cleanup_temp_credentials(self, *paths):
        """Clean up temporary credential files."""
        for path in paths:
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except Exception as e:
                    logger.error(f"Failed to remove temporary credential file {path}: {e}")
