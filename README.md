# YouTube Shorts Automation

This project automates the creation of YouTube Shorts videos using AI-generated scripts, background videos, and text-to-speech (TTS) technology.

## Features

- Fetches the latest AI news using the NewsAPI
- Generates a script for the YouTube Short based on the latest AI news
- Creates a video with background clips, text overlays, and TTS audio
- Optionally uploads the video to YouTube

## Project Structure

```
youtube-shorts-automation/
├── .env                       # Environment variables
├── main.py                    # Main script
├── README.md                  # Project documentation
├── requirements.txt           # Project dependencies
├── run_short_automation.bat   # Batch file to run the automation
├── schedule.py                # Script for scheduling tasks (optional)
├── script_generator.py        # Script generation logic
├── video_maker.py             # Video creation logic
├── voiceover.py               # Text-to-speech logic
├── youtube_auth.py            # YouTube authentication logic
├── youtube_upload.py          # YouTube upload logic
├── ai_shorts_output/          # Output directory for generated videos
├── ffmpeg/                    # Directory for ffmpeg binaries
├── fonts/                     # Directory for font files
```

## Setup

1. **Clone the repository:**

   ```sh
   git clone https://github.com/yourusername/youtube-shorts-automation.git
   cd youtube-shorts-automation
   ```

2. **Install dependencies:**

   ```sh
   pip install -r requirements.txt
   ```

3. **Set up API credentials:**

   - Create a `client_secret.json` file for YouTube API credentials
   - Create a `google_credentials.json` file for Google API credentials

   _Note: These credential files are not included in the repository for security reasons and must be created by each user._

4. **Set up environment variables:**

   Create a `.env` file in the root directory and add the following variables:

   ```env
   NEWS_API_KEY=your_news_api_key
   PEXELS_API_KEY=your_pexels_api_key
   PIXABAY_API_KEY=your_pixabay_api_key
   USE_AZURE_TTS=false
   AZURE_VOICE=en-US-JennyNeural
   YOUTUBE_TOPIC=Artificial Intelligence
   ENABLE_YOUTUBE_UPLOAD=false
   ADD step 3, 4 and other large files to .gitignore file
   ```

5. **Download NLTK data:**

   ```sh
   python -m nltk.downloader stopwords punkt
   ```

6. **Set up fonts:**

   Place your font files in the `fonts/` directory.

## Usage

1. **Run the main script:**

   ```sh
   python main.py
   ```

   This will generate a YouTube Short video based on the latest AI news and save it in the `ai_shorts_output` directory.

2. **Optional: Upload to YouTube:**

   If you want to upload the generated video to YouTube:

   - Set `ENABLE_YOUTUBE_UPLOAD=true` in the `.env` file
   - Ensure you have valid credentials in your `client_secret.json` file
   - Run the script as normal with `python main.py`

## Scripts Description

- **main.py**: Main script to generate and optionally upload YouTube Shorts.
- **script_generator.py**: Generates the script for the YouTube Short.
- **video_maker.py**: Creates the video with background clips, text overlays, and TTS audio.
- **youtube_upload.py**: Handles uploading the video to YouTube.
- **voiceover.py**: Handles text-to-speech (TTS) using Azure TTS or gTTS.

## Creating API Credentials

### YouTube API Credentials

1. Go to the [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select an existing project
3. Enable the YouTube Data API v3
4. Create OAuth 2.0 client credentials and download as `client_secret.json`
5. Place the file in the root directory of the project

### Google API Credentials

1. Go to the [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new service account or use an existing one
3. Download the service account key as JSON
4. Rename the file to `google_credentials.json` and place it in the root directory

## License

This project is licensed under the MIT License.
