# YouTube Shorts Automation

This project automates the creation of YouTube Shorts videos using AI-generated scripts, images, background videos, and text-to-speech (TTS) technology.

## Features

- Creates AI-powered YouTube Shorts with dynamic transitions and effects
- Supports both image-based and video-based shorts generation
- AI image generation using Hugging Face Stable Diffusion API
- Background video fetching from Pexels and Pixabay
- Multiple TTS options:
  - Google Cloud TTS (high quality)
  - Azure TTS (configurable voices)
  - gTTS (fallback option)
- Advanced video features:
  - Parallel video processing for faster rendering
  - Word-by-word text animations
  - Customizable transitions and effects
  - Smart background video processing
  - Edge blur and background blur options
  - Watermark support
  - Automatic video duration adjustment
  - Hardware acceleration support (NVIDIA GPU)
- Automatic fallback mechanisms for reliability
- Progress logging and error handling

## Project Structure

```
youtube-shorts-automation/
├── automation/               # Core automation modules
│   ├── shorts_maker_I.py    # Image-based shorts creator
│   ├── shorts_maker_V.py    # Video-based shorts creator
│   ├── parallel_renderer.py # Parallel video rendering
│   ├── parallel_tasks.py    # Parallel task management
│   ├── thumbnail.py         # Thumbnail generation
│   ├── voiceover.py         # Text-to-speech base class
│   ├── voiceover_azure.py   # Azure TTS implementation
│   ├── youtube_auth.py      # YouTube authentication
│   ├── youtube_upload.py    # YouTube upload logic
│   ├── content_generator.py # Content generation logic
│   └── schedule.py          # Scheduling functionality
├── helper/                  # Helper utilities
│   ├── image.py            # Image processing utilities
│   ├── fetch.py            # Content fetching utilities
│   ├── crossfade.py        # Transition effects
│   ├── text.py             # Text processing utilities
│   ├── audio.py            # Audio processing utilities
│   ├── memory.py           # Memory management
│   ├── process.py          # Process management
│   ├── minor_helper.py     # Miscellaneous utilities
│   └── blur.py             # Blur effects
├── .env                    # Environment variables
├── main.py                 # Main script
├── ai_shorts_output/       # Output directory for generated videos
├── ffmpeg/                 # Directory for ffmpeg binaries
├── fonts/                  # Directory for font files
└── logs/                   # Directory for log files
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

   - **Google Cloud:**
     - Create a Google Cloud Project.
     - Enable the "Secret Manager API" and "YouTube Data API v3".
     - Create a service account with at least the "Secret Manager Secret Accessor" role.
     - Download the JSON key for this service account. This will be your "master" credential file.
     - Store your other secrets (like your YouTube OAuth `client_secret.json` content and other API keys) in Google Secret Manager within your project. The application expects secrets named `youtube-client-secrets` and `google-credentials`.
   - **OAuth Consent Screen (For YouTube Upload):**
     - In the Google Cloud Console, go to "APIs & Services" -> "OAuth consent screen".
     - Set the User Type to "External".
     - Fill in the required app information (app name, user support email, developer contact).
     - On the "Scopes" page, you don't need to add scopes manually; the app will request them.
     - On the "Test users" page, click "+ ADD USERS" and add the Google account email you will use to authorize the YouTube uploads. **This is a critical step to avoid `access_denied` errors while the app is in "Testing" mode.**
   - Set up Hugging Face API token for AI image generation
   - Set up Pexels and Pixabay API keys for video backgrounds

4. **Set up environment variables:**
   Create a `.env` file in the root directory and add the following variables:

   ```env
   # Google Cloud Platform
   GCP_PROJECT_ID=your-gcp-project-id
   # --- IMPORTANT ---
   # Use an ABSOLUTE path to your master service account JSON key file.
   # Example for Linux/macOS: /home/user/keys/my-gcp-key.json
   # Example for Windows: C:\Users\user\keys\my-gcp-key.json
   GOOGLE_APPLICATION_CREDENTIALS=/path/to/your/master-gcp-service-account-key.json

   # API Keys
   HUGGINGFACE_API_KEY=your_huggingface_key
   NEWS_API_KEY=your_news_api_key
   PEXELS_API_KEY=your_pexels_api_key
   PIXABAY_API_KEY=your_pixabay_api_key

   # TTS Configuration
   USE_GOOGLE_TTS=true
   USE_AZURE_TTS=false
   AZURE_VOICE=en-US-JennyNeural

   # Video Settings
   ENABLE_YOUTUBE_UPLOAD=false
   YOUTUBE_TOPIC=Artificial Intelligence
   HF_MODEL=stabilityai/stable-diffusion-xl-base-1.0

   # Parallel Processing
   ENABLE_PARALLEL_RENDERING=true
   MAX_PARALLEL_TASKS=4
   ENABLE_GPU_ACCELERATION=true
   ```

5. **Install additional requirements:**
   - FFmpeg (required for video processing)
   - ImageMagick (required for text effects)
   - CUDA toolkit (optional, for GPU acceleration)

## Usage

1. **Basic usage:**

   ```sh
   python main.py
   ```

2. **With specific options:**

   ```sh
   python main.py --style "digital art" --voice-style "enthusiastic" --add-watermark
   ```

3. **Schedule automated runs:**
   ```sh
   python schedule.py
   ```

## Advanced Features

### Parallel Processing System

- **parallel_renderer.py**: Handles parallel video rendering with GPU acceleration
- **parallel_tasks.py**: Manages concurrent task execution and resource allocation
- Features:
  - Multi-process video rendering for faster output
  - Smart clip pre-rendering for complex compositions
  - Hardware acceleration support
  - Memory-efficient processing of large videos
  - Automatic task distribution and load balancing
  - Progress tracking and error recovery

### Image-based Shorts (shorts_maker_I.py)

- Uses AI-generated images for visually appealing content
- Automatic style selection and prompt enhancement
- Smart fallback to video mode if image generation fails
- Zoom and transition effects for still images

### Video-based Shorts (shorts_maker_V.py)

- Smart video background selection and processing
- Multiple transition effects between scenes
- Word-by-word text animation
- Background blur and edge blur effects

### Helper Utilities

- **image.py**: Advanced image processing and manipulation
- **fetch.py**: Efficient content fetching with caching
- **crossfade.py**: Customizable transition effects
- **text.py**: Text processing and animation utilities
- **audio.py**: Audio processing and enhancement
- **memory.py**: Memory management and optimization
- **process.py**: Process management and monitoring
- **minor_helper.py**: Miscellaneous utility functions
- **blur.py**: Advanced blur effects and filters

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License.

## Acknowledgments

- FFmpeg for video processing
- MoviePy for Python video editing
- Hugging Face for AI image generation
- Pexels and Pixabay for video content
