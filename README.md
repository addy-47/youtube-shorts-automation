# YouTube Shorts Automation

Automatically generate and upload engaging YouTube Shorts with minimal effort.

## Overview

This project automates the creation of YouTube Shorts by:
1. Generating engaging content on any topic
2. Creating visually appealing videos with text overlays
3. Adding automated voiceovers
4. Uploading videos to YouTube

The system supports both image-based and video-based content creation, alternating between them to provide variety.

## Key Features

- **Content Generation**: AI-powered script and content generation
- **Text-to-Speech**: Automated voiceovers with natural-sounding voices
- **Dynamic Visuals**: Image or video sourcing based on content
- **Automated Publishing**: Direct upload to YouTube
- **Performance Optimized**: Parallel processing for fast video creation

## Optimized Video Generation

The system employs several optimization techniques to dramatically reduce video generation time:

### 1. Keyframe-Based Animation System
- Replaces callable animation functions with keyframe data
- Enables direct serialization for cross-process sharing
- Eliminates pre-rendering steps for complex animations

### 2. Parallel Processing
- **Background Fetching**: Concurrent downloading of visual assets
- **Audio Generation**: Parallel processing of voiceover clips
- **Text Rendering**: Simultaneous creation of text overlays
- **Clip Rendering**: Multi-process video segment rendering

### 3. Optimized Encoding
- Hardware acceleration detection (NVIDIA, Intel, AMD)
- Ultrafast presets for intermediate files
- Quality-optimized settings for final output

### 4. Memory Management
- Efficient resource cleanup
- Graduated rendering to avoid memory issues
- Optimized buffer sizes

## Project Structure

```
youtube-shorts-automation/
├── automation/          # Core automation modules
│   ├── content_generator.py     # AI content creation
│   ├── parallel_renderer.py     # Optimized multi-process rendering
│   ├── shorts_maker_I.py        # Image-based shorts creator
│   ├── shorts_maker_V.py        # Video-based shorts creator
│   ├── thumbnail.py             # Thumbnail generation
│   ├── voiceover.py             # Text-to-speech generation
│   ├── youtube_auth.py          # YouTube API authentication
│   └── youtube_upload.py        # Video uploading
│
├── helper/              # Utility functions and optimizations
│   ├── benchmark.py              # Performance testing tools
│   ├── blur.py                   # Image blur effects
│   ├── fetch.py                  # Resource downloading
│   ├── image.py                  # Image processing
│   ├── keyframe_animation.py     # Animation system
│   ├── news.py                   # News fetching
│   ├── parallel_tasks.py         # Concurrent task processing
│   ├── process.py                # Subprocess handling
│   ├── text.py                   # Text generation and formatting
│   └── video_encoder.py          # Optimized video encoding
│
├── run_benchmark.py     # Benchmark runner script
├── test_optimized_generation.py  # End-to-end test script
└── main.py              # Entry point
```

## Performance Improvements

Our optimizations deliver significant performance improvements:

1. **Keyframe Animation**: ~40% faster than the previous callable animation system
2. **Parallel Task Processing**: ~60% faster content preparation
3. **Optimized Encoding**: ~30% faster video generation with no quality loss
4. **Hardware Acceleration**: Up to 4x speed improvement when available

## Requirements

- Python 3.8+
- FFmpeg
- Required Python packages (see requirements.txt)
- API keys for various services (see .env.example)

## Setup

1. Clone the repository
2. Install required packages: `pip install -r requirements.txt`
3. Copy `.env.example` to `.env` and add your API keys
4. Run the application: `python main.py`

## Configuration

The system can be configured via environment variables or by editing the `.env` file:

- `YOUTUBE_TOPIC`: Default topic for content generation
- `ENABLE_YOUTUBE_UPLOAD`: Set to "true" to enable automatic uploading
- `DEBUG_MODE`: Set to "true" for verbose logging
- `TEMP_DIR`: Set a custom temporary directory to avoid disk space issues

## Advanced Usage

### Command Line Options

```bash
# Generate a short with video-based content
python main.py video

# Generate a short with image-based content
python main.py image

# Generate with explicit optimization flag
python main.py --enable-optimizations
```

### Testing and Benchmarking

The project includes comprehensive testing and benchmarking tools to evaluate performance improvements:

#### End-to-End Testing

```bash
# Run a basic end-to-end test of the optimized pipeline
python test_optimized_generation.py

# Compare optimized performance with baseline
python test_optimized_generation.py --compare
```

#### Benchmarking Tools

```bash
# Run all benchmarks
python run_benchmark.py

# Run specific benchmark tests
python run_benchmark.py --test keyframe
python run_benchmark.py --test parallel
python run_benchmark.py --test encoder

# Clean previous results
python run_benchmark.py --clean

# Save results to custom directory
python run_benchmark.py --output my_benchmark_results
```

The benchmark results include:
- JSON reports with detailed timing information
- Performance comparison plots for each optimization
- Summary report showing combined speedup factors

### Customization

You can customize the generation process by modifying:
- `style` parameters for different visual styles
- `voice_style` settings for different voiceover styles
- Content generation prompts in `content_generator.py`
- Animation parameters in `keyframe_animation.py`
- Encoding settings in `video_encoder.py`

## Temporary Files Management

The system provides intelligent temporary file management:
- Creates a dedicated temporary directory
- Cleans up intermediate files automatically
- Configurable via `TEMP_DIR` environment variable
- Monitors disk space usage during operations
