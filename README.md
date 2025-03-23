# YouTube Video QA Bot

This application allows you to download YouTube videos, transcribe them, and create a searchable question-answering system based on the video content. It supports processing multiple videos to build a comprehensive knowledge base that you can query.

Created by John Clark 

## Features

- Download audio from YouTube videos with progress tracking
- Skip re-downloading for previously processed videos
- Transcribe audio using OpenAI's Whisper model
- Chunk and organize transcripts with timestamps
- Generate vector embeddings for text chunks
- Store embeddings in LanceDB for efficient similarity search
- Answer questions across multiple videos using a language model
- View source information with video titles and timestamps
- Selectively update or delete embeddings for specific videos
- List and manage your video database
- Fast mode for instant startup of management commands
- Lazy loading of models for improved performance
- Compatibility with various LanceDB versions

## Requirements

- Python 3.8 or higher
- OpenAI API key for the QA component
- Internet connection for downloading videos and accessing API
- FFmpeg for audio processing
- Various Python libraries (see Installation section)

## Installation

### 1. Clone this repository

```bash
git clone https://github.com/yourusername/youtube-qa-bot.git
cd youtube-qa-bot
```

### 2. Set up a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install the required Python packages

```bash
pip install -r requirements.txt
```

### 4. Install FFmpeg (required for audio processing)

FFmpeg is an essential dependency for handling audio files. Install it based on your operating system:

**For Ubuntu/Debian Linux:**
```bash
sudo apt update
sudo apt install ffmpeg
```

**For Red Hat/Fedora Linux:**
```bash
sudo dnf install ffmpeg
```

**For macOS (using Homebrew):**
```bash
brew install ffmpeg
```

**For Windows:**
1. Download an FFmpeg build from the official site: https://ffmpeg.org/download.html
2. Extract the files to a location on your computer (e.g., C:\ffmpeg)
3. Add the bin folder to your system PATH, or specify the path directly in your code

Verify FFmpeg installation:
```bash
ffmpeg -version
```

### 5. Configure the application

```bash
cp .env.template .env
# Edit the .env file with your OpenAI API key and other settings
```

## Usage

### Adding YouTube Videos

To add a single YouTube video to your database:

```bash
python youtube_qa_app.py add "https://www.youtube.com/watch?v=VIDEO_ID"
```

You'll see progress indicators for:
- Download progress
- Transcription progress
- Embedding generation progress

#### Using Fast Mode for Downloads

If you want to download videos quickly and process them later:

```bash
# Download only without immediate processing
python youtube_qa_app.py add "https://www.youtube.com/watch?v=VIDEO_ID" --fast

# Process all previously downloaded videos later
python process_later.py --all

# Or process a specific video
python process_later.py --video-id VIDEO_ID
```

This approach is especially useful when downloading multiple videos, as it allows you to queue up the downloads quickly and then process them in batch later.

#### Processing Multiple Videos

You can process multiple videos sequentially:

```bash
python youtube_qa_app.py add "https://www.youtube.com/watch?v=VIDEO_ID1"
python youtube_qa_app.py add "https://www.youtube.com/watch?v=VIDEO_ID2"
python youtube_qa_app.py add "https://www.youtube.com/watch?v=VIDEO_ID3"
```

Or use a batch file approach:

```bash
# Create a file with video URLs
echo "https://www.youtube.com/watch?v=VIDEO_ID1" > videos.txt
echo "https://www.youtube.com/watch?v=VIDEO_ID2" >> videos.txt
echo "https://www.youtube.com/watch?v=VIDEO_ID3" >> videos.txt

# Process all videos in the file
./process_videos.sh videos.txt
```

#### Re-processing and Updating Videos

If you want to update a previously processed video (for example, if the transcription needs improvement or you want to change the chunking parameters), you can now do so without resetting the entire database:

```bash
# Re-process a video (automatically updates existing embeddings)
python youtube_qa_app.py add "https://www.youtube.com/watch?v=VIDEO_ID" --force
```

By default, the system will:
1. Identify existing entries for the same video
2. Delete those entries from the database
3. Process the video again and store the new embeddings

If you want to keep the old entries and add new ones instead (not recommended as it could create duplicates), use:

```bash
python youtube_qa_app.py add "https://www.youtube.com/watch?v=VIDEO_ID" --force --no-update
```

### Managing Your Video Database

#### Listing Videos

You can view all videos in your database:

```bash
python youtube_qa_app.py list
```

This will show video IDs, titles, tags, and the number of chunks for each video.

For instant startup without loading models:

```bash
python youtube_qa_app.py list --fast
```

You can filter by tags:

```bash
python youtube_qa_app.py list --tags "machine-learning,tutorial"
```

#### Deleting Videos

To remove a specific video from your database:

```bash
# Delete by video ID
python youtube_qa_app.py delete --video-id "VIDEO_ID"

# Delete by video URL
python youtube_qa_app.py delete --url "https://www.youtube.com/watch?v=VIDEO_ID"

# Use fast mode for instant operation
python youtube_qa_app.py delete --video-id "VIDEO_ID" --fast
```

This will remove all chunks and embeddings associated with the specified video without affecting other videos in the database.

#### System Information

To view system information and database statistics:

```bash
python youtube_qa_app.py info
```

This shows:
- Configuration settings
- LanceDB version
- Total chunks in database
- Number of unique videos
- List of videos with chunk counts

### Asking Questions About Videos

To ask a question about the content in your video database:

```bash
python youtube_qa_app.py query "What is the main topic discussed in these videos?"
```

The system will:
1. Search across ALL videos in your database
2. Find the most relevant content from any video
3. Generate an answer that synthesizes information from potentially multiple videos
4. Show which videos and timestamps the information came from

#### Example Queries Across Multiple Videos

```bash
# Find information that might span multiple tutorial videos
python youtube_qa_app.py query "What are the key steps in setting up a machine learning project?"

# Compare concepts across different videos
python youtube_qa_app.py query "What different approaches to neural network training are discussed across these videos?"

# Find specific information that might be in any video
python youtube_qa_app.py query "How is backpropagation explained in these videos?"
```

### Advanced Query Options

You can specify additional parameters for queries:

```bash
# Specify the number of chunks to retrieve for answering
python youtube_qa_app.py query "Explain gradient descent" --results 10

# Filter results to specific videos (comma-separated video IDs)
python youtube_qa_app.py query "What is discussed about transformers?" --videos "VIDEO_ID1,VIDEO_ID3"

# Filter results by tags
python youtube_qa_app.py query "Show examples of backpropagation" --tags "neural-networks,tutorial"

# Adjust the response temperature (0.0-1.0) for more/less creative answers
python youtube_qa_app.py query "Summarize the key points" --temperature 0.7
```

## Performance Optimization

The application includes several optimizations to improve performance:

### Fast Mode

Most commands support a `--fast` flag that significantly improves startup time:

```bash
# List videos instantly without loading models
python youtube_qa_app.py list --fast

# Delete a video without loading models
python youtube_qa_app.py delete --video-id "VIDEO_ID" --fast

# View system information
python youtube_qa_app.py info

# Download a video without immediate processing
python youtube_qa_app.py add "URL" --fast
```

Fast mode works by:
1. Only loading heavy models (Whisper, embeddings) when absolutely necessary
2. Deferring processing operations to be done later
3. Using lightweight database operations for management commands

### Lazy Loading

Models are only loaded when they're actually needed:
- The Whisper model only loads when transcription is performed
- The embedding model only loads when generating or searching embeddings
- Commands like "list" and "delete" don't load any models at all

### Batch Processing

Several operations use batch processing to improve efficiency:
- Database operations process chunks in batches to optimize memory usage
- The `process_later.py` script can batch process multiple videos
- Vector operations are optimized for better performance

### Separate Processing Steps

You can separate the downloading and processing steps:
1. Download videos quickly using `--fast` mode
2. Process them later with `process_later.py`

This is especially useful for:
- Downloading many videos at once
- Scheduling processing for when you have more resources available
- Using different machines for downloading vs. processing

## Runtime Parameters

### Add Command Parameters

```bash
# Specify a custom model for whisper transcription (tiny, base, small, medium, large)
python youtube_qa_app.py add "https://www.youtube.com/watch?v=VIDEO_ID" --whisper-model base

# Force re-download and re-processing even if the video exists
python youtube_qa_app.py add "https://www.youtube.com/watch?v=VIDEO_ID" --force

# Re-process without updating (adds as new entries)
python youtube_qa_app.py add "https://www.youtube.com/watch?v=VIDEO_ID" --force --no-update

# Specify custom chunk size and overlap
python youtube_qa_app.py add "https://www.youtube.com/watch?v=VIDEO_ID" --chunk-size 600 --chunk-overlap 150

# Add custom tags to videos for filtering later
python youtube_qa_app.py add "https://www.youtube.com/watch?v=VIDEO_ID" --tags "machine-learning,tutorial"

# Download only (for later processing)
python youtube_qa_app.py add "https://www.youtube.com/watch?v=VIDEO_ID" --fast
```

### Delete Command Parameters

```bash
# Delete by video ID
python youtube_qa_app.py delete --video-id "VIDEO_ID"

# Delete by video URL
python youtube_qa_app.py delete --url "https://www.youtube.com/watch?v=VIDEO_ID"

# Use fast mode
python youtube_qa_app.py delete --video-id "VIDEO_ID" --fast
```

### List Command Parameters

```bash
# List all videos
python youtube_qa_app.py list

# List videos with specific tags
python youtube_qa_app.py list --tags "machine-learning,tutorial"

# Use fast mode
python youtube_qa_app.py list --fast
```

### Query Command Parameters

```bash
# Specify the number of relevant chunks to use
python youtube_qa_app.py query "What is the main topic?" --results 5

# Filter by video IDs
python youtube_qa_app.py query "Explain the concept" --videos "VIDEO_ID1,VIDEO_ID2"

# Filter by tags
python youtube_qa_app.py query "Show examples of backpropagation" --tags "neural-networks,tutorial"

# Adjust the LLM temperature
python youtube_qa_app.py query "Summarize the key points" --temperature 0.7

# Specify the maximum token length for the response
python youtube_qa_app.py query "Give a detailed explanation" --max-tokens 1000
```

## Embedding Models

The application uses Sentence Transformers for generating embeddings. By default, it uses the `all-MiniLM-L6-v2` model, which provides a good balance between performance and quality.

### Available Embedding Models

Here are some popular Sentence Transformer models you can use:

1. `all-MiniLM-L6-v2` (default) - Fast, lightweight model with good performance
2. `paraphrase-MiniLM-L6-v2` - Optimized for paraphrase identification
3. `multi-qa-MiniLM-L6-cos-v1` - Good for question-answering tasks
4. `all-mpnet-base-v2` - More powerful but slower model
5. `distiluse-base-multilingual-cased-v1` - Good for multilingual applications

### Using a Different Embedding Model

To change the embedding model, you can:

1. **Update the .env file**:
   ```
   EMBEDDING_MODEL=all-mpnet-base-v2
   ```

2. **Modify the code**: In `youtube_qa_app.py`, find the YouTubeProcessor.__init__ method and update the embedding model initialization:
   ```python
   self.embedding_model = SentenceTransformer("your-preferred-model")
   ```

3. **Add a command-line parameter**: You can extend the application to accept an embedding model parameter:
   ```python
   # Add to the argument parser:
   add_parser.add_argument("--embedding-model", help="Embedding model to use")
   
   # Use in the processor initialization:
   embedding_model = args.embedding_model if hasattr(args, 'embedding_model') and args.embedding_model else None
   processor = YouTubeProcessor(config, 
                                whisper_model=args.whisper_model if hasattr(args, 'whisper_model') else None,
                                embedding_model=embedding_model)
   ```

### Important Notes About Embedding Models

1. **Vector Dimensions**: If you change the embedding model, you might also need to update the vector dimensions in the `VideoChunk` class. Different models produce embeddings of different sizes.

2. **Consistency**: Once you start building a database with one embedding model, you should stick with it. Switching models will require rebuilding your database from scratch.

3. **Performance Trade-offs**: Larger models (like `all-mpnet-base-v2`) generally provide better quality embeddings but are slower to run and require more memory.

4. **Storage Requirements**: The size of your LanceDB database will depend on the dimensionality of your chosen embedding model and the number/length of your videos.

## Troubleshooting

### YouTube Download Issues

If you encounter errors when downloading videos:

1. **HTTP 400 Error**: YouTube occasionally changes their API, which can cause download failures. Try updating yt-dlp:
   ```bash
   pip install --upgrade yt-dlp
   ```

2. **Geo-Restricted Videos**: Some videos may not be accessible from your location. Consider using a VPN.

3. **Rate Limiting**: If you're downloading many videos in succession, YouTube might rate-limit you. Add delays between downloads.

### Database Management Issues

1. **Failed to Update Video**: If you encounter errors when updating a video, you can try:
   - Deleting the video first: `python youtube_qa_app.py delete --video-id "VIDEO_ID" --fast`
   - Then adding it again: `python youtube_qa_app.py add "https://www.youtube.com/watch?v=VIDEO_ID"`

2. **Database Corruption**: If your LanceDB database becomes corrupted, you might still need to reset it:
   ```bash
   ./reset_db.sh  # This will create a backup before resetting
   # Or use the Python version
   ./reset_db_py.sh
   ```

3. **Video ID Extraction Failure**: If you see errors about failing to extract a video ID from a URL, ensure you're using the standard YouTube URL format (`https://www.youtube.com/watch?v=VIDEO_ID`). URLs from playlists or mobile apps might need to be converted.

### LanceDB Compatibility Notes

This application has been tested with LanceDB versions 0.21.1 through 0.3.x. If you encounter errors related to database operations, they may be version-specific. The application includes fallback mechanisms for different LanceDB versions.

1. **Unhashable Type Errors**: If you see errors about "unhashable type: numpy.ndarray", the application should handle these automatically with the latest code.

2. **Method Not Found Errors**: If you see errors like "object has no attribute 'query'", the application now uses alternative methods that work across versions.

3. **Database Upgrades**: If you upgrade your LanceDB version, you may need to recreate your database to take advantage of new features.

4. **Version Check**: You can see your LanceDB version with:
   ```bash
   python youtube_qa_app.py info
   ```

### FFmpeg Missing

If you see errors related to FFmpeg:

```
ERROR: Postprocessing: ffprobe and ffmpeg not found. Please install or provide the path using --ffmpeg-location
```

Install FFmpeg following the instructions in the Installation section. If you can't install it system-wide, you can download a portable version and specify its location:

```python
# In your code:
ydl_opts = {
    # other options...
    'ffmpeg_location': '/path/to/ffmpeg/bin',
}
```

### Embedding Model Errors

If you encounter errors related to the embedding model:

1. **Model Not Found**: Ensure you're using a valid model name from the Sentence Transformers library. If you see an error like:
   ```
   OSError: model-name is not a local folder and is not a valid model identifier
   ```
   Use one of the recommended models listed in the "Available Embedding Models" section.

2. **Vector Dimension Mismatch**: If you change the embedding model after creating your database, you might encounter dimension mismatches. In this case, you'll need to:
   - Update the vector dimension in the `VideoChunk` class
   - Rebuild your database by reprocessing all videos

### Whisper Transcription Issues

1. **CUDA/GPU-related errors**: If you see warnings about FP16 not being supported on CPU, these are normal and can be ignored if you're running on CPU.

2. **Out of Memory**: Large videos might cause memory issues during transcription. Try using a smaller Whisper model:
   ```bash
   python youtube_qa_app.py add "URL" --whisper-model tiny
   ```

3. **Slow Transcription**: Transcription can be slow on CPU. Consider using a GPU if available, or use smaller Whisper models for faster (though less accurate) transcription.

## Data Organization

When you add multiple videos, the system organizes them as follows:

- Each video gets a unique `video_id` (based on the YouTube video ID)
- Each transcript is divided into chunks with overlap for context preservation
- Chunks are stored with metadata including:
  - Video ID and title
  - Timestamp information
  - Chunk index
  - Generated embeddings

This structure allows the system to:
1. Track which information came from which video
2. Provide accurate source attribution
3. Support cross-video knowledge retrieval
4. Enable efficient vector similarity search
5. Selectively update or delete specific videos

## Customization

You can customize various aspects of the application by editing the .env file:

- `CHUNK_SIZE`: The target size of text chunks (in characters)
- `CHUNK_OVERLAP`: The amount of overlap between chunks for context continuity
- `EMBEDDING_MODEL`: The Sentence Transformers model to use for embeddings
- `LLM_MODEL`: The OpenAI model to use for question answering

## Example Workflow

Here's a complete example of building a knowledge base on machine learning topics:

```bash
# Add several educational videos
python youtube_qa_app.py add "https://www.youtube.com/watch?v=aircAruvnKk" --tags "neural-networks,3blue1brown"
python youtube_qa_app.py add "https://www.youtube.com/watch?v=IHZwWFHWa-w" --tags "neural-networks,gradient-descent,3blue1brown"
python youtube_qa_app.py add "https://www.youtube.com/watch?v=Ilg3gGewQ5U" --tags "neural-networks,backpropagation,3blue1brown"

# List videos in the database
python youtube_qa_app.py list --fast

# List only videos with a specific tag
python youtube_qa_app.py list --tags "backpropagation" --fast

# Update a video with better chunking parameters
python youtube_qa_app.py add "https://www.youtube.com/watch?v=Ilg3gGewQ5U" --force --chunk-size 600 --chunk-overlap 200

# Ask questions that span across videos
python youtube_qa_app.py query "How do neural networks learn from data?" --results 8

# Ask about specific topics with tag filtering
python youtube_qa_app.py query "Explain backpropagation in detail" --tags "backpropagation"

# Ask about specific videos only
python youtube_qa_app.py query "What is covered in this specific video?" --videos "aircAruvnKk"

# Remove a video from the database
python youtube_qa_app.py delete --video-id "IHZwWFHWa-w" --fast

# View system information
python youtube_qa_app.py info
```

The system will retrieve information across all relevant videos, synthesize an answer, and show which parts came from which videos with timestamps.

## License

This project is licensed under the MIT License:

```
MIT License

Copyright (c) 2025 John Clark

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```