#!/usr/bin/env python
# fast_download.py - Quickly download YouTube videos and queue them for later processing

import os
import json
import argparse
import time
from typing import List, Dict
from dotenv import load_dotenv

# Load the main application config
from youtube_qa_app import Config

def main():
    parser = argparse.ArgumentParser(description="Fast YouTube downloader")
    parser.add_argument("url", help="YouTube video URL or file with URLs")
    parser.add_argument("--batch", action="store_true", help="URL is a file containing multiple URLs")
    parser.add_argument("--tags", help="Comma-separated list of tags to associate with the video")
    parser.add_argument("--whisper-model", choices=["tiny", "base", "small", "medium", "large"], 
                       help="Whisper model to use for transcription")
    parser.add_argument("--chunk-size", type=int, help="Override the default chunk size")
    parser.add_argument("--chunk-overlap", type=int, help="Override the default chunk overlap")
    parser.add_argument("--no-update", action="store_true", 
                       help="Don't update existing entries (add new ones instead)")
    
    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv()
    
    # Initialize config
    config = Config()
    
    # Create directories
    os.makedirs(config.AUDIO_DIR, exist_ok=True)
    pending_dir = os.path.join(config.AUDIO_DIR, "pending")
    os.makedirs(pending_dir, exist_ok=True)
    
    # Process tags
    tags = None
    if args.tags:
        tags = [tag.strip() for tag in args.tags.split(',')]
    
    # Process URLs
    if args.batch:
        # Process multiple URLs from a file
        if not os.path.exists(args.url):
            print(f"Batch file not found: {args.url}")
            return
            
        with open(args.url, 'r') as f:
            urls = [line.strip() for line in f if line.strip()]
            
        print(f"Found {len(urls)} URLs to process")
        for i, url in enumerate(urls):
            print(f"\nProcessing URL {i+1}/{len(urls)}: {url}")
            download_video(url, config, args, tags)
    else:
        # Process a single URL
        download_video(args.url, config, args, tags)
        
    print("\nAll downloads complete. To process these videos, run:")
    print("python process_later.py --all")


def download_video(video_url: str, config: Config, args, tags: List[str] = None) -> bool:
    """Download a YouTube video and queue it for later processing"""
    try:
        import yt_dlp
        
        # Extract video ID from URL
        video_id = video_url.split("watch?v=")[1].split("&")[0]
        
        # Check if audio file already exists
        audio_path = os.path.join(config.AUDIO_DIR, f"{video_id}.mp3")
        pending_file = os.path.join(config.AUDIO_DIR, "pending", f"{video_id}.json")
        
        if os.path.exists(audio_path):
            print(f"Audio file already exists: {audio_path}")
            with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
                info = ydl.extract_info(video_url, download=False)
                title = info.get('title', 'Unknown Title')
            print(f"Using existing audio file for: {title}")
        else:
            # Set up yt-dlp options
            ydl_opts = {
                'format': 'bestaudio/best',
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'mp3',
                    'preferredquality': '192',
                }],
                'outtmpl': f'{config.AUDIO_DIR}/{video_id}',
                'progress_hooks': [lambda d: print(f"\rDownloading: {d['_percent_str']} of {d.get('_total_bytes_str', 'Unknown size')}       ", end='')],
            }
            
            # Download the audio
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(video_url, download=True)
                title = info.get('title', 'Unknown Title')
            
            print(f"\nDownloaded audio for: {title}")
        
        # Create or update pending file with metadata
        metadata = {
            'video_id': video_id,
            'title': title,
            'url': video_url,
            'date_downloaded': time.strftime('%Y-%m-%d %H:%M:%S'),
            'whisper_model': args.whisper_model if args.whisper_model else config.WHISPER_MODEL,
            'chunk_size': args.chunk_size if args.chunk_size else config.CHUNK_SIZE,
            'chunk_overlap': args.chunk_overlap if args.chunk_overlap else config.CHUNK_OVERLAP,
            'update': not args.no_update if hasattr(args, 'no_update') else True,
            'tags': tags
        }
        
        with open(pending_file, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        print(f"Video queued for processing: {title}")
        return True
        
    except Exception as e:
        print(f"Error downloading video {video_url}: {str(e)}")
        return False


if __name__ == "__main__":
    main()
