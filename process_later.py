#!/usr/bin/env python
# process_later.py - Process downloaded videos in a separate step

import os
import json
import argparse
import time
from pathlib import Path
from dotenv import load_dotenv

# Load the main application
from youtube_qa_app import Config, YouTubeProcessor

def main():
    parser = argparse.ArgumentParser(description="Process downloaded YouTube videos")
    parser.add_argument("--all", action="store_true", help="Process all pending videos")
    parser.add_argument("--video-id", help="Process a specific video ID")
    parser.add_argument("--whisper-model", choices=["tiny", "base", "small", "medium", "large"], 
                       help="Whisper model to use for transcription")
    parser.add_argument("--chunk-size", type=int, help="Override the default chunk size")
    parser.add_argument("--chunk-overlap", type=int, help="Override the default chunk overlap")
    parser.add_argument("--tags", help="Comma-separated list of tags to associate with the video")
    
    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv()
    
    # Initialize config
    config = Config()
    
    # Create the pending directory if it doesn't exist
    pending_dir = os.path.join(config.AUDIO_DIR, "pending")
    os.makedirs(pending_dir, exist_ok=True)
    
    # Process based on arguments
    if args.video_id:
        # Process a specific video
        process_video(args.video_id, config, args)
    elif args.all:
        # Process all pending videos
        pending_files = list(Path(pending_dir).glob("*.json"))
        if not pending_files:
            print("No pending videos to process")
            return
            
        print(f"Found {len(pending_files)} pending videos to process")
        for pending_file in pending_files:
            try:
                with open(pending_file, 'r') as f:
                    video_data = json.load(f)
                    
                video_id = video_data.get('video_id')
                if not video_id:
                    print(f"Invalid pending file: {pending_file}")
                    continue
                    
                print(f"\nProcessing video: {video_id}")
                process_video(video_id, config, args, video_data)
                
                # Remove the pending file after successful processing
                os.remove(pending_file)
            except Exception as e:
                print(f"Error processing pending file {pending_file}: {str(e)}")
    else:
        parser.print_help()

def process_video(video_id, config, args, video_data=None):
    """Process a specific video"""
    try:
        # Load metadata if not provided
        if not video_data:
            pending_file = os.path.join(config.AUDIO_DIR, "pending", f"{video_id}.json")
            if os.path.exists(pending_file):
                with open(pending_file, 'r') as f:
                    video_data = json.load(f)
            else:
                video_data = {'video_id': video_id}
        
        # Get video title
        title = video_data.get('title', video_id)
        
        # Get audio path
        audio_path = os.path.join(config.AUDIO_DIR, f"{video_id}.mp3")
        if not os.path.exists(audio_path):
            print(f"Audio file not found: {audio_path}")
            return False
        
        # Initialize the processor
        whisper_model = args.whisper_model if args.whisper_model else video_data.get('whisper_model', config.WHISPER_MODEL)
        print(f"Initializing processor with Whisper model: {whisper_model}")
        processor = YouTubeProcessor(config, whisper_model=whisper_model)
        
        # Get parameters
        chunk_size = args.chunk_size if args.chunk_size else video_data.get('chunk_size', config.CHUNK_SIZE)
        chunk_overlap = args.chunk_overlap if args.chunk_overlap else video_data.get('chunk_overlap', config.CHUNK_OVERLAP)
        
        # Process tags
        tags = None
        if args.tags:
            tags = [tag.strip() for tag in args.tags.split(',')]
        elif 'tags' in video_data:
            tags = video_data.get('tags')
        
        # Delete existing entries if needed
        update = video_data.get('update', True)
        if update:
            if "video_chunks" in processor.db.table_names():
                table = processor.db.open_table("video_chunks")
                count = table.count_rows(f"video_id = '{video_id}'")
                if count > 0:
                    print(f"Found {count} existing chunks for video ID: {video_id}")
                    print(f"Deleting existing data before reprocessing...")
                    processor.delete_video_data(video_id)
        
        # Transcribe the audio
        print(f"Transcribing audio for: {title}")
        transcript = processor.transcribe_audio(audio_path)
        
        # Process the transcript into chunks
        original_chunk_size = processor.config.CHUNK_SIZE
        original_chunk_overlap = processor.config.CHUNK_OVERLAP
        
        try:
            if chunk_size:
                processor.config.CHUNK_SIZE = chunk_size
            if chunk_overlap:
                processor.config.CHUNK_OVERLAP = chunk_overlap
                
            print(f"Processing transcript with chunk size: {processor.config.CHUNK_SIZE}, overlap: {processor.config.CHUNK_OVERLAP}")
            chunks = processor.process_transcript(transcript, video_id, title, tags=tags)
            
            # Generate embeddings
            print("Generating embeddings for text chunks...")
            chunks_with_embeddings = processor.generate_embeddings(chunks)
            
            # Store in LanceDB
            processor.store_in_lancedb(chunks_with_embeddings)
            
            print(f"\nSuccessfully processed video: {title} ({video_id})")
            if tags:
                print(f"Added tags: {', '.join(tags)}")
                
            return True
        finally:
            # Restore original settings
            processor.config.CHUNK_SIZE = original_chunk_size
            processor.config.CHUNK_OVERLAP = original_chunk_overlap
    
    except Exception as e:
        print(f"Error processing video {video_id}: {str(e)}")
        return False

if __name__ == "__main__":
    main()
