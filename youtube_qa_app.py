import os
import tempfile
import time
import sys
from typing import List, Dict, Optional, Union, Set
import argparse
from pathlib import Path
import logging
import json
from tqdm import tqdm
import pandas as pd
import numpy as np
from dotenv import load_dotenv

# Video downloading and transcription
from pytube import YouTube
from pytube.cli import on_progress
import whisper

# Vector database
import lancedb
from lancedb.pydantic import LanceModel, Vector

# Embedding and QA components
from sentence_transformers import SentenceTransformer
from langchain_openai import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Set up configuration
class Config:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    LANCEDB_PATH = os.getenv("LANCEDB_PATH", "lancedb")
    AUDIO_DIR = os.getenv("AUDIO_DIR", "audio_files")
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    LLM_MODEL = os.getenv("LLM_MODEL", "gpt-3.5-turbo")
    DEFAULT_MAX_TOKENS = int(os.getenv("DEFAULT_MAX_TOKENS", "1000"))
    DEFAULT_TEMPERATURE = float(os.getenv("DEFAULT_TEMPERATURE", "0.0"))
    WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")

# Pydantic model for LanceDB
class VideoChunk(LanceModel):
    id: str
    video_id: str
    title: str
    chunk_index: int
    text: str
    timestamp: Optional[str] = None
    tags: Optional[List[str]] = None
    embedding: Vector(384) = None
    
    class Config:
        use_enum_values = True

class YouTubeProcessor:
    def __init__(self, config: Config, whisper_model=None):
        self.config = config
        
        # Create directories if they don't exist
        os.makedirs(self.config.AUDIO_DIR, exist_ok=True)
        os.makedirs(self.config.LANCEDB_PATH, exist_ok=True)
        
        # Initialize the embedding model
        self.embedding_model = SentenceTransformer(self.config.EMBEDDING_MODEL)
        
        # Connect to LanceDB
        self.db = lancedb.connect(self.config.LANCEDB_PATH)
        
        # Initialize the transcription model
        whisper_model = whisper_model or self.config.WHISPER_MODEL
        logger.info(f"Loading Whisper model: {whisper_model}")
        with tqdm(total=100, desc=f"Loading {whisper_model} model") as pbar:
            self.whisper_model = whisper.load_model(whisper_model)
            pbar.update(100)


    def download_youtube_audio(self, video_url: str, force: bool = False) -> str:
        """Download audio from a YouTube video using yt-dlp and save it to a file."""
        try:
            import yt_dlp
            
            logger.info(f"Processing video: {video_url}")
            
            # Extract video ID from URL
            video_id = video_url.split("watch?v=")[1].split("&")[0]
            
            # Check if audio file already exists
            audio_path = os.path.join(self.config.AUDIO_DIR, f"{video_id}.mp3")
            if os.path.exists(audio_path) and not force:
                logger.info(f"Audio file already exists: {audio_path}")
                # We need to get the title separately
                with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
                    info = ydl.extract_info(video_url, download=False)
                    title = info.get('title', 'Unknown Title')
                print(f"\nUsing existing audio file for: {title}")
                return audio_path, video_id, title
            
            # Set up yt-dlp options
            ydl_opts = {
                'format': 'bestaudio/best',
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'mp3',
                    'preferredquality': '192',
                }],
                'outtmpl': f'{self.config.AUDIO_DIR}/{video_id}',
                'progress_hooks': [lambda d: print(f"\rDownloading: {d['_percent_str']} of {d.get('_total_bytes_str', 'Unknown size')}       ", end='')],
            }
            
            # Download the audio
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(video_url, download=True)
                title = info.get('title', 'Unknown Title')
            
            print(f"\nDownloaded audio for: {title}")
            
            # Return path with .mp3 extension (yt-dlp adds this)
            return audio_path, video_id, title
        
        except Exception as e:
            logger.error(f"Error downloading YouTube audio: {str(e)}")
            raise


    def transcribe_audio(self, audio_path: str) -> Dict:
        """Transcribe the audio file using Whisper."""
        try:
            logger.info(f"Transcribing audio: {audio_path}")
            print(f"\nTranscribing audio... (this may take a while depending on the length)")
            
            # Create a simple spinner to show progress since Whisper doesn't provide progress updates
            spinner = ['|', '/', '-', '\\']
            i = 0
            
            # Start transcription in a way that allows us to show a spinner
            start_time = time.time()
            transcription_started = False
            
            while not transcription_started or time.time() - start_time < 1:
                sys.stdout.write('\r' + spinner[i % len(spinner)] + ' Transcribing...')
                sys.stdout.flush()
                i += 1
                time.sleep(0.1)
                
                if not transcription_started:
                    transcription_started = True
                    # Actually perform the transcription
                    result = self.whisper_model.transcribe(audio_path)
            
            duration = time.time() - start_time
            print(f"\rTranscription completed in {duration:.2f} seconds                      ")
            
            return result
        except Exception as e:
            logger.error(f"Error transcribing audio: {str(e)}")
            raise

    def process_transcript(self, transcript: Dict, video_id: str, title: str, tags: List[str] = None) -> List[Dict]:
        """Process the transcript by chunking it and preparing it for embedding."""
        full_text = transcript.get("text", "")
        segments = transcript.get("segments", [])
        
        print(f"Processing transcript into chunks...")
        
        # We'll use the segments to chunk the text with timestamps
        chunks = []
        current_chunk = ""
        current_segment_start = None
        current_segment_end = None
        
        for segment in segments:
            segment_text = segment.get("text", "").strip()
            segment_start = self._format_timestamp(segment.get("start", 0))
            segment_end = self._format_timestamp(segment.get("end", 0))
            
            # Start a new chunk if we don't have one
            if not current_chunk:
                current_chunk = segment_text
                current_segment_start = segment_start
                current_segment_end = segment_end
                continue
            
            # If adding this segment would exceed our chunk size, save the current chunk
            if len(current_chunk) + len(segment_text) > self.config.CHUNK_SIZE:
                chunk_data = {
                    "video_id": video_id,
                    "title": title,
                    "text": current_chunk,
                    "timestamp": f"{current_segment_start} - {current_segment_end}",
                    "tags": tags or []
                }
                chunks.append(chunk_data)
                
                # Start a new chunk with overlap
                words = current_chunk.split()
                if len(words) > self.config.CHUNK_OVERLAP:
                    # Take the last CHUNK_OVERLAP words for overlap
                    overlap_text = " ".join(words[-self.config.CHUNK_OVERLAP:])
                    current_chunk = overlap_text + " " + segment_text
                else:
                    current_chunk = segment_text
                
                current_segment_start = segment_start
                current_segment_end = segment_end
            else:
                current_chunk += " " + segment_text
                current_segment_end = segment_end
        
        # Add the last chunk if it exists
        if current_chunk:
            chunk_data = {
                "video_id": video_id,
                "title": title,
                "text": current_chunk,
                "timestamp": f"{current_segment_start} - {current_segment_end}",
                "tags": tags or []
            }
            chunks.append(chunk_data)
        
        # Add chunk indices
        for i, chunk in enumerate(chunks):
            chunk["chunk_index"] = i
            chunk["id"] = f"{video_id}_{i}"
        
        print(f"Created {len(chunks)} chunks from the transcript")
        return chunks

    def _format_timestamp(self, seconds: float) -> str:
        """Format seconds into a timestamp string (MM:SS)."""
        minutes, seconds = divmod(int(seconds), 60)
        return f"{minutes:02d}:{seconds:02d}"

    def generate_embeddings(self, chunks: List[Dict]) -> List[Dict]:
        """Generate embeddings for each text chunk."""
        logger.info(f"Generating embeddings for {len(chunks)} chunks")
        
        for chunk in tqdm(chunks, desc="Generating embeddings"):
            # Generate embedding
            embedding = self.embedding_model.encode(chunk["text"])
            chunk["embedding"] = embedding.tolist()
        
        return chunks

    def store_in_lancedb(self, chunks: List[Dict]) -> None:
        """Store the chunks with embeddings in LanceDB."""
        logger.info(f"Storing {len(chunks)} chunks in LanceDB")
        
        # Convert to VideoChunk objects
        video_chunks = [VideoChunk(**chunk) for chunk in chunks]
        
        # Get or create the table
        table_name = "video_chunks"
        if table_name in self.db.table_names():
            table = self.db.open_table(table_name)
            # Add data to existing table
            table.add(video_chunks)
        else:
            # Create new table
            self.db.create_table(table_name, data=video_chunks)
        
        logger.info(f"Successfully stored chunks in LanceDB table: {table_name}")

    def process_video(self, video_url: str, tags: List[str] = None, force: bool = False, 
                      chunk_size: int = None, chunk_overlap: int = None) -> None:
        """Process a YouTube video: download, transcribe, chunk, embed, and store."""
        try:
            # Override chunk settings if provided
            original_chunk_size = self.config.CHUNK_SIZE
            original_chunk_overlap = self.config.CHUNK_OVERLAP
            
            if chunk_size:
                self.config.CHUNK_SIZE = chunk_size
            if chunk_overlap:
                self.config.CHUNK_OVERLAP = chunk_overlap
                
            try:
                # Download the audio
                audio_path, video_id, title = self.download_youtube_audio(video_url, force=force)
                
                # Transcribe the audio
                transcript = self.transcribe_audio(audio_path)
                
                # Process the transcript into chunks
                chunks = self.process_transcript(transcript, video_id, title, tags=tags)
                
                # Generate embeddings
                print("Generating embeddings for text chunks...")
                chunks_with_embeddings = self.generate_embeddings(chunks)
                
                # Store in LanceDB
                self.store_in_lancedb(chunks_with_embeddings)
                
                print(f"\nSuccessfully processed video: {title} ({video_id})")
                if tags:
                    print(f"Added tags: {', '.join(tags)}")
            finally:
                # Restore original settings
                self.config.CHUNK_SIZE = original_chunk_size
                self.config.CHUNK_OVERLAP = original_chunk_overlap
                
        except Exception as e:
            logger.error(f"Error processing video {video_url}: {str(e)}")
            raise

    def search_similar(self, query: str, limit: int = 5, video_ids: List[str] = None, tags: List[str] = None) -> List[Dict]:
        """Search for chunks similar to the query, with optional filtering by video_id or tags."""
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode(query)
            
            # Open the table
            table = self.db.open_table("video_chunks")
            
            # Start building the search query
            search_query = table.search(query_embedding)
            
            # Apply filters if specified
            if video_ids:
                video_filter = " OR ".join([f"video_id = '{vid}'" for vid in video_ids])
                search_query = search_query.where(video_filter)
            
            if tags:
                # Filter for chunks that have at least one of the specified tags
                # LanceDB supports array_contains function for array fields
                tags_filter = " OR ".join([f"array_contains(tags, '{tag}')" for tag in tags])
                search_query = search_query.where(tags_filter)
            
            # Execute the search and limit results
            results = search_query.limit(limit).to_pandas()
            
            return results.to_dict('records')
        except Exception as e:
            logger.error(f"Error searching: {str(e)}")
            raise


def answer_question(self, question: str, num_results: int = 5, video_ids: List[str] = None, 
                    tags: List[str] = None, temperature: float = None, max_tokens: int = None) -> str:
    """Answer a question based on the content of the videos."""
    try:
        # Show what we're searching for
        filter_info = []
        if video_ids:
            filter_info.append(f"from {len(video_ids)} specific videos")
        if tags:
            filter_info.append(f"with tags: {', '.join(tags)}")
        
        filter_msg = f" ({', '.join(filter_info)})" if filter_info else ""
        print(f"Searching for relevant content{filter_msg}...")
        
        # Search for relevant chunks with optional filters
        relevant_chunks = self.search_similar(
            question, 
            limit=num_results,
            video_ids=video_ids,
            tags=tags
        )
        
        if not relevant_chunks:
            return "I couldn't find any relevant information to answer your question."
        
        # Extract the text from the chunks
        context = "\n\n".join([chunk["text"] for chunk in relevant_chunks])
        
        # Set default values if not provided
        temperature = temperature if temperature is not None else self.config.DEFAULT_TEMPERATURE
        max_tokens = max_tokens if max_tokens is not None else self.config.DEFAULT_MAX_TOKENS
        
        print(f"Found {len(relevant_chunks)} relevant segments. Generating answer...")
        
        # Initialize a language model
        llm = ChatOpenAI(
            api_key=self.config.OPENAI_API_KEY,
            model_name=self.config.LLM_MODEL,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        # Create a QA chain
        qa_chain = load_qa_chain(llm, chain_type="stuff")
        
        # Generate the answer
        result = qa_chain.invoke({"input_documents": [{"page_content": context}], "question": question})
        
        # Enhance the answer with source information
        answer = result["output_text"]
        
        # Get unique video sources for better organization
        video_sources = {}
        for chunk in relevant_chunks:
            video_id = chunk.get("video_id")
            title = chunk.get("title", "Unknown")
            timestamp = chunk.get("timestamp", "Unknown")
            
            if video_id not in video_sources:
                video_sources[video_id] = {"title": title, "timestamps": []}
            
            video_sources[video_id]["timestamps"].append(timestamp)
        
        # Add source information in a more organized way
        answer += "\n\nSources:"
        for i, (video_id, info) in enumerate(video_sources.items()):
            title = info["title"]
            timestamps = info["timestamps"]
            answer += f"\n{i+1}. {title}"
            
            # If there are multiple timestamps from the same video, list them
            if len(timestamps) > 1:
                answer += " (Times: " + ", ".join(timestamps) + ")"
            else:
                answer += f" (Time: {timestamps[0]})"
        
        return answer
    except Exception as e:
        logger.error(f"Error answering question: {str(e)}")
        return f"An error occurred while trying to answer your question: {str(e)}"




def main():
    parser = argparse.ArgumentParser(description="YouTube Search QA Bot")
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Add video command
    add_parser = subparsers.add_parser("add", help="Add a YouTube video")
    add_parser.add_argument("url", help="YouTube video URL")
    add_parser.add_argument("--force", action="store_true", help="Force re-download and re-processing")
    add_parser.add_argument("--tags", help="Comma-separated list of tags to associate with the video")
    add_parser.add_argument("--whisper-model", choices=["tiny", "base", "small", "medium", "large"], 
                           help="Whisper model to use for transcription")
    add_parser.add_argument("--chunk-size", type=int, help="Override the default chunk size")
    add_parser.add_argument("--chunk-overlap", type=int, help="Override the default chunk overlap")
    
    # Query command
    query_parser = subparsers.add_parser("query", help="Query the video database")
    query_parser.add_argument("question", help="Your question")
    query_parser.add_argument("--results", type=int, default=5, help="Number of results to use")
    query_parser.add_argument("--videos", help="Comma-separated list of video IDs to search within")
    query_parser.add_argument("--tags", help="Comma-separated list of tags to filter by")
    query_parser.add_argument("--temperature", type=float, help="Temperature for the language model (0.0-1.0)")
    query_parser.add_argument("--max-tokens", type=int, help="Maximum tokens for the response")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Initialize the processor
    config = Config()
    processor = YouTubeProcessor(config, whisper_model=args.whisper_model if hasattr(args, 'whisper_model') else None)
    
    if args.command == "add":
        # Process tags if provided
        tags = None
        if hasattr(args, 'tags') and args.tags:
            tags = [tag.strip() for tag in args.tags.split(',')]
            
        processor.process_video(
            args.url,
            tags=tags,
            force=args.force if hasattr(args, 'force') else False,
            chunk_size=args.chunk_size if hasattr(args, 'chunk_size') else None,
            chunk_overlap=args.chunk_overlap if hasattr(args, 'chunk_overlap') else None
        )
    elif args.command == "query":
        # Process video IDs if provided
        video_ids = None
        if hasattr(args, 'videos') and args.videos:
            video_ids = [vid.strip() for vid in args.videos.split(',')]
            
        # Process tags if provided
        tags = None
        if hasattr(args, 'tags') and args.tags:
            tags = [tag.strip() for tag in args.tags.split(',')]
            
        answer = processor.answer_question(
            args.question,
            num_results=args.results,
            video_ids=video_ids,
            tags=tags,
            temperature=args.temperature if hasattr(args, 'temperature') else None,
            max_tokens=args.max_tokens if hasattr(args, 'max_tokens') else None
        )
        print("\nAnswer:")
        print(answer)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
