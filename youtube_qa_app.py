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

from langchain.schema import Document

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

# Helper function for cosine similarity
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

class YouTubeProcessor:
    def __init__(self, config: Config, whisper_model=None, embedding_model_name=None):
        self.config = config
        
        # Store model names for lazy loading
        self._whisper_model = None
        self._whisper_model_name = whisper_model or self.config.WHISPER_MODEL
        self._embedding_model = None
        self._embedding_model_name = embedding_model_name or self.config.EMBEDDING_MODEL
        
        # Create directories if they don't exist
        os.makedirs(self.config.AUDIO_DIR, exist_ok=True)
        os.makedirs(self.config.LANCEDB_PATH, exist_ok=True)
        
        # Connect to LanceDB - this operation is fast so we can do it immediately
        self.db = lancedb.connect(self.config.LANCEDB_PATH)
    
    @property
    def embedding_model(self):
        """Lazy load the embedding model only when needed"""
        if self._embedding_model is None:
            logger.info(f"Loading embedding model: {self._embedding_model_name}")
            self._embedding_model = SentenceTransformer(self._embedding_model_name)
        return self._embedding_model
    
    @property
    def whisper_model(self):
        """Lazy load the whisper model only when needed"""
        if self._whisper_model is None:
            logger.info(f"Loading Whisper model: {self._whisper_model_name}")
            with tqdm(total=100, desc=f"Loading {self._whisper_model_name} model") as pbar:
                self._whisper_model = whisper.load_model(self._whisper_model_name)
                pbar.update(100)
        return self._whisper_model

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
        """Store the chunks with embeddings in LanceDB with optimized batch processing."""
        logger.info(f"Storing {len(chunks)} chunks in LanceDB")
        
        # Convert to VideoChunk objects
        video_chunks = [VideoChunk(**chunk) for chunk in chunks]
        
        # Get or create the table
        table_name = "video_chunks"
        if table_name in self.db.table_names():
            table = self.db.open_table(table_name)
            
            # Add data in optimized batch size - this should work with LanceDB 0.21.1
            BATCH_SIZE = 100  # Adjust based on your system's memory
            for i in range(0, len(video_chunks), BATCH_SIZE):
                batch = video_chunks[i:i+BATCH_SIZE]
                table.add(batch)
                logger.info(f"Added batch {i//BATCH_SIZE + 1}/{(len(video_chunks)-1)//BATCH_SIZE + 1} ({len(batch)} chunks)")
        else:
            # Create new table - this should work with any LanceDB version
            self.db.create_table(table_name, data=video_chunks)
        
        logger.info(f"Successfully stored chunks in LanceDB table: {table_name}")

    def process_video(self, video_url: str, tags: List[str] = None, force: bool = False, 
                    chunk_size: int = None, chunk_overlap: int = None, update: bool = True) -> None:
        """
        Process a YouTube video: download, transcribe, chunk, embed, and store.
        
        Args:
            video_url: The YouTube video URL to process
            tags: Optional list of tags to associate with the video
            force: Force re-download and re-processing even if audio exists
            chunk_size: Override the default chunk size
            chunk_overlap: Override the default chunk overlap
            update: If True, delete existing entries for this video before adding new ones
        """
        try:
            # Override chunk settings if provided
            original_chunk_size = self.config.CHUNK_SIZE
            original_chunk_overlap = self.config.CHUNK_OVERLAP
            
            if chunk_size:
                self.config.CHUNK_SIZE = chunk_size
            if chunk_overlap:
                self.config.CHUNK_OVERLAP = chunk_overlap
                
            try:
                # Extract video ID from URL before downloading
                video_id = video_url.split("watch?v=")[1].split("&")[0]
                
                # If update is True, delete existing entries
                if update:
                    # Check if we have existing data
                    if "video_chunks" in self.db.table_names():
                        # We'll use the updated delete_video_data method
                        existing_data = self.delete_video_data(video_id)
                        if existing_data:
                            logger.info(f"Deleted existing data for video ID: {video_id}")
                
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
        """Search for chunks similar to the query, compatible with LanceDB 0.21.1."""
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode(query)
            
            # Open the table
            table = self.db.open_table("video_chunks")
            
            # Get all data from the table and convert to DataFrame
            df = table.to_pandas()
            
            if df.empty:
                return []
            
            # Convert embeddings to numpy arrays if they're stored as lists
            df['embedding_array'] = df['embedding'].apply(lambda x: np.array(x) if isinstance(x, list) else x)
            
            # Calculate similarities manually
            similarities = []
            for idx, row in df.iterrows():
                # Calculate cosine similarity
                sim = cosine_similarity(query_embedding, row['embedding_array'])
                similarities.append((row, sim))
            
            # Sort by similarity (highest first)
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            # Apply filters if needed
            filtered_results = []
            for row, sim in similarities:
                keep_row = True
                
                # Filter by video_ids if specified
                if video_ids and row['video_id'] not in video_ids:
                    keep_row = False
                    
                # Filter by tags if specified
                if tags and keep_row:
                    row_tags = row.get('tags', [])
                    # Convert numpy arrays to lists if needed
                    if hasattr(row_tags, 'tolist'):
                        row_tags = row_tags.tolist()
                        
                    if not any(tag in row_tags for tag in tags):
                        keep_row = False
                
                if keep_row:
                    filtered_results.append(row)
                    if len(filtered_results) >= limit:
                        break
            
            # Convert results to dict format
            return [row.to_dict() for row in filtered_results]
            
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

            # Import necessary LangChain components
            from langchain.schema import Document
            from langchain.prompts import PromptTemplate

            # Convert chunks to proper LangChain documents
            documents = []
            for chunk in relevant_chunks:
                doc = Document(
                    page_content=chunk["text"],
                    metadata={
                        "source": chunk["title"],
                        "timestamp": chunk["timestamp"],
                        "video_id": chunk["video_id"]
                    }
                )
                documents.append(doc)

            print(f"Found {len(relevant_chunks)} relevant segments. Generating answer...")

            # Set default values if not provided
            temperature = temperature if temperature is not None else self.config.DEFAULT_TEMPERATURE
            max_tokens = max_tokens if max_tokens is not None else self.config.DEFAULT_MAX_TOKENS

            # Create a better prompt template
            prompt_template = """
            You are an expert at explaining concepts from educational videos.
            
            Answer the following question based on the information from the video segments.
            If the information needed isn't contained in the segments, say "The video segments don't cover this topic."
            Provide a comprehensive answer that synthesizes information from all relevant segments.
            
            Question: {question}
            
            Video Content:
            {context}
            
            Answer:
            """

            # Create the prompt
            prompt = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            )

            # Initialize a language model
            llm = ChatOpenAI(
                api_key=self.config.OPENAI_API_KEY,
                model_name=self.config.LLM_MODEL,
                temperature=temperature,
                max_tokens=max_tokens
            )

            # Create a QA chain with our prompt
            qa_chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt)

            # Generate the answer
            result = qa_chain.invoke({"input_documents": documents, "question": question})

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

    def delete_video_data(self, video_id: str) -> bool:
        """Delete all chunks associated with a specific video ID from the database."""
        try:
            # Check if the table exists
            if "video_chunks" not in self.db.table_names():
                logger.warning(f"No video_chunks table exists yet")
                return False
                
            # Open the table
            table = self.db.open_table("video_chunks")
            
            # Get the existing data
            df = table.to_pandas()
            
            # Count rows before deletion
            matching_rows = df[df['video_id'] == video_id]
            count_before = len(matching_rows)
            
            if count_before == 0:
                logger.warning(f"No data found for video ID: {video_id}")
                return False
            
            # Get rows that don't match the video_id
            remaining_df = df[df['video_id'] != video_id]
            
            # Check if all data is being deleted
            if len(remaining_df) == 0:
                logger.info(f"Deleting all data from the table...")
                # If all data is being deleted, drop the table
                self.db.drop_table("video_chunks")
                logger.info(f"Deleted {count_before} chunks for video ID: {video_id}")
                return True
            
            # Otherwise, create a new table with the remaining data
            logger.info(f"Creating backup of existing data...")
            backup_table_name = f"video_chunks_backup_{int(time.time())}"
            self.db.create_table(backup_table_name, data=df)
            
            logger.info(f"Recreating table without deleted chunks...")
            # Drop the original table
            self.db.drop_table("video_chunks")
            
            # Create a new table with the remaining data
            remaining_records = []
            for _, row in remaining_df.iterrows():
                record = row.to_dict()
                # Handle numpy arrays in tags
                if 'tags' in record and hasattr(record['tags'], 'tolist'):
                    record['tags'] = record['tags'].tolist()
                # Handle numpy arrays in embedding
                if 'embedding' in record and hasattr(record['embedding'], 'tolist'):
                    record['embedding'] = record['embedding'].tolist()
                remaining_records.append(VideoChunk(**record))
            
            if remaining_records:
                self.db.create_table("video_chunks", data=remaining_records)
            
            logger.info(f"Deleted {count_before} chunks for video ID: {video_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting video data: {str(e)}")
            return False

    def delete_video_by_url(self, video_url: str) -> bool:
        """
        Delete all chunks associated with a YouTube video URL from the database.
        
        Args:
            video_url: The YouTube video URL to delete
            
        Returns:
            bool: True if successful, False if no data was found
        """
        try:
            # Extract video ID from URL
            video_id = video_url.split("watch?v=")[1].split("&")[0]
            return self.delete_video_data(video_id)
        except Exception as e:
            logger.error(f"Error extracting video ID from URL: {str(e)}")
            return False

    def list_videos(self, tags: List[str] = None) -> pd.DataFrame:
        """List all videos in the database with optional tag filtering."""
        try:
            if "video_chunks" not in self.db.table_names():
                logger.warning("No video_chunks table exists yet")
                return pd.DataFrame()
                
            # Open the table
            table = self.db.open_table("video_chunks")
            
            # Get all data
            df = table.to_pandas()
            
            if df.empty:
                return pd.DataFrame()
                
            # Process the data to get unique video entries
            video_data = {}
            for _, row in df.iterrows():
                video_id = row['video_id']
                title = row['title']
                row_tags = row.get('tags', [])
                
                # Convert numpy arrays to lists if needed
                if hasattr(row_tags, 'tolist'):
                    row_tags = row_tags.tolist()
                    
                # Skip if we're filtering by tags and this video doesn't match
                if tags and not any(tag in row_tags for tag in tags):
                    continue
                    
                if video_id not in video_data:
                    video_data[video_id] = {
                        'video_id': video_id,
                        'title': title,
                        'tags': row_tags,
                        'num_chunks': 1
                    }
                else:
                    video_data[video_id]['num_chunks'] += 1
                    
            # Convert to DataFrame
            results = pd.DataFrame(list(video_data.values())) if video_data else pd.DataFrame()
            return results
            
        except Exception as e:
            logger.error(f"Error listing videos: {str(e)}")
            return pd.DataFrame()    


# Standalone function implementations for lightweight operations
def list_videos_standalone(config, tags=None):
    """List all videos in the database with optional tag filtering without loading models."""
    try:
        db = lancedb.connect(config.LANCEDB_PATH)
        
        if "video_chunks" not in db.table_names():
            print("No video_chunks table exists yet")
            return
            
        # Open the table
        table = db.open_table("video_chunks")
        
        # Get all data from the table
        df = table.to_pandas()
        
        if df.empty:
            print("No videos found")
            return
        
        # Process the data in Python instead of using SQL queries
        # Group by video_id and get the first title for each
        video_data = {}
        for _, row in df.iterrows():
            video_id = row['video_id']
            title = row['title']
            row_tags = row.get('tags', [])
            
            # Convert numpy arrays to lists if needed
            if hasattr(row_tags, 'tolist'):
                row_tags = row_tags.tolist()
            
            # Skip if we're filtering by tags and this video doesn't match
            if tags:
                tags_list = [tag.strip() for tag in tags.split(',')]
                if not any(tag in row_tags for tag in tags_list):
                    continue
            
            if video_id not in video_data:
                video_data[video_id] = {
                    'title': title,
                    'tags': row_tags,
                    'count': 1
                }
            else:
                video_data[video_id]['count'] += 1
        
        if not video_data:
            print("No videos found")
            return
            
        print(f"Found {len(video_data)} videos in database:")
        for i, (video_id, data) in enumerate(video_data.items()):
            tags_str = ", ".join(data['tags']) if data['tags'] else "No tags"
            print(f"{i+1}. {data['title']} (ID: {video_id}) - {tags_str}")
            print(f"   Chunks: {data['count']}")
            
    except Exception as e:
        print(f"Error listing videos: {str(e)}")
        import traceback
        traceback.print_exc()  # Print the full error for debugging

def delete_video_data_standalone(config, video_id):
    """Delete all chunks for a specific video ID without loading models."""
    try:
        db = lancedb.connect(config.LANCEDB_PATH)
        
        if "video_chunks" not in db.table_names():
            print(f"No video_chunks table exists yet")
            return False
            
        # Open the table
        table = db.open_table("video_chunks")
        
        # Get the existing data
        df = table.to_pandas()
        
        # Count rows before deletion
        matching_rows = df[df['video_id'] == video_id]
        count_before = len(matching_rows)
        
        if count_before == 0:
            print(f"No data found for video ID: {video_id}")
            return False
        
        # For older versions of LanceDB, we need to recreate the table without the rows
        # This is inefficient but works for any version
        print(f"Found {count_before} chunks to delete...")
        
        # Get rows that don't match the video_id
        remaining_df = df[df['video_id'] != video_id]
        
        # Check if all data is being deleted
        if len(remaining_df) == 0:
            print("Deleting all data from the table...")
            # If all data is being deleted, drop the table and recreate it empty
            db.drop_table("video_chunks")
            if count_before > 0:
                print(f"Successfully deleted {count_before} chunks for video ID: {video_id}")
            return True
        
        # Otherwise, create a new table with the remaining data
        print("Creating backup of existing data...")
        backup_table_name = f"video_chunks_backup_{int(time.time())}"
        db.create_table(backup_table_name, data=df)
        
        print("Recreating table without deleted chunks...")
        # Drop the original table
        db.drop_table("video_chunks")
        
        # Create a new table with the remaining data
        remaining_records = []
        for _, row in remaining_df.iterrows():
            record = row.to_dict()
            # Handle numpy arrays in tags
            if 'tags' in record and hasattr(record['tags'], 'tolist'):
                record['tags'] = record['tags'].tolist()
            # Handle numpy arrays in embedding
            if 'embedding' in record and hasattr(record['embedding'], 'tolist'):
                record['embedding'] = record['embedding'].tolist()
            remaining_records.append(VideoChunk(**record))
        
        if remaining_records:
            db.create_table("video_chunks", data=remaining_records)
        
        print(f"Successfully deleted {count_before} chunks for video ID: {video_id}")
        return True
        
    except Exception as e:
        print(f"Error deleting video data: {str(e)}")
        import traceback
        traceback.print_exc()  # Print the full error for debugging
        return False

def display_system_info(config):
    """Display system information without loading models."""
    print("YouTube QA Bot - System Information")
    print(f"Configuration:")
    print(f"  Database path: {config.LANCEDB_PATH}")
    print(f"  Audio directory: {config.AUDIO_DIR}")
    print(f"  Chunk size: {config.CHUNK_SIZE}")
    print(f"  Chunk overlap: {config.CHUNK_OVERLAP}")
    print(f"  Default embedding model: {config.EMBEDDING_MODEL}")
    print(f"  Default Whisper model: {config.WHISPER_MODEL}")
    print(f"  LLM model: {config.LLM_MODEL}")
    
    # Check if LanceDB version can be determined
    try:
        lancedb_version = getattr(lancedb, "__version__", "unknown")
        print(f"  LanceDB version: {lancedb_version}")
    except:
        print(f"  LanceDB version: unknown")
    
    # Check if database exists and show stats
    if os.path.exists(config.LANCEDB_PATH):
        db = lancedb.connect(config.LANCEDB_PATH)
        if "video_chunks" in db.table_names():
            table = db.open_table("video_chunks")
            
            try:
                # Get all data to count manually
                df = table.to_pandas()
                total_chunks = len(df)
                unique_videos = df['video_id'].nunique() if not df.empty else 0
                
                print(f"\nDatabase Statistics:")
                print(f"  Total chunks in database: {total_chunks}")
                print(f"  Total unique videos: {unique_videos}")
                
                if unique_videos > 0:
                    # Show some details about the videos
                    print("\nVideos in database:")
                    video_counts = {}
                    for _, row in df.iterrows():
                        video_id = row['video_id']
                        title = row['title']
                        if video_id not in video_counts:
                            video_counts[video_id] = {'title': title, 'count': 1}
                        else:
                            video_counts[video_id]['count'] += 1
                    
                    for video_id, data in video_counts.items():
                        print(f"  - {data['title']} (ID: {video_id}, Chunks: {data['count']})")
                
            except Exception as e:
                print(f"\nError getting statistics: {str(e)}")
                
        else:
            print("\nNo video data in database yet.")
    else:
        print("\nDatabase directory does not exist yet.")


def download_only(config, args):
    """Download video only and prepare for later processing."""
    try:
        video_url = args.url
        import yt_dlp
        
        # Extract video ID from URL
        video_id = video_url.split("watch?v=")[1].split("&")[0]
        
        # Create pending directory if it doesn't exist
        pending_dir = os.path.join(config.AUDIO_DIR, "pending")
        os.makedirs(pending_dir, exist_ok=True)
        
        # Check if audio file already exists
        audio_path = os.path.join(config.AUDIO_DIR, f"{video_id}.mp3")
        pending_file = os.path.join(pending_dir, f"{video_id}.json")
        
        force = args.force if hasattr(args, 'force') else False
        
        if os.path.exists(audio_path) and not force:
            print(f"Audio file already exists: {audio_path}")
            # We need to get the title separately
            with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
                info = ydl.extract_info(video_url, download=False)
                title = info.get('title', 'Unknown Title')
            print(f"\nUsing existing audio file for: {title}")
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
        
        # Process tags if provided
        tags = None
        if hasattr(args, 'tags') and args.tags:
            tags = [tag.strip() for tag in args.tags.split(',')]
        
        # Create metadata for later processing
        metadata = {
            'video_id': video_id,
            'title': title,
            'url': video_url,
            'date_downloaded': time.strftime('%Y-%m-%d %H:%M:%S'),
            'whisper_model': args.whisper_model if hasattr(args, 'whisper_model') and args.whisper_model else config.WHISPER_MODEL,
            'chunk_size': args.chunk_size if hasattr(args, 'chunk_size') and args.chunk_size else config.CHUNK_SIZE,
            'chunk_overlap': args.chunk_overlap if hasattr(args, 'chunk_overlap') and args.chunk_overlap else config.CHUNK_OVERLAP,
            'update': not args.no_update if hasattr(args, 'no_update') else True,
            'tags': tags
        }
        
        # Save metadata for later processing
        with open(pending_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Video queued for processing: {title}")
        print(f"To complete processing, run: python process_later.py --video-id {video_id}")
        print(f"Or process all pending videos: python process_later.py --all")
        
    except Exception as e:
        print(f"Error downloading video: {str(e)}")
        import traceback
        traceback.print_exc()  # Print the full error for debugging


def main():
    parser = argparse.ArgumentParser(description="YouTube Search QA Bot")
    parser.add_argument("--fast", action="store_true", help="Run in fast mode with minimal loading")
    
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
    add_parser.add_argument("--no-update", action="store_true", 
                           help="Don't update existing entries (add new ones instead)")
    
    # Query command
    query_parser = subparsers.add_parser("query", help="Query the video database")
    query_parser.add_argument("question", help="Your question")
    query_parser.add_argument("--results", type=int, default=5, help="Number of results to use")
    query_parser.add_argument("--videos", help="Comma-separated list of video IDs to search within")
    query_parser.add_argument("--tags", help="Comma-separated list of tags to filter by")
    query_parser.add_argument("--temperature", type=float, help="Temperature for the language model (0.0-1.0)")
    query_parser.add_argument("--max-tokens", type=int, help="Maximum tokens for the response")
    
    # Delete command
    delete_parser = subparsers.add_parser("delete", help="Delete video data from the database")
    delete_parser.add_argument("--video-id", help="YouTube video ID to delete")
    delete_parser.add_argument("--url", help="YouTube video URL to delete")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List videos in the database")
    list_parser.add_argument("--tags", help="Filter by comma-separated list of tags")
    
    # Info command (new)
    info_parser = subparsers.add_parser("info", help="Display system information")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Initialize config
    config = Config()
    
    # Check for fast mode
    fast_mode = args.fast if hasattr(args, 'fast') else False
    
    # Commands that can use lightweight processing
    fast_compatible_commands = ["list", "delete", "info"]
    
    if args.command in fast_compatible_commands or fast_mode:
        # Process commands that don't need the full models loaded
        if args.command == "list":
            list_videos_standalone(config, tags=args.tags if hasattr(args, 'tags') and args.tags else None)
        
        elif args.command == "delete":
            if hasattr(args, 'video_id') and args.video_id:
                success = delete_video_data_standalone(config, args.video_id)
                if success:
                    print(f"Successfully deleted data for video ID: {args.video_id}")
                else:
                    print(f"No data found for video ID: {args.video_id}")
            elif hasattr(args, 'url') and args.url:
                try:
                    video_id = args.url.split("watch?v=")[1].split("&")[0]
                    success = delete_video_data_standalone(config, video_id)
                    if success:
                        print(f"Successfully deleted data for video URL: {args.url}")
                    else:
                        print(f"Failed to delete data for video URL: {args.url}")
                except Exception as e:
                    print(f"Error extracting video ID from URL: {str(e)}")
            else:
                print("Please provide either --video-id or --url to delete")
        
        elif args.command == "info":
            # Show system information without loading models
            display_system_info(config)
        
        # Handle add and query in fast mode with limited functionality
        elif args.command == "add" and fast_mode:
            print("Running 'add' in fast mode - only downloading the video without processing.")
            print("Run process_later.py afterward to complete the processing.")
            # Perform download-only operation here
            download_only(config, args)
        
        elif args.command == "query" and fast_mode:
            print("Warning: Running query in fast mode is not supported.")
            print("For queries, a full processor initialization is required.")
            print("Run without --fast flag for query operations.")
    else:
        # For commands requiring the full processor, initialize it as normal
        processor = YouTubeProcessor(config, 
                                   whisper_model=args.whisper_model if hasattr(args, 'whisper_model') else None)
        
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
                chunk_overlap=args.chunk_overlap if hasattr(args, 'chunk_overlap') else None,
                update=not args.no_update if hasattr(args, 'no_update') else True
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