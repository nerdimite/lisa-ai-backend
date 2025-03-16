import os
from typing import Any, Dict, List, Tuple

from lisa.llm_utils import (
    action_items,
    create_embeddings,
    create_transcript_chunks,
    create_transcript_str,
    minutes_of_meeting,
    transcribe_audio,
)
from lisa.vector_store import VectorStore


class LISAPipeline:
    def __init__(self, vector_store: VectorStore = None):
        """
        Initialize LISA Pipeline

        Args:
            vector_store: Vector store instance for transcript search
        """
        # Initialize vector store
        if vector_store is None:
            self.vector_store = VectorStore()
        else:
            self.vector_store = vector_store

    def __call__(
        self, audio_path: str
    ) -> Tuple[List[str], List[str], List[Dict[str, Any]], str]:
        """
        Run the pipeline on an audio file

        Args:
            audio_path: Path to the audio file

        Returns:
            Tuple containing:
            - minutes: List of meeting minute points
            - action_items: List of action items
            - transcript_chunks: List of transcript chunks
            - transcript_str: Complete transcript as a string
        """
        # Transcribe audio
        whisper_response = transcribe_audio(audio_path)

        # Convert to transcript string
        transcript_str = create_transcript_str(whisper_response)

        # Generate meeting minutes
        minutes = minutes_of_meeting(transcript_str)
        print(minutes)

        # Extract action items
        actions = action_items(transcript_str)
        print(actions)

        # Create transcript chunks for search
        transcript_chunks = create_transcript_chunks(whisper_response)

        return minutes, actions, transcript_chunks, transcript_str

    def store_transcript(
        self, job_id: str, transcript_chunks: List[Dict[str, Any]]
    ) -> None:
        """
        Store transcript chunks in vector store

        Args:
            job_id: Unique identifier for the meeting
            transcript_chunks: List of transcript chunks
        """
        self.vector_store.add_transcript_chunks(job_id, transcript_chunks)

    def search(
        self, job_id: str, query: str, top_k: int = 3
    ) -> List[Tuple[float, float, str]]:
        """
        Search for transcript chunks that match the query

        Args:
            job_id: Job ID to search within
            query: Search query
            top_k: Number of results to return

        Returns:
            List of tuples with (start_time, end_time, text)
        """
        return self.vector_store.search(query, job_id, top_k)
