import os
from typing import Any, Dict, List, Optional, Tuple

import chromadb
from chromadb.utils import embedding_functions


class VectorStore:
    def __init__(
        self,
        collection_name: str = "meeting_transcripts",
        persist_directory: str = ".chroma_db",
        embedding_function: Optional[Any] = None,
    ):
        """
        Initialize a vector store using ChromaDB

        Args:
            collection_name: Name of the collection to use
            persist_directory: Directory to persist the database
            embedding_function: Function to use for embeddings (defaults to OpenAI)
        """
        # Create or load the client
        self.client = chromadb.PersistentClient(path=persist_directory)

        # Set up the embedding function
        if embedding_function is None:
            # Use OpenAI as the default embedding function
            self.embedding_function = embedding_functions.OpenAIEmbeddingFunction(
                api_key=os.environ.get("OPENAI_API_KEY"),
                model_name="text-embedding-ada-002",
            )
        else:
            self.embedding_function = embedding_function

        # Get or create the collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name, embedding_function=self.embedding_function
        )

    def add_transcript_chunks(
        self, job_id: str, transcript_chunks: List[Dict[str, Any]]
    ) -> None:
        """
        Add transcript chunks to the vector store

        Args:
            job_id: Unique identifier for the meeting
            transcript_chunks: List of transcript chunks with text and timestamps
        """
        # Convert transcript chunks to format for ChromaDB
        ids = [f"{job_id}_{i}" for i in range(len(transcript_chunks))]
        texts = [chunk["text"] for chunk in transcript_chunks]
        metadatas = [
            {"job_id": job_id, "start_time": chunk["start"], "end_time": chunk["end"]}
            for chunk in transcript_chunks
        ]

        # Add to collection
        self.collection.add(ids=ids, documents=texts, metadatas=metadatas)

    def search(
        self, query: str, job_id: str, top_k: int = 3
    ) -> List[Tuple[float, float, str]]:
        """
        Search for transcript chunks that match the query

        Args:
            query: Search query
            job_id: Job ID to filter results
            top_k: Number of results to return

        Returns:
            List of tuples with (start_time, end_time, text)
        """
        # Query the collection with where filter to get only documents for this job_id
        results = self.collection.query(
            query_texts=[query], where={"job_id": job_id}, n_results=top_k
        )

        # Format results
        formatted_results = []

        if not results["documents"] or not results["documents"][0]:
            return []

        for i, doc in enumerate(results["documents"][0]):
            start_time = results["metadatas"][0][i]["start_time"]
            end_time = results["metadatas"][0][i]["end_time"]
            text = doc
            formatted_results.append((start_time, end_time, text))

        return formatted_results

    def delete_job(self, job_id: str) -> None:
        """
        Delete all chunks for a specific job

        Args:
            job_id: Job ID to delete
        """
        self.collection.delete(where={"job_id": job_id})

    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the collection

        Returns:
            Dictionary with collection statistics
        """
        count = self.collection.count()
        return {"name": self.collection.name, "count": count}
