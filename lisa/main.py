import base64
import json
import os
from contextlib import asynccontextmanager
from typing import List, Tuple

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from moviepy import VideoFileClip
from pydantic import BaseModel

from lisa.inference import LISAPipeline
from lisa.random_uid import generate_random_uid
from lisa.vector_store import VectorStore


# Models for request/response
class VideoInput(BaseModel):
    video: str


class SearchQuery(BaseModel):
    job_id: str
    query: str


class PredictResponse(BaseModel):
    job_id: str
    minutes: List[str]
    action_items: List[str]
    transcription: str


class SearchResponse(BaseModel):
    results: List[tuple]


# Initialize components
vector_store = VectorStore()
lisa_pipeline = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize on startup
    global lisa_pipeline
    lisa_pipeline = LISAPipeline(vector_store=vector_store)
    yield
    # Cleanup on shutdown (if needed)


# Create FastAPI app
app = FastAPI(
    title="LISA API",
    description="Meeting Transcription and Analysis API",
    version="2.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/predict", response_model=PredictResponse)
async def predict(payload: VideoInput):
    """
    Process a meeting video:
    1. Transcribe audio
    2. Generate meeting minutes
    3. Extract action items
    4. Store transcript for search
    """
    try:
        print("Received Video Payload")

        # Create unique job ID
        while True:
            job_id = generate_random_uid()
            job_dir = f"results/{job_id}"
            if not os.path.exists(job_dir):
                os.makedirs(job_dir, exist_ok=True)
                break

        # Save video
        with open(os.path.join(job_dir, "meeting.mp4"), "wb") as f:
            video_bytes = base64.decodebytes(
                payload.video.split(",")[1].encode("utf-8")
            )
            f.write(video_bytes)

        # Convert video to audio
        video = VideoFileClip(os.path.join(job_dir, "meeting.mp4"))
        audio_path = os.path.join(job_dir, "meeting.mp3")
        video.audio.write_audiofile(audio_path)

        # Run inference
        minutes, action_items, transcript_chunks, transcript_str = lisa_pipeline(
            audio_path
        )

        # Store transcript chunks for search
        lisa_pipeline.store_transcript(job_id, transcript_chunks)

        # Save results
        results = {
            "job_id": job_id,
            "minutes": minutes,
            "action_items": action_items,
            "transcript_chunks": transcript_chunks,
            "transcription": transcript_str,
        }

        with open(os.path.join(job_dir, "results.json"), "w") as f:
            json.dump(results, f)

        output = {
            "job_id": job_id,
            "minutes": minutes,
            "action_items": action_items,
            "transcription": transcript_str,
        }
        print(f"Processing complete for job: {job_id}")

        return output

    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/results/{job_id}")
async def fetch_results(job_id: str):
    """
    Fetch results for a specific job ID
    """
    try:
        results_path = os.path.join("results", job_id, "results.json")

        if not os.path.exists(results_path):
            raise HTTPException(
                status_code=404, detail=f"Results for job {job_id} not found"
            )

        with open(results_path, "r") as f:
            results = json.load(f)

        return results

    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search", response_model=List[Tuple[float, float, str]])
async def search(search_query: SearchQuery):
    """
    Search transcript chunks for a specific query
    """
    try:
        job_id = search_query.job_id
        query = search_query.query

        # Search using the pipeline
        search_results = lisa_pipeline.search(job_id, query)

        return search_results

    except Exception as e:
        print(f"Error in search: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    return RedirectResponse(url="/docs")


def main():
    import uvicorn

    uvicorn.run("lisa.main:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    main()
