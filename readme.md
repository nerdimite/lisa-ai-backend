# LISA - AI Meeting Assistant

LISA (Live Intelligent Speech Analysis) is an AI-powered meeting assistant that:

1. Transcribes meeting recordings using OpenAI's Whisper API
2. Generates concise meeting minutes
3. Extracts action items when triggered with "Hey Lisa" in the meeting
4. Provides semantic search capabilities for meeting content

## Features

- **Transcription**: Convert meeting audio/video to searchable text
- **Meeting Minutes**: Automatically generate concise meeting summaries
- **Action Items**: Extract to-do items mentioned during the meeting
- **Semantic Search**: Search through meeting transcripts with natural language queries
- **Vector Storage**: Persists embeddings using ChromaDB for efficient retrieval

## How It Works

LISA follows a streamlined pipeline to process meeting recordings:

1. **Video Processing**: Extracts audio from uploaded meeting videos
2. **Transcription**: Uses OpenAI's Whisper API to convert speech to text with timestamps
3. **Analysis**: Processes the transcript with GPT models to:
   - Generate concise meeting minutes
   - Extract action items when "Hey Lisa" is mentioned
4. **Semantic Indexing**: Creates embeddings of transcript chunks
5. **Storage**: Stores embeddings in ChromaDB for efficient semantic search
6. **Retrieval**: Enables natural language queries to find relevant meeting moments

## Installation

LISA uses Poetry for dependency management:

```bash
# Install dependencies with Poetry
poetry install
```

Set OpenAI API key as an environment variable:

- Linux/MacOS: `export OPENAI_API_KEY=<your-api-key>`
- Windows: `set OPENAI_API_KEY=<your-api-key>`

Or create a `.env` file in the project root:

```
OPENAI_API_KEY=<your-api-key>
```

## Usage

### Starting the Server

```bash
# Using Poetry
poetry run start
```

The server will start at http://localhost:8000

### API Endpoints

- **POST /predict**: Process a meeting recording and generate transcription, minutes, and action items
- **GET /results/{job_id}**: Retrieve results for a processed meeting
- **POST /search**: Search within a meeting transcript

### Example API Calls

**Processing a meeting video:**

```python
import requests
import base64

# Read video file as base64
with open("meeting.mp4", "rb") as f:
    video_base64 = base64.b64encode(f.read()).decode("utf-8")

# Call the API
response = requests.post(
    "http://localhost:8000/predict",
    json={"video": f"data:video/mp4;base64,{video_base64}"}
)

# Get job ID from response
job_id = response.json()["job_id"]
```

**Searching a meeting transcript:**

```python
import requests

# Search for mentions of "project deadline"
response = requests.post(
    "http://localhost:8000/search",
    json={"job_id": "polite-elephant-123", "query": "project deadline"}
)

# Search results with timestamps
results = response.json()
```
