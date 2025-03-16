import json
import os
from typing import Any, Dict, List, Optional

from jinja2 import Template
from openai import OpenAI

# Initialize OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


def transcribe_audio(
    audio_path: str,
    response_format: str = "verbose_json",
    timestamp_granularities: List[str] = ["segment"],
) -> Dict[str, Any]:
    """
    Transcribe audio using OpenAI's Whisper API

    Args:
        audio_path: Path to the audio file
        response_format: Format for the response (verbose_json, json, text, srt, vtt)
        timestamp_granularities: Level of timestamp detail (segment, word)

    Returns:
        Transcription result
    """
    with open(audio_path, "rb") as audio_file:
        response = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            response_format=response_format,
            timestamp_granularities=timestamp_granularities,
        )

    return response


def create_transcript_str(transcript_response) -> str:
    """
    Convert the Whisper API response to a formatted transcript string

    Args:
        transcript_response: Response from Whisper API

    Returns:
        Formatted transcript string
    """
    transcript_str = []

    # Handle different response formats
    if hasattr(transcript_response, "segments"):
        # verbose_json format
        for segment in transcript_response.segments:
            transcript_str.append(
                f"[{format_timestamp(segment.start)}]:\t{segment.text}"
            )
    else:
        # Simple text format
        return transcript_response.text

    return "\n".join(transcript_str)


def format_timestamp(
    seconds: float, always_include_hours: bool = False, decimal_marker: str = "."
) -> str:
    """Format seconds into a timestamp string"""
    assert seconds >= 0, "non-negative timestamp expected"
    milliseconds = round(seconds * 1000.0)

    hours = milliseconds // 3_600_000
    milliseconds -= hours * 3_600_000

    minutes = milliseconds // 60_000
    milliseconds -= minutes * 60_000

    seconds = milliseconds // 1_000
    milliseconds -= seconds * 1_000

    hours_marker = f"{hours:02d}:" if always_include_hours or hours > 0 else ""
    return (
        f"{hours_marker}{minutes:02d}:{seconds:02d}{decimal_marker}{milliseconds:03d}"
    )


def generate_completion(
    prompt: str,
    model: str = "gpt-4o-mini",
    **kwargs,
) -> str:
    """
    Generate text completion using OpenAI API

    Args:
        prompt: The prompt to generate from
        model: The model to use
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature

    Returns:
        Generated text
    """
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful Meeting AI assistant."},
            {"role": "user", "content": prompt},
        ],
        **kwargs,
    )

    return response.choices[0].message.content.strip()


def minutes_of_meeting(transcript_str: str) -> List[str]:
    """
    Generate minutes of meeting from transcript

    Args:
        transcript_str: The meeting transcript

    Returns:
        List of minute points
    """
    mom_prompt = Template(
        """Generate concise and informative minutes of the meeting from the following transcript.
        Focus on key decisions, important discussions, and main topics covered.
        Organize points in order of significance, not necessarily chronological order.
        Exclude small talk, greetings, and irrelevant discussions.
        
        Meeting Transcription:
        {{transcript_str}}

        Return the Meeting Minutes as a JSON list of strings with the following format:
        {
            "minutes": [
                "Minute 1",
                "Minute 2",
                "Minute 3"
            ]
        }
        
        Ensure each point is clear, specific, and provides valuable information about the meeting.
        """
    ).render(transcript_str=transcript_str)

    raw_minutes = generate_completion(
        mom_prompt, temperature=0.5, response_format={"type": "json_object"}
    )
    minutes = json.loads(raw_minutes)["minutes"]

    return minutes


def action_items(transcript_str: str) -> List[str]:
    """
    Extract action items from transcript

    Args:
        transcript_str: The meeting transcript

    Returns:
        List of action items
    """
    if "hey lisa" not in transcript_str.lower():
        return []

    action_prompt = Template(
        """Extract the Action Items / To-Do List from the Transcript.
        Focus on tasks that were explicitly assigned during the meeting.
        Look for phrases like 'will do', 'needs to', 'should complete', 'is responsible for', etc.
        Only include actionable items with clear ownership or deadlines when possible.
        If there are no action items, return an empty list.
        
        Meeting Transcription:
        {{transcript_str}}

        Return the Action Items as a JSON list of strings with the following format:
        {
            "action_items": [
                "Action Item 1",
                "Action Item 2",
                "Action Item 3"
            ]
        }
        """
    ).render(transcript_str=transcript_str)

    raw_action_items = generate_completion(
        action_prompt, temperature=0.3, response_format={"type": "json_object"}
    )
    action_items = json.loads(raw_action_items)["action_items"]

    return action_items


def create_embeddings(texts: List[str]) -> List[List[float]]:
    """
    Create embeddings for a list of texts using OpenAI's embeddings API

    Args:
        texts: List of text strings to embed

    Returns:
        List of embedding vectors
    """
    response = client.embeddings.create(model="text-embedding-ada-002", input=texts)

    # Extract embeddings from the response
    embeddings = [item.embedding for item in response.data]

    return embeddings


def create_transcript_chunks(
    whisper_response, stride: int = 45, length: int = 60
) -> List[Dict[str, Any]]:
    """
    Create chunks of transcript for search indexing

    Args:
        whisper_response: Response from Whisper API
        stride: Step size for sliding window (seconds)
        length: Window size (seconds)

    Returns:
        List of transcript chunks with start time, end time, and text
    """
    transcript_chunks = []

    # Process based on response format
    if hasattr(whisper_response, "segments"):
        segments = whisper_response.segments
        all_start_times = [segment.start for segment in segments]
        all_end_times = [segment.end for segment in segments]

        for seek in range(0, int(all_end_times[-1]), stride):
            chunk = {"start": None, "end": None, "text": None}

            # Find closest start time
            start_index = all_start_times.index(
                find_closest_time(seek, all_start_times)
            )
            chunk["start"] = all_start_times[start_index]

            # Find closest end time
            end_index = all_end_times.index(
                find_closest_time(seek + length, all_end_times)
            )
            chunk["end"] = all_end_times[end_index]

            # Extract text for this chunk
            chunk["text"] = "".join(
                [segment.text for segment in segments[start_index : end_index + 1]]
            ).strip()

            transcript_chunks.append(chunk)

    return transcript_chunks


def find_closest_time(time: float, all_times: List[float]) -> float:
    """Find the closest time in a list of times"""
    closest_time = min(all_times, key=lambda x: abs(x - time))
    return closest_time
