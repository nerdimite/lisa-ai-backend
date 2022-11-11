from words import ADJECTIVES, POLITE_ADJECTIVES, NOUNS, ANIMALS

def format_timestamp(seconds: float, always_include_hours: bool = False, decimal_marker: str = '.'):
    assert seconds >= 0, "non-negative timestamp expected"
    milliseconds = round(seconds * 1000.0)

    hours = milliseconds // 3_600_000
    milliseconds -= hours * 3_600_000

    minutes = milliseconds // 60_000
    milliseconds -= minutes * 60_000

    seconds = milliseconds // 1_000
    milliseconds -= seconds * 1_000

    hours_marker = f"{hours:02d}:" if always_include_hours or hours > 0 else ""
    return f"{hours_marker}{minutes:02d}:{seconds:02d}{decimal_marker}{milliseconds:03d}"

def create_transcript_str(whisper_out):
        
    transcript_str = []
    for segment in whisper_out['segments']:
        transcript_str.append(f"[{format_timestamp(segment['start'])}]:\t{segment['text']}")
    
    transcript_str = "\n".join(transcript_str)
    return transcript_str

def postprocess_points(raw_output):
    points = raw_output.split('\n-')
    points = [point.strip() for point in points]
    points = [point for point in points if point != '']
    return points

def find_closest_time(time, all_times):
    closest_time = min(all_times, key=lambda x: abs(x - time))
    return closest_time

def create_transcript_chunks(all_start_times, all_end_times, whisper_out, stride=45, length=60):
    '''Create larger chunks of the segments using a sliding window'''

    transcript_chunks = []
    for seek in range(0, int(all_end_times[-1]), stride):
        chunk = {'start': None, 'end': None, 'text': None}

        start_index = all_start_times.index(find_closest_time(seek, all_start_times))
        chunk['start'] = all_start_times[start_index]
        end_index = all_end_times.index(find_closest_time(seek + length, all_end_times))
        chunk['end'] = all_end_times[end_index]

        chunk['text'] = "".join([segment['text'] for segment in whisper_out['segments'][start_index:end_index+1]]).strip()

        transcript_chunks.append(chunk)
    
    return transcript_chunks