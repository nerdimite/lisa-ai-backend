import openai
import whisper
from sentence_transformers import SentenceTransformer, util
from utils import create_transcript_chunks, create_transcript_str, postprocess_points

openai.api_key = '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' # os.getenv("OPENAI_API_KEY")

class LISAPipeline():
    def __init__(self, whisper_model, search_model):
        print('Loading Whisper...')
        self.whisper_model = whisper.load_model(whisper_model)
        print('Loading Sentence Transformer...')
        self.search_model = SentenceTransformer.load(search_model)
        print('Models loaded!')
    
    def run_gpt3(self, prompt, max_tokens=256, temperature=0.5, top_p=1, frequency_penalty=0.0, presence_penalty=0.0):
        response = openai.Completion.create(
            engine="text-davinci-002",
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty
        )
        return response.choices[0].text
    
    def transcribe(self, audio_path):
        whisper_out = self.whisper_model.transcribe(audio_path, verbose=False)
        return whisper_out
    
    def minutes_of_meeting(self, transcript_str):

        mom_prompt = f"""Generate the minutes of the meeting for the following transcript:
        Meeting Transcription:
        {transcript_str}

        Meeting Minutes:
        -"""

        raw_minutes = '\n-' + self.run_gpt3(mom_prompt, temperature=0.5)
        minutes = postprocess_points(raw_minutes)

        return minutes

    def action_items(self, transcript_str):

        if 'hey lisa' not in transcript_str.lower():
            return []

        action_prompt = f"""Extract the Action Items / To-Do List from the Transcript.
        Meeting Transcription:
        {transcript_str}

        Action Items:
        -"""
        raw_action_items = self.run_gpt3(action_prompt, temperature=0.4)
        action_items = postprocess_points(raw_action_items)

        return action_items

    def create_index(self, whisper_out):
        '''Create search index by embedding the transcript segments'''
        all_start_times = [segment['start'] for segment in whisper_out['segments']]
        all_end_times = [segment['end'] for segment in whisper_out['segments']]

        transcript_chunks = create_transcript_chunks(all_start_times, all_end_times, whisper_out, stride=45, length=60)

        # Encode query and documents
        chunk_texts = [chunk['text'] for chunk in transcript_chunks]
        doc_emb = self.search_model.encode(chunk_texts)

        return doc_emb, transcript_chunks
    
    def search(self, doc_embeddings, transcript_chunks, query, top_k=3, threshold=16):
        # Compute dot score between query and all document embeddings
        query_embeddings = self.search_model.encode(query)
        scores = util.dot_score(query_embeddings, doc_embeddings)[0].cpu().tolist()

        chunks = [(chunk['start'], chunk['end'], chunk['text']) for chunk in transcript_chunks]

        # Combine docs & scores
        chunk_score_tuples = [(*chunks[i], scores[i]) for i in range(len(chunks))]

        # Sort by decreasing score
        chunk_score_tuples = sorted(chunk_score_tuples, key=lambda x: x[-1], reverse=True)

        # Output passages & scores
        results = []
        for start, end, text, score in chunk_score_tuples[:top_k]:
            if score > threshold:
                results.append((start, end, text))

        return results

    def __call__(self, audio_path):
        '''Run the pipeline on an audio file'''
        whisper_out = self.transcribe(audio_path)
        transcript_str = create_transcript_str(whisper_out)
        minutes = self.minutes_of_meeting(transcript_str)
        action_items = self.action_items(transcript_str)
        doc_emb, transcript_chunks = self.create_index(whisper_out)

        return minutes, action_items, doc_emb, transcript_chunks