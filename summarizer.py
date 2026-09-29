"""Two summaries of a transcript: an LLM (abstractive) one and a TF-IDF (extractive) one."""
import os

from dotenv import load_dotenv
from google import genai
from google.genai import errors, types
from sklearn.feature_extraction.text import TfidfVectorizer

load_dotenv()

GEMINI_MODEL = "gemini-3.8-flash"
# Tried in order when the main model is overloaded (503) or rate-limited (429).
FALLBACK_MODELS = ["gemini-3.5-flash", "gemini-3.5-flash-lite", "gemini-3.1-flash-lite"]

PROMPT = """Summarize this YouTube transcript in English (the transcript may be in another language).
Give a 2-line overview, then 5 key points as bullets, then a one-line takeaway.
Format the answer in Markdown with the headings "Overview", "Key points" and "Takeaway".

Transcript:
{text}"""

_client = None


def llm_summary(text):
    """Abstractive summary: the LLM writes new sentences."""
    global _client
    if _client is None:
        # Don't let the SDK retry an overloaded model with backoff (that took 90s+ in
        # testing); fail fast and move on to the next model in the fallback list.
        _client = genai.Client(
            api_key=os.getenv("GEMINI_API_KEY"),
            http_options=types.HttpOptions(
                timeout=45_000,  # ms
                retry_options=types.HttpRetryOptions(attempts=1),  # no retries: fail fast
            ),
        )
    prompt = PROMPT.format(text=text)
    last_error = None
    for model in [GEMINI_MODEL, *FALLBACK_MODELS]:
        try:
            return _client.models.generate_content(model=model, contents=prompt).text
        except Exception as e:
            # A bad request or API key (4xx other than 429) won't be fixed by another model.
            if isinstance(e, errors.ClientError) and e.code != 429:
                raise
            print(f"[llm_summary] {model} failed ({type(e).__name__}), trying next model")
            last_error = e
    raise last_error


def chunk_words(text, size=25):
    """Auto-generated transcripts have no punctuation, so split every `size` words."""
    words = text.split()
    return [" ".join(words[i:i + size]) for i in range(0, len(words), size)]


def extractive_summary(text, n_chunks=5, chunk_size=25):
    """Extractive summary: pick the chunks with the highest total TF-IDF weight."""
    chunks = chunk_words(text, chunk_size)
    if len(chunks) <= n_chunks:
        return " ".join(chunks)

    tfidf = TfidfVectorizer(stop_words="english")
    matrix = tfidf.fit_transform(chunks)
    scores = matrix.sum(axis=1).A1  # one score per chunk

    top = scores.argsort()[::-1][:n_chunks]
    top_in_order = sorted(top)  # re-sort by original position so it reads in order
    return "\n\n".join(f"• {chunks[i]}" for i in top_in_order)
