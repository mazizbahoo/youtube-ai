"""LLM (Gemini) summary of a transcript."""
import os

from dotenv import load_dotenv
from google import genai
from google.genai import errors, types

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

