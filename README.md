# youtube-ai

AI-powered YouTube analyzer that **summarizes videos**, **recommends related content**, and **analyzes comment sentiment**. Built with Python, Gradio, Transformers, spaCy, and Gemini.

> **Status:** all three features work end to end.

---

## Features

| Tab | Input | What it does |
|-----|-------|--------------|
| **Summarize** | A YouTube link | Fetches the transcript and shows the video's thumbnail and an **LLM summary** (Gemini). |
| **Recommend** | A topic *or* a YouTube link | Searches YouTube, embeds every result with a sentence-transformer, and reranks by cosine similarity to your query (nearest neighbours in vector space). With a link, it runs in "more like this video" mode. |
| **Sentiment** | A YouTube link | Pulls up to 200 top comments, classifies them as positive / neutral / negative with a RoBERTa model, and charts the breakdown. spaCy pulls out the nouns that people praising and complaining mention most. |

## Tech stack

- **UI:** [Gradio](https://www.gradio.app/)
- **Data:** YouTube Data API v3 (`google-api-python-client`), `youtube-transcript-api`
- **LLM:** Google Gemini (`google-genai`)
- **Sentiment:** `cardiffnlp/twitter-roberta-base-sentiment-latest` via 🤗 Transformers + PyTorch
- **Embeddings:** `all-MiniLM-L6-v2` via `sentence-transformers`
- **NLP:** spaCy (`en_core_web_sm`)
- **Analysis and charts:** Pandas, Plotly

## Project structure

```
youtube-ai/
├── app.py              # Gradio UI: three tabs plus the glue functions
├── youtube_utils.py    # Video ID parsing, transcripts, search, comments, video metadata
├── sentiment.py        # Comment cleaning, sentiment classification, charts, spaCy keywords
├── summarizer.py       # Gemini summary
├── recommender.py      # Embedding-based reranking of search results
├── requirements.txt    # Pinned versions (pip freeze)
├── .env                # API keys (not committed)
└── README.md
```

## Getting started

### 1. Clone and create a virtual environment

```bash
git clone https://github.com/mazizbahoo/youtube-ai.git
cd youtube-ai
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

`requirements.txt` pins exact versions and includes the spaCy English model (`en_core_web_sm`). If the model is still missing, run `python -m spacy download en_core_web_sm`.

Check the install with:

```bash
python -c "import gradio, transformers, spacy; print('ok')"
```

### 3. Get API keys (both free)

- **YouTube Data API v3:** [Google Cloud Console](https://console.cloud.google.com/) → create a project → *APIs & Services* → enable **YouTube Data API v3** → *Credentials* → *Create API key*.
- **Gemini:** [Google AI Studio](https://aistudio.google.com/) → *Get API key*.

### 4. Create a `.env` file in the project root

```env
YOUTUBE_API_KEY=your-youtube-key
GEMINI_API_KEY=your-gemini-key
```

`.env` is already in `.gitignore`. Never commit your keys.

### 5. Run the app

```bash
python app.py
```

Then open the local URL Gradio prints (usually http://127.0.0.1:7860).

> ⏳ **The first run is slow.** The sentiment model (~500 MB), the embedding model (~90 MB), and PyTorch weights download on first use. Later runs load from the local cache.

## How it works

### Summarize

1. `get_video_id()` pulls the 11-character ID (`[A-Za-z0-9_-]{11}`) from `watch?v=`, `youtu.be/`, `/shorts/` and `/embed/` links with regex. It returns `None` for anything else.
2. `get_transcript()` lists the video's caption tracks and prefers English. If there's no English track, it falls back to whatever language exists.
3. **LLM summary:** the whole transcript goes to Gemini in a single prompt (the large context window means no chunking). The prompt asks for a 2-line overview, 5 key points, and a one-line takeaway, always in English.
   - Free-tier Gemini models are often overloaded (`503`) or rate-limited (`429`). The SDK's slow automatic retries are turned off. Instead, `llm_summary()` moves straight on to the next model in `FALLBACK_MODELS`.
   - If every model fails, the error is shown in the summary panel.

### Recommend

1. `search_videos(query, 25)` gets candidates from the YouTube search API.
2. Each video becomes one text: `title + " " + description`.
3. The query and all video texts are encoded with `all-MiniLM-L6-v2`.
4. `util.cos_sim()` scores each video and the top 5 are returned. This is **KNN**: nearest neighbours in embedding space.
5. **"More like this video" mode:** when the input is a link, the video's title becomes the search query, its title + description becomes the comparison vector, and the original video is removed from the results.

Results render as HTML cards with a thumbnail, a clickable title, the match percentage, and the video's original YouTube rank, so you can see how the reranking changed the order.

### Sentiment

1. `get_comments()` pages through `commentThreads().list(...)` (100 per call) until it reaches 200 comments or runs out of pages.
2. Regex cleaning replaces URLs with `http` and `@username` with `@user`, the format the Twitter-RoBERTa model was trained on. Emojis are kept on purpose because they carry a lot of sentiment.
3. The model classifies comments in batches of 16, and the results go into a DataFrame with `comment`, `label`, and `score` columns.
   - Each batch is padded to its longest comment. Classifying the comments in order of length keeps batches uniform, which made this step about 4× faster on CPU (roughly 90s → 23s for 200 comments).
   - The predictions are then put back in the original comment order.
4. Outputs: an overall verdict (the most common label), a Plotly donut chart of the labels, a histogram of confidence scores, and the full comment table.
5. **Keywords:** positive and negative comments go through spaCy separately. The app keeps lemmas of non-stopword nouns and uses `Counter` to find the top 10 for each group, for example *"people complaining mention: audio, ads, length."*

## Notes and gotchas

- **YouTube API quota:** 10,000 units per day. A `search` costs **100 units**, and a comments or videos call costs about **1**. Don't put search in a loop while testing.
- **Transcript API versions:** the code uses the 1.x syntax (`YouTubeTranscriptApi().list(...)` / `.fetch()`). If those don't exist, you have an old version, so upgrade it.
- **Gemini model names change often.** `gemini-2.5-flash` is no longer available to new users. The app uses `gemini-3.8-flash` with fallbacks (see `GEMINI_MODEL` and `FALLBACK_MODELS` in `summarizer.py`). If a name stops working, check AI Studio for the current free models.
- **API keys in logs:** a `googleapiclient` `HttpError` message includes the request URL, and that URL contains your YouTube API key. The code prints only the status code and reason, never the whole error.
- **Load models once.** The sentiment pipeline and the embedding model are created at module import, not inside functions. Otherwise they reload on every button click.
- **Run it locally.** Transcript fetching is often blocked from cloud hosts such as Hugging Face Spaces, so demo on your own machine.

## License

[MIT](LICENSE) © 2026 Aziz
