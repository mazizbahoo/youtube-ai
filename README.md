# youtube-ai

AI-powered YouTube analyzer that **summarizes videos**, **recommends related content**, and **analyzes comment sentiment**. Built with Python, Gradio, Transformers, spaCy, and Gemini.

> **Status:** in development. This is a solo project, built in phases. See the [Roadmap](#roadmap) for progress.

---

## Features

| Tab | Input | What it does |
|-----|-------|--------------|
| **Summarize** | A YouTube link | Fetches the transcript and shows two summaries side by side: an **LLM summary** (Gemini, abstractive) and an **extractive summary** (TF-IDF, built from scratch). |
| **Recommend** | A topic *or* a YouTube link | Searches YouTube, embeds every result with a sentence-transformer, and reranks by cosine similarity to your query (nearest neighbours in vector space). With a link, it runs in "more like this video" mode. |
| **Sentiment** | A YouTube link | Pulls up to 200 top comments, classifies them as positive / neutral / negative with a RoBERTa model, and charts the breakdown. spaCy pulls out the nouns that people praising and complaining mention most. |

## Tech stack

- **UI:** [Gradio](https://www.gradio.app/)
- **Data:** YouTube Data API v3 (`google-api-python-client`), `youtube-transcript-api`
- **LLM:** Google Gemini (`google-genai`)
- **Sentiment:** `cardiffnlp/twitter-roberta-base-sentiment-latest` via 🤗 Transformers + PyTorch
- **Embeddings:** `all-MiniLM-L6-v2` via `sentence-transformers`
- **NLP:** spaCy (`en_core_web_sm`), scikit-learn `TfidfVectorizer`
- **Analysis and charts:** Pandas, Plotly

## Project structure

```
youtube-ai/
├── app.py              # Gradio UI: three tabs plus the glue functions
├── youtube_utils.py    # Video ID parsing, transcripts, search, comments, video metadata
├── sentiment.py        # Comment cleaning, sentiment classification, charts, spaCy keywords
├── summarizer.py       # Gemini summary + TF-IDF extractive summary
├── recommender.py      # Embedding-based reranking of search results
├── notebooks/          # Scratch notebooks used to prototype each module
├── requirements.txt
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
python -m spacy download en_core_web_sm
```

If `requirements.txt` doesn't exist yet, install the packages directly:

```bash
pip install gradio pandas plotly spacy scikit-learn youtube-transcript-api \
            google-api-python-client google-genai transformers torch \
            sentence-transformers python-dotenv
python -m spacy download en_core_web_sm
```

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

1. `get_video_id()` pulls the 11-character ID (`[A-Za-z0-9_-]{11}`) from `watch?v=`, `youtu.be/`, and `/shorts/` links with regex.
2. `get_transcript()` fetches English captions with `YouTubeTranscriptApi().fetch(...)`.
3. **LLM summary:** the whole transcript goes to Gemini in a single prompt (the large context window means no chunking). The prompt asks for a 2-line overview, 5 key points, and a one-line takeaway.
4. **Extractive summary:**
   - Split the transcript into ~25-word chunks. Auto-generated captions have no punctuation, so sentence splitting doesn't work.
   - Fit a `TfidfVectorizer` on the chunks and score each chunk by the sum of its TF-IDF weights.
   - Take the top 5 chunks, then re-sort them by their original position so the summary reads in order.

The LLM summary usually reads better because it is **abstractive** (it writes new sentences), while the TF-IDF one is **extractive** (it copies existing ones).

### Recommend

1. `search_videos(query, 25)` gets candidates from the YouTube search API.
2. Each video becomes one text: `title + " " + description`.
3. The query and all video texts are encoded with `all-MiniLM-L6-v2`.
4. `util.cos_sim()` scores each video and the top 5 are returned. This is **KNN**: nearest neighbours in embedding space.
5. **"More like this video" mode:** when the input is a link, the video's title becomes the search query, its title + description becomes the comparison vector, and the original video is removed from the results.

Results render as an HTML list with thumbnails and clickable titles.

### Sentiment

1. `get_comments()` pages through `commentThreads().list(...)` (100 per call) until it reaches 200 comments or runs out of pages.
2. Regex cleaning replaces URLs with `http` and `@username` with `@user`, the format the Twitter-RoBERTa model was trained on. Emojis are kept on purpose because they carry a lot of sentiment.
3. The model classifies comments in batches of 16, and the results go into a DataFrame with `comment`, `label`, and `score` columns.
4. Outputs: a Plotly pie chart of labels, an overall verdict (the most common label), and optionally a histogram of confidence scores.
5. **Keywords:** positive and negative comments go through spaCy separately. The app keeps lemmas of non-stopword nouns and uses `Counter` to find the top 10 for each group, for example *"people complaining mention: audio, ads, length."*

## Roadmap

Each phase has a checkpoint that must pass before moving on.

- [ ] **Phase 0: Setup.** venv, dependencies, API keys, `.env`.
  *Checkpoint:* `import gradio, transformers, spacy` runs.
- [ ] **Phase 1: `youtube_utils.py`.** `get_video_id`, `get_transcript`, `search_videos`, `get_comments`, video metadata. Prototype in a notebook first.
  *Checkpoint:* for a real link, print its transcript, 5 search results, and 10 comments.
- [ ] **Phase 2: `sentiment.py`.** Cleaning, classification, pie chart, spaCy keywords.
  *Checkpoint:* on a popular video, the pie chart looks sensible and the negative keywords match what the comments say.
- [ ] **Phase 3: `summarizer.py`.** Gemini summary + TF-IDF extractive summary.
  *Checkpoint:* the summary of a 10-minute video is accurate (watch it to check).
- [ ] **Phase 4: `recommender.py`.** Embedding rerank + "more like this" mode.
  *Checkpoint:* for "linear regression", the reranked order makes more sense than YouTube's raw order.
- [ ] **Phase 5: `app.py`.** Gradio UI with three tabs, `gr.Error` popups, HTML thumbnails.
  *Checkpoint:* all three tabs work back to back on 3 different videos.
- [ ] **Phase 6: Polish.** Edge-case testing, a loading hint, `pip freeze > requirements.txt`, the report.

### Edge cases to test

- [ ] YouTube Shorts link
- [ ] Video with no captions
- [ ] Video with comments disabled
- [ ] Non-English video
- [ ] Garbage / non-YouTube link

Each one should show a clean `gr.Error` popup instead of crashing.

## Course mapping

| Feature | Technique | Lesson |
|---------|-----------|--------|
| Video ID parsing, comment cleaning | Regex | L30 |
| Keyword extraction | spaCy | L31 |
| Search results, sentiment tables | Pandas | L13 |
| Sentiment pie chart and histogram | Plotly | L15 |
| Recommendation reranking | KNN / cosine similarity | L22 |
| Web interface | Gradio | L32 |
| Sentiment model, embeddings | PyTorch / Transformers | L41 |
| Abstractive summary | LLM (Gemini) | L45 |

## Notes and gotchas

- **YouTube API quota:** 10,000 units per day. A `search` costs **100 units**, and a comments or videos call costs about **1**. Don't put search in a loop while testing, and cache results in the notebook.
- **Transcript API versions:** the code uses the 1.x syntax (`YouTubeTranscriptApi().fetch(...)`). If `fetch` doesn't exist, you have an old version. Upgrade it, or use `YouTubeTranscriptApi.get_transcript(video_id)`, which returns a list of dicts with a `"text"` key.
- **Gemini model names change often.** If `gemini-2.5-flash` fails, check AI Studio for the current free model and update the name.
- **Load models once.** The sentiment pipeline and the embedding model are created at module import, not inside functions. Otherwise they reload on every button click.
- **Run it locally.** Transcript fetching is often blocked from cloud hosts such as Hugging Face Spaces, so demo on your own machine.

## Troubleshooting

| Error | Likely cause | Fix |
|-------|--------------|-----|
| `TranscriptsDisabled` / `NoTranscriptFound` | The video has no (English) captions | Try another video. The app shows a popup. |
| `HttpError 403` on comments | Comments are disabled on the video | Expected. The app shows a popup. |
| `HttpError 403 quotaExceeded` | Daily YouTube quota is used up | Wait until the quota resets (midnight Pacific time). |
| `OSError: [E050] Can't find model 'en_core_web_sm'` | spaCy model not downloaded | `python -m spacy download en_core_web_sm` |
| `AttributeError: ... has no attribute 'fetch'` | Old `youtube-transcript-api` | `pip install -U youtube-transcript-api` |
| Gemini `404 model not found` | Model was renamed | Update the model name from AI Studio. |

## License

[MIT](LICENSE) © 2026 Aziz
