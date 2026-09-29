"""Comment sentiment with a pretrained RoBERTa model, plus spaCy keywords per sentiment."""
import re
from collections import Counter

import pandas as pd
import plotly.express as px
import spacy
from transformers import pipeline

# Load models once at import time, not on every click.
#
# Model choice: cardiffnlp/twitter-roberta-base-sentiment-latest, a RoBERTa model
# pretrained on ~124M tweets and fine-tuned for sentiment. It fits YouTube comments
# because it already understands short, messy social-media text (slang, emojis, typos),
# and it outputs negative / neutral / positive directly, which is what the brief asks for.
clf = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
    truncation=True,
    max_length=256,  # plenty for a comment; shorter inputs run faster
)
nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])

URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")

LABEL_COLORS = {"positive": "#2e9e5b", "neutral": "#9aa0a6", "negative": "#d64545"}


def clean(text):
    """Same preprocessing the model was trained with. Emojis are kept on purpose."""
    text = URL_RE.sub("http", text)
    text = MENTION_RE.sub("@user", text)
    return text.strip()


def analyze(comments_list):
    """Return (DataFrame[comment, label, score], label counts, overall verdict)."""
    cleaned = [clean(c) for c in comments_list]
    # Each batch is padded to its longest comment, so one long comment makes the whole
    # batch slow. Classifying in length order keeps batches uniform (~4x faster on CPU).
    order = sorted(range(len(cleaned)), key=lambda i: len(cleaned[i]))
    sorted_preds = clf([cleaned[i] for i in order], batch_size=16)
    preds = [None] * len(cleaned)
    for i, pred in zip(order, sorted_preds):
        preds[i] = pred
    df = pd.DataFrame({
        "comment": comments_list,
        "label": [p["label"] for p in preds],
        "score": [round(p["score"], 3) for p in preds],
    })
    counts = df["label"].value_counts()
    verdict = counts.idxmax()
    return df, counts, verdict


def pie_chart(counts):
    fig = px.pie(
        names=counts.index, values=counts.values, color=counts.index,
        color_discrete_map=LABEL_COLORS, title="Comment sentiment",
    )
    fig.update_traces(textinfo="percent+label", hole=0.45)
    return _style(fig)


def score_histogram(df):
    fig = px.histogram(
        df, x="score", color="label", nbins=20, color_discrete_map=LABEL_COLORS,
        title="Model confidence", barmode="overlay",
    )
    return _style(fig)


def _style(fig):
    """Transparent background so the chart sits cleanly in the app's light or dark theme."""
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=50, l=20, r=20, b=20), legend_title_text="",
        font_color="#8b95a5",  # mid-gray: readable in both light and dark mode
    )
    return fig


def top_keywords(texts, k=10):
    """Most common non-stopword noun lemmas across the texts."""
    counter = Counter()
    for doc in nlp.pipe(texts, batch_size=64):
        counter.update(
            tok.lemma_.lower() for tok in doc
            if tok.pos_ == "NOUN" and not tok.is_stop and tok.is_alpha and len(tok) > 2
        )
    return [word for word, _ in counter.most_common(k)]


def keywords_by_sentiment(df, k=10):
    """Return {'positive': [...], 'negative': [...]}."""
    return {
        label: top_keywords(df.loc[df["label"] == label, "comment"].tolist(), k)
        for label in ("positive", "negative")
    }
