"""Rerank YouTube search results by embedding similarity (nearest neighbours in vector space)."""
from sentence_transformers import SentenceTransformer, util

from youtube_utils import search_videos

emb = SentenceTransformer("all-MiniLM-L6-v2")


def _rerank(df, target_text, top_k):
    video_texts = (df["title"] + " " + df["description"]).tolist()
    query_vec = emb.encode(target_text, convert_to_tensor=True)
    video_vecs = emb.encode(video_texts, convert_to_tensor=True)
    df = df.copy()
    df["youtube_rank"] = range(1, len(df) + 1)
    df["similarity"] = util.cos_sim(query_vec, video_vecs)[0].cpu().numpy().round(3)
    return df.sort_values("similarity", ascending=False).head(top_k).reset_index(drop=True)


def recommend(query, top_k=5):
    """Topic mode: search the query, then rerank by similarity to the query."""
    df = search_videos(query, 25)
    if df.empty:
        return df
    return _rerank(df, query, top_k)


def recommend_similar(video_id, info, top_k=5):
    """'More like this' mode: search by the video's title, compare to title + description."""
    df = search_videos(info["title"], 25)
    df = df[df["video_id"] != video_id]
    if df.empty:
        return df
    return _rerank(df, info["title"] + " " + info["description"], top_k)
