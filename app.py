"""Gradio UI: summarize, recommend and analyze sentiment for YouTube videos."""
import html

import gradio as gr

import recommender
import sentiment
import summarizer
from youtube_utils import get_comments, get_transcript, get_video_id, get_video_info

EXAMPLE_VIDEO = "https://www.youtube.com/watch?v=aircAruvnKk"

BG = "#040404"       # page background: rgb(4, 4, 4)
SURFACE = "#161616"  # cards, panels, blocks: rgb(22, 22, 22)
BORDER = "#2a2a2a"   # subtle borders between surfaces

# Light mode: white page, black text, light-gray cards.
LIGHT_SURFACE = "#f4f4f4"
LIGHT_BORDER = "#e2e2e2"

# "neutral" grays have no blue tint (the old "slate" palette did).
# Every color has a light and a dark (`_dark`) value; Gradio's own theme setting
# (Settings in the footer: light, dark or system) picks which one is used.
THEME = gr.themes.Soft(
    primary_hue="red",
    neutral_hue="neutral",
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
).set(
    body_background_fill="white",
    body_text_color="black",
    background_fill_primary="white",
    background_fill_secondary=LIGHT_SURFACE,
    block_background_fill=LIGHT_SURFACE,
    panel_background_fill=LIGHT_SURFACE,
    input_background_fill="white",
    border_color_primary=LIGHT_BORDER,
    block_border_color=LIGHT_BORDER,
    input_border_color=LIGHT_BORDER,
    body_background_fill_dark=BG,
    background_fill_primary_dark=BG,
    background_fill_secondary_dark=SURFACE,
    block_background_fill_dark=SURFACE,
    panel_background_fill_dark=SURFACE,
    input_background_fill_dark=BG,
    input_background_fill_focus_dark=BG,
    border_color_primary_dark=BORDER,
    block_border_color_dark=BORDER,
    input_border_color_dark=BORDER,
    table_even_background_fill_dark=SURFACE,
    table_odd_background_fill_dark=BG,
    button_secondary_background_fill_dark=SURFACE,
    button_primary_background_fill="#e62117",
    button_primary_background_fill_dark="#e62117",
    button_primary_background_fill_hover="#c81b12",
    button_primary_background_fill_hover_dark="#c81b12",
    button_primary_text_color="white",
    button_primary_text_color_dark="white",
    block_radius="12px",
)

CSS = """
.gradio-container { width: 100% !important; max-width: 1100px !important; margin: 0 auto !important; }
#header { text-align: center; padding: 18px 0 6px; }
#header h1 { font-size: 2rem; margin-bottom: 4px; }
#header p { opacity: .7; margin: 0; }
.panel { border: 1px solid var(--border-color-primary); border-radius: 12px;
         padding: 16px 18px; background: var(--block-background-fill); min-height: 120px; }
.panel-title { font-size: .8rem; font-weight: 600; letter-spacing: .06em;
               text-transform: uppercase; opacity: .6; margin-bottom: 6px; }
.video-card { display: flex; gap: 20px; align-items: flex-start; margin: 8px 0; }
/* Gradio pads HTML blocks and links; remove it so everything lines up on one left edge. */
.html-container { padding: 0 !important; }
.video-card a, .rec-card a { padding: 0 !important; }
.video-card a:focus { outline: none; }
.video-card .thumb-link { flex: 0 0 360px; max-width: 45%; }
.video-card .thumb { width: 100%; border-radius: 12px; display: block; }
.video-card .title { display: block; font-weight: 600; font-size: 1.2rem; text-decoration: none;
                     color: var(--body-text-color); }
.video-card .title:hover { color: #e62117; }
.video-desc { opacity: .7; font-size: .9rem; margin-top: 8px; white-space: pre-line;
              display: -webkit-box; -webkit-line-clamp: 5; -webkit-box-orient: vertical; overflow: hidden; }
.loading { text-align: center; padding: 28px 0; opacity: .8; font-style: italic; font-size: 1.05rem; }
.rec-card { display: flex; gap: 16px; align-items: flex-start; padding: 12px;
            border: 1px solid var(--border-color-primary); border-radius: 12px;
            background: var(--block-background-fill); margin-bottom: 12px; }
.rec-card img { width: 200px; border-radius: 8px; flex-shrink: 0; }
.rec-card a.title { font-weight: 600; font-size: 1.05rem; text-decoration: none;
                    color: var(--body-text-color); }
.rec-card a.title:hover { color: #e62117; }
.rec-meta { opacity: .7; font-size: .9rem; margin-top: 4px; }
.rec-badge { display: inline-block; margin-top: 8px; padding: 2px 10px; border-radius: 999px;
             font-size: .8rem; font-weight: 600; background: rgba(230, 33, 23, .12); color: #e62117; }
button.gallery-item { margin: 2px !important; }
@media (max-width: 640px) {
  .rec-card { flex-direction: column; }
  .rec-card img { width: 100%; }
  .video-card { flex-direction: column; }
  .video-card .thumb-link { flex-basis: auto; width: 100%; max-width: 100%; }
}
"""


LOADING = "<p class='loading'>Loading...</p>"


def video_card(video_id):
    """Thumbnail with the video's title, channel and a truncated description beside it."""
    link = f"https://www.youtube.com/watch?v={video_id}"
    # The thumbnail comes from YouTube's image CDN (no API quota needed).
    thumb = (f"<a class='thumb-link' href='{link}' target='_blank'>"
             f"<img class='thumb' src='https://i.ytimg.com/vi/{video_id}/hqdefault.jpg' alt=''></a>")
    try:
        info = get_video_info(video_id)
    except Exception as e:
        print(f"[video_card] {type(e).__name__}")
        info = None
    details = ""
    if info:
        details = (f"<div><a class='title' href='{link}' target='_blank'>{html.escape(info['title'])}</a>"
                   f"<div class='rec-meta'>{html.escape(info['channel'])}</div>"
                   f"<div class='video-desc'>{html.escape(info['description'])}</div></div>")
    return f"<div class='video-card'>{thumb}{details}</div>"


def summarize_fn(url):
    video_id = get_video_id(url)
    if not video_id:
        raise gr.Error("That doesn't look like a YouTube link.")
    # Show the thumbnail and a loading note right away so the page doesn't look frozen.
    card = video_card(video_id)
    yield card, LOADING
    text = get_transcript(video_id)
    if not text:
        raise gr.Error("This video has no captions, so it can't be summarized.")
    try:
        llm = summarizer.llm_summary(text)
    except Exception as e:
        llm = f"*LLM summary failed: {e}*"
    yield card, llm


def results_html(df):
    cards = []
    for rank, row in enumerate(df.itertuples(), start=1):
        link = f"https://www.youtube.com/watch?v={row.video_id}"
        cards.append(f"""
        <div class="rec-card">
          <a href="{link}" target="_blank"><img src="{row.thumbnail}" alt=""></a>
          <div>
            <a class="title" href="{link}" target="_blank">{rank}. {html.escape(html.unescape(row.title))}</a>
            <div class="rec-meta">{html.escape(html.unescape(row.channel))}</div>
            <span class="rec-badge">{row.similarity:.0%} match · YouTube rank #{row.youtube_rank}</span>
          </div>
        </div>""")
    return "".join(cards)


def recommend_fn(query):
    query = (query or "").strip()
    if not query:
        raise gr.Error("Type a topic or paste a YouTube link.")
    video_id = get_video_id(query)
    if video_id:
        info = get_video_info(video_id)
        if not info:
            raise gr.Error("Couldn't find that video.")
        df = recommender.recommend_similar(video_id, info)
    else:
        df = recommender.recommend(query)
    if df.empty:
        raise gr.Error("No results found.")
    return results_html(df)


def sentiment_fn(url):
    video_id = get_video_id(url)
    if not video_id:
        raise gr.Error("That doesn't look like a YouTube link.")
    # Show the thumbnail and a loading note right away so the page doesn't look frozen.
    card = video_card(video_id)
    yield gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), card + LOADING
    comments = get_comments(video_id)
    if comments is None:
        raise gr.Error("Comments are disabled on this video (or it doesn't exist).")
    if not comments:
        raise gr.Error("This video has no comments yet.")

    df, counts, verdict = sentiment.analyze(comments)
    kw = sentiment.keywords_by_sentiment(df)
    breakdown = " · ".join(f"**{label}** {n}" for label, n in counts.items())
    report = (
        f"### Overall: {verdict} ({len(df)} comments)\n{breakdown}\n\n"
        f"👍 **People who liked it mention:** {', '.join(kw['positive']) or '—'}\n\n"
        f"👎 **People complaining mention:** {', '.join(kw['negative']) or '—'}"
    )
    # The last value un-hides the results column (it starts hidden so the page has no
    # empty placeholder boxes before the first analysis).
    yield (sentiment.pie_chart(counts), sentiment.score_histogram(df), report, df,
           gr.Column(visible=True), card)


with gr.Blocks(title="YouTube AI Analyzer") as app:
    gr.HTML(
        "<div id='header'><h1>▶ YouTube AI Content Analyzer</h1>"
        "<p>Summarize videos, find better recommendations and read the mood of the comments.</p>"
        "<p style='font-size:.85rem'>The first run downloads the AI models, so it can take a minute.</p></div>"
    )

    with gr.Tab("📝 Summarize"):
        with gr.Row(equal_height=True):
            sum_url = gr.Textbox(label="YouTube link", placeholder="https://www.youtube.com/watch?v=...", scale=5)
            sum_btn = gr.Button("Summarize", variant="primary", scale=1)
        gr.Examples([EXAMPLE_VIDEO], inputs=sum_url)
        sum_thumb = gr.HTML()
        with gr.Column(elem_classes="panel"):
            gr.HTML("<div class='panel-title'>Summary</div>")
            llm_out = gr.Markdown()
        sum_btn.click(summarize_fn, inputs=sum_url, outputs=[sum_thumb, llm_out])
        sum_url.submit(summarize_fn, inputs=sum_url, outputs=[sum_thumb, llm_out])

    with gr.Tab("🔎 Recommend"):
        with gr.Row(equal_height=True):
            rec_in = gr.Textbox(label="Topic or YouTube link", placeholder="linear regression", scale=5)
            rec_btn = gr.Button("Recommend", variant="primary", scale=1)
        gr.Examples(["linear regression", EXAMPLE_VIDEO], inputs=rec_in)
        rec_out = gr.HTML()
        rec_btn.click(recommend_fn, inputs=rec_in, outputs=rec_out)
        rec_in.submit(recommend_fn, inputs=rec_in, outputs=rec_out)

    with gr.Tab("💬 Sentiment"):
        with gr.Row(equal_height=True):
            sent_url = gr.Textbox(label="YouTube link", placeholder="https://www.youtube.com/watch?v=...", scale=5)
            sent_btn = gr.Button("Analyze comments", variant="primary", scale=1)
        gr.Examples([EXAMPLE_VIDEO], inputs=sent_url)
        sent_thumb = gr.HTML()
        with gr.Column(visible=False) as sent_results:
            sent_md = gr.Markdown(elem_classes="panel")
            with gr.Row():
                pie = gr.Plot(show_label=False)
                hist = gr.Plot(show_label=False)
            sent_df = gr.Dataframe(wrap=True, max_height=400, column_widths=["70%", "15%", "15%"])
        outputs = [pie, hist, sent_md, sent_df, sent_results, sent_thumb]
        # The outputs are hidden while it runs, so show the progress spinner on the button.
        sent_btn.click(sentiment_fn, inputs=sent_url, outputs=outputs, show_progress_on=sent_btn)
        sent_url.submit(sentiment_fn, inputs=sent_url, outputs=outputs, show_progress_on=sent_btn)


if __name__ == "__main__":
    app.launch(theme=THEME, css=CSS)
