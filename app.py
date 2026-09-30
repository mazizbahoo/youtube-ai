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

# "neutral" grays have no blue tint (the old "slate" palette did).
# The colors are set for both light and dark mode, and FORCE_DARK below keeps
# the page in dark mode so text stays light on these dark surfaces.
THEME = gr.themes.Soft(
    primary_hue="red",
    neutral_hue="neutral",
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
).set(
    body_background_fill=BG,
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

# Runs in the browser on page load: switch Gradio to its dark mode.
FORCE_DARK = "() => { document.body.classList.add('dark'); }"

CSS = """
.gradio-container { width: 100% !important; max-width: 1100px !important; margin: 0 auto !important; }
#header { text-align: center; padding: 18px 0 6px; }
#header h1 { font-size: 2rem; margin-bottom: 4px; }
#header p { opacity: .7; margin: 0; }
.panel { border: 1px solid var(--border-color-primary); border-radius: 12px;
         padding: 16px 18px; background: var(--block-background-fill); min-height: 120px; }
.panel-title { font-size: .8rem; font-weight: 600; letter-spacing: .06em;
               text-transform: uppercase; opacity: .6; margin-bottom: 6px; }
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
}
"""


def summarize_fn(url):
    video_id = get_video_id(url)
    if not video_id:
        raise gr.Error("That doesn't look like a YouTube link.")
    text = get_transcript(video_id)
    if not text:
        raise gr.Error("This video has no captions, so it can't be summarized.")
    try:
        llm = summarizer.llm_summary(text)
    except Exception as e:
        llm = f"*LLM summary failed: {e}*"
    return llm, summarizer.extractive_summary(text)


def results_html(df):
    cards = []
    for rank, row in enumerate(df.itertuples(), start=1):
        link = f"https://www.youtube.com/watch?v={row.video_id}"
        cards.append(f"""
        <div class="rec-card">
          <a href="{link}" target="_blank"><img src="{row.thumbnail}" alt=""></a>
          <div>
            <a class="title" href="{link}" target="_blank">{rank}. {html.escape(row.title)}</a>
            <div class="rec-meta">{html.escape(row.channel)}</div>
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
    return sentiment.pie_chart(counts), sentiment.score_histogram(df), report, df, gr.Column(visible=True)


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
        with gr.Row(equal_height=True):
            with gr.Column(elem_classes="panel"):
                gr.HTML("<div class='panel-title'>LLM summary · abstractive</div>")
                llm_out = gr.Markdown()
            with gr.Column(elem_classes="panel"):
                gr.HTML("<div class='panel-title'>TF-IDF summary · extractive</div>")
                ext_out = gr.Markdown()
        sum_btn.click(summarize_fn, inputs=sum_url, outputs=[llm_out, ext_out])
        sum_url.submit(summarize_fn, inputs=sum_url, outputs=[llm_out, ext_out])

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
        with gr.Column(visible=False) as sent_results:
            sent_md = gr.Markdown(elem_classes="panel")
            with gr.Row():
                pie = gr.Plot(show_label=False)
                hist = gr.Plot(show_label=False)
            sent_df = gr.Dataframe(wrap=True, max_height=400, column_widths=["70%", "15%", "15%"])
        outputs = [pie, hist, sent_md, sent_df, sent_results]
        # The outputs are hidden while it runs, so show the progress spinner on the button.
        sent_btn.click(sentiment_fn, inputs=sent_url, outputs=outputs, show_progress_on=sent_btn)
        sent_url.submit(sentiment_fn, inputs=sent_url, outputs=outputs, show_progress_on=sent_btn)


if __name__ == "__main__":
    app.launch(theme=THEME, css=CSS, js=FORCE_DARK)
