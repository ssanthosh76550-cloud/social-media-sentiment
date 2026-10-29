"""
Visualizations Module.

Generates responsive, interactive Plotly charts and high-resolution Matplotlib Word Clouds
using a cohesive dark navy and vibrant blue aesthetic.
"""

from typing import Optional
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Design Palette: Dark Navy & Blue Accent Theme
THEME_COLORS = {
    "background_paper": "#0f172a",    # Slate 900
    "background_plot": "#1e293b",     # Slate 800
    "text": "#f8fafc",                # Slate 50
    "text_muted": "#94a3b8",          # Slate 400
    "grid": "#334155",                # Slate 700
    "blue_primary": "#3b82f6",        # Blue 500
    "blue_hover": "#60a5fa",          # Blue 400
    "cyan_accent": "#06b6d4",         # Cyan 500
    "positive": "#22c55e",            # Green 500
    "negative": "#ef4444",            # Red 500
    "neutral": "#94a3b8",             # Slate 400
    "platforms": {
        "X": "#1da1f2",
        "Instagram": "#e1306c",
        "Facebook": "#1877f2",
        "YouTube": "#ff0000"
    }
}

SENTIMENT_COLOR_MAP = {
    "Positive": THEME_COLORS["positive"],
    "Neutral": THEME_COLORS["neutral"],
    "Negative": THEME_COLORS["negative"]
}


def _apply_dark_theme(fig: go.Figure, title: str = "", height: int = 400) -> go.Figure:
    """Standardize dark navy aesthetic across all Plotly figures."""
    fig.update_layout(
        title={
            "text": f"<b>{title}</b>",
            "x": 0.02,
            "xanchor": "left",
            "font": {"size": 16, "color": THEME_COLORS["text"], "family": "Inter, sans-serif"}
        },
        height=height,
        paper_bgcolor=THEME_COLORS["background_paper"],
        plot_bgcolor=THEME_COLORS["background_plot"],
        font={"color": THEME_COLORS["text"], "family": "Inter, sans-serif"},
        margin=dict(l=40, r=30, t=55, b=40),
        legend=dict(
            bgcolor="rgba(15, 23, 42, 0.6)",
            bordercolor=THEME_COLORS["grid"],
            borderwidth=1,
            font=dict(color=THEME_COLORS["text"], size=11)
        ),
        xaxis=dict(
            gridcolor=THEME_COLORS["grid"],
            zerolinecolor=THEME_COLORS["grid"],
            tickfont=dict(color=THEME_COLORS["text_muted"], size=11),
            title_font=dict(color=THEME_COLORS["text"], size=12)
        ),
        yaxis=dict(
            gridcolor=THEME_COLORS["grid"],
            zerolinecolor=THEME_COLORS["grid"],
            tickfont=dict(color=THEME_COLORS["text_muted"], size=11),
            title_font=dict(color=THEME_COLORS["text"], size=12)
        ),
        hoverlabel=dict(
            bgcolor=THEME_COLORS["background_plot"],
            font_size=12,
            font_family="Inter, sans-serif"
        )
    )
    return fig


def plot_sentiment_distribution(df: pd.DataFrame) -> go.Figure:
    """Generate interactive donut chart displaying sentiment share."""
    if df is None or df.empty or "sentiment_label" not in df.columns:
        fig = go.Figure()
        return _apply_dark_theme(fig, "Sentiment Distribution (No Data)")

    counts = df["sentiment_label"].value_counts().reset_index()
    counts.columns = ["sentiment", "count"]

    colors = [SENTIMENT_COLOR_MAP.get(s, THEME_COLORS["blue_primary"]) for s in counts["sentiment"]]

    fig = go.Figure(
        data=[
            go.Pie(
                labels=counts["sentiment"],
                values=counts["count"],
                hole=0.60,
                marker=dict(colors=colors, line=dict(color=THEME_COLORS["background_paper"], width=2)),
                textinfo="percent+label",
                textfont=dict(size=13, color="#ffffff"),
                hovertemplate="<b>%{label}</b><br>Post Count: %{value:,}<br>Share: %{percent}<extra></extra>"
            )
        ]
    )
    return _apply_dark_theme(fig, "Overall Sentiment Distribution", height=380)


def plot_sentiment_trends_over_time(trends_df: pd.DataFrame) -> go.Figure:
    """Generate multi-line or stacked area chart tracking sentiment evolution over time."""
    if trends_df is None or trends_df.empty or "date" not in trends_df.columns:
        fig = go.Figure()
        return _apply_dark_theme(fig, "Sentiment Trends Over Time (No Data)")

    fig = go.Figure()

    for sentiment in ["Positive", "Neutral", "Negative"]:
        if sentiment in trends_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=trends_df["date"],
                    y=trends_df[sentiment],
                    mode="lines+markers",
                    name=sentiment,
                    line=dict(color=SENTIMENT_COLOR_MAP[sentiment], width=3),
                    marker=dict(size=5),
                    hovertemplate=f"<b>{sentiment}</b><br>Date: %{{x|%b %d, %Y}}<br>Posts: %{{y:,}}<extra></extra>"
                )
            )

    fig.update_xaxes(title="Timeline", tickformat="%b %d")
    fig.update_yaxes(title="Volume of Posts")
    return _apply_dark_theme(fig, "Daily Sentiment Trajectory Over Time", height=400)


def plot_platform_comparison(platform_df: pd.DataFrame) -> go.Figure:
    """Generate dual-axis grouped bar chart comparing volume and engagement by platform."""
    if platform_df is None or platform_df.empty:
        fig = go.Figure()
        return _apply_dark_theme(fig, "Platform Breakdown (No Data)")

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=platform_df["platform"],
            y=platform_df["post_count"],
            name="Total Posts",
            marker_color=THEME_COLORS["blue_primary"],
            hovertemplate="<b>%{x}</b><br>Posts: %{y:,}<extra></extra>"
        )
    )

    fig.add_trace(
        go.Bar(
            x=platform_df["platform"],
            y=platform_df["avg_engagement"],
            name="Avg Engagement / Post",
            marker_color=THEME_COLORS["cyan_accent"],
            hovertemplate="<b>%{x}</b><br>Avg Engagement: %{y:,.1f}<extra></extra>"
        )
    )

    fig.update_layout(barmode="group")
    fig.update_xaxes(title="Social Media Platform")
    fig.update_yaxes(title="Metric Count")
    return _apply_dark_theme(fig, "Platform Volume vs. Average Engagement", height=380)


def plot_top_hashtags(hashtags_df: pd.DataFrame, top_n: int = 10) -> go.Figure:
    """Generate horizontal bar chart displaying top trending hashtags."""
    if hashtags_df is None or hashtags_df.empty:
        fig = go.Figure()
        return _apply_dark_theme(fig, "Top Trending Hashtags (No Data)")

    df_plot = hashtags_df.head(top_n).sort_values("count", ascending=True)

    fig = go.Figure(
        go.Bar(
            x=df_plot["count"],
            y=df_plot["hashtag"],
            orientation="h",
            marker=dict(
                color=df_plot["count"],
                colorscale=[[0, "#1d4ed8"], [1, "#60a5fa"]],
                line=dict(color=THEME_COLORS["cyan_accent"], width=1)
            ),
            hovertemplate="<b>%{y}</b><br>Frequency: %{x:,}<extra></extra>"
        )
    )

    fig.update_xaxes(title="Post Count")
    fig.update_yaxes(title="Hashtag")
    return _apply_dark_theme(fig, f"Top {top_n} Most Discussed Hashtags", height=380)


def plot_post_volume_timeline(volume_df: pd.DataFrame, title_suffix: str = "Daily") -> go.Figure:
    """Generate bar chart representing post volume velocity."""
    if volume_df is None or volume_df.empty or "date" not in volume_df.columns:
        fig = go.Figure()
        return _apply_dark_theme(fig, f"{title_suffix} Post Volume (No Data)")

    fig = go.Figure(
        go.Bar(
            x=volume_df["date"],
            y=volume_df["post_count"],
            marker_color=THEME_COLORS["blue_primary"],
            hovertemplate="<b>Date: %{x|%b %d, %Y}</b><br>Posts: %{y:,}<extra></extra>"
        )
    )

    fig.update_xaxes(title="Date", tickformat="%b %d")
    fig.update_yaxes(title="Posts Published")
    return _apply_dark_theme(fig, f"{title_suffix} Post Volume Distribution", height=360)


def plot_engagement_by_sentiment(df: pd.DataFrame) -> go.Figure:
    """Generate box plot visualizing engagement distribution across sentiment classes."""
    if df is None or df.empty or "sentiment_label" not in df.columns:
        fig = go.Figure()
        return _apply_dark_theme(fig, "Engagement by Sentiment (No Data)")

    fig = px.box(
        df,
        x="sentiment_label",
        y="total_engagement",
        color="sentiment_label",
        color_discrete_map=SENTIMENT_COLOR_MAP,
        points="outliers",
        labels={"sentiment_label": "Sentiment", "total_engagement": "Total Engagement"}
    )
    return _apply_dark_theme(fig, "Engagement Distribution by Sentiment Class", height=380)


def plot_topic_distribution(topic_df: pd.DataFrame) -> go.Figure:
    """Generate horizontal bar chart of categories ranked by post volume."""
    if topic_df is None or topic_df.empty:
        fig = go.Figure()
        return _apply_dark_theme(fig, "Topic Volume (No Data)")

    df_plot = topic_df.sort_values("post_count", ascending=True)

    fig = go.Figure(
        go.Bar(
            x=df_plot["post_count"],
            y=df_plot["topic"],
            orientation="h",
            marker=dict(
                color=df_plot["avg_sentiment"],
                colorscale="Tealgrn",
                colorbar=dict(title="Avg Compound", tickfont=dict(color=THEME_COLORS["text_muted"]))
            ),
            hovertemplate="<b>%{y}</b><br>Posts: %{x:,}<br>Avg Sentiment: %{marker.color:+.2f}<extra></extra>"
        )
    )

    fig.update_xaxes(title="Number of Posts")
    fig.update_yaxes(title="Topic")
    return _apply_dark_theme(fig, "Topic Volume & Sentiment Overview", height=380)


def plot_trending_keywords_velocity(velocity_df: pd.DataFrame) -> go.Figure:
    """Generate bar chart displaying velocity growth % and trend scores."""
    if velocity_df is None or velocity_df.empty:
        fig = go.Figure()
        return _apply_dark_theme(fig, "Velocity-Ranked Trending Keywords (No Data)")

    df_plot = velocity_df.head(10).sort_values("trend_score", ascending=True)

    fig = go.Figure(
        go.Bar(
            x=df_plot["trend_score"],
            y=df_plot["keyword"],
            orientation="h",
            text=df_plot["trend_status"],
            textposition="inside",
            marker=dict(
                color=df_plot["growth_pct"],
                colorscale="Viridis",
                colorbar=dict(title="Growth %", tickfont=dict(color=THEME_COLORS["text_muted"]))
            ),
            hovertemplate="<b>Keyword: %{y}</b><br>Trend Score: %{x:.1f}<br>Growth: %{marker.color:+.1f}%<extra></extra>"
        )
    )

    fig.update_xaxes(title="Velocity Trend Score (Acceleration x Recency)")
    fig.update_yaxes(title="Keyword")
    return _apply_dark_theme(fig, "Trending Keywords Ranked by Velocity (Recent vs Baseline)", height=380)


def generate_wordcloud_figure(
    df: pd.DataFrame,
    text_column: str = "cleaned_text",
    max_words: int = 100
) -> Optional[plt.Figure]:
    """
    Generate high-definition Matplotlib figure displaying Word Cloud.
    """
    if df is None or df.empty:
        return None

    col = text_column if text_column in df.columns else "post_text"
    if col not in df.columns:
        return None

    combined_text = " ".join(df[col].dropna().astype(str).tolist())
    if not combined_text.strip():
        return None

    try:
        wc = WordCloud(
            width=900,
            height=450,
            background_color=THEME_COLORS["background_paper"],
            colormap="Blues_r",
            max_words=max_words,
            collocations=False,
            random_state=42
        ).generate(combined_text)

        fig, ax = plt.subplots(figsize=(10, 5), facecolor=THEME_COLORS["background_paper"])
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        plt.tight_layout(pad=0)
        return fig
    except Exception:
        return None
