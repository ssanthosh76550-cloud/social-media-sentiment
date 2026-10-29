"""
Social Media Sentiment and Trend Analysis Dashboard.

BCA Data Analytics Internship Project.
A comprehensive web dashboard built with Streamlit, Pandas, NLTK VADER, and Plotly
to monitor public sentiment, detect trending topics and keywords, and analyze
social media engagement dynamics.
"""

from datetime import datetime
from io import BytesIO
from pathlib import Path
import streamlit as st
import pandas as pd

# Core analytical modules
from src.data_loader import (
    load_dataset,
    get_default_dataset_path,
    get_sample_csv_template,
)
from src.preprocessing import validate_and_clean_data
from src.sentiment_analysis import (
    initialize_vader,
    analyze_dataframe_sentiment,
)
from src.metrics import calculate_kpi_summary, format_number
from src.trend_analysis import (
    get_top_hashtags,
    get_top_keywords,
    get_post_volume_over_time,
    get_sentiment_trends_over_time,
    get_topic_distribution,
    get_engagement_by_platform,
    get_top_posts,
    get_average_engagement_metrics,
    get_trending_keywords_velocity,
)
from src.visualizations import (
    plot_sentiment_distribution,
    plot_sentiment_trends_over_time,
    plot_platform_comparison,
    plot_top_hashtags,
    plot_post_volume_timeline,
    plot_engagement_by_sentiment,
    plot_topic_distribution,
    plot_trending_keywords_velocity,
    generate_wordcloud_figure,
    THEME_COLORS,
)

# ---------------------------------------------------------
# Page Configuration & Global Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="Social Media Sentiment & Trend Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for high-contrast dark theme aesthetics and polished UI
st.markdown(
    """
    <style>
    /* Metric Card Styling */
    div[data-testid="stMetric"] {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 14px 18px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    }
    div[data-testid="stMetric"]:hover {
        border-color: #3b82f6;
        transition: all 0.2s ease-in-out;
    }
    div[data-testid="stMetricLabel"] {
        color: #94a3b8;
        font-size: 0.85rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    div[data-testid="stMetricValue"] {
        color: #f8fafc;
        font-size: 1.6rem;
        font-weight: 700;
    }
    
    /* Sentiment Badges */
    .badge-pos {
        background-color: rgba(34, 197, 94, 0.2);
        color: #4ade80;
        padding: 3px 8px;
        border-radius: 6px;
        font-weight: 600;
        border: 1px solid rgba(34, 197, 94, 0.4);
    }
    .badge-neg {
        background-color: rgba(239, 68, 68, 0.2);
        color: #f87171;
        padding: 3px 8px;
        border-radius: 6px;
        font-weight: 600;
        border: 1px solid rgba(239, 68, 68, 0.4);
    }
    .badge-neu {
        background-color: rgba(148, 163, 184, 0.2);
        color: #cbd5e1;
        padding: 3px 8px;
        border-radius: 6px;
        font-weight: 600;
        border: 1px solid rgba(148, 163, 184, 0.4);
    }

    /* Subheader accents */
    .section-title {
        color: #60a5fa;
        font-weight: 600;
        font-size: 1.25rem;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* Download button hover styling */
    .stDownloadButton > button {
        background-color: #1e293b;
        color: #f8fafc;
        border: 1px solid #3b82f6;
        border-radius: 8px;
        font-weight: 500;
    }
    .stDownloadButton > button:hover {
        background-color: #2563eb;
        color: #ffffff;
        border-color: #60a5fa;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# Cached Data Processing Pipeline
# ---------------------------------------------------------
@st.cache_data(show_spinner="Processing dataset & scoring sentiment...")
def get_processed_data(file_bytes: bytes = None, filename: str = "default") -> pd.DataFrame:
    """
    Load raw CSV, run data cleaning and validation, and execute NLTK VADER sentiment scoring.
    Cached by file content to ensure lightning-fast dashboard interactions.
    """
    if file_bytes is None:
        path = get_default_dataset_path()
        raw_df = load_dataset(path)
    else:
        buffer = BytesIO(file_bytes)
        raw_df = load_dataset(buffer)

    clean_df = validate_and_clean_data(raw_df)
    scored_df = analyze_dataframe_sentiment(clean_df, text_column="cleaned_text")
    return scored_df


# ---------------------------------------------------------
# Sidebar Controls & Dynamic Filters
# ---------------------------------------------------------
def render_sidebar() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Render sidebar upload controls, filter parameters, and dataset stats."""
    with st.sidebar:
        st.image(
            "https://raw.githubusercontent.com/streamlit/brand/main/media/download/streamlit-logo-primary-colormark-darktext.png",
            width=160,
        )
        st.title("Settings & Filters")
        st.caption("BCA Data Analytics Internship Project")

        st.markdown("---")
        st.subheader("📁 Data Source")
        data_source = st.radio(
            "Select Dataset",
            options=["Default Sample (1,200 Posts)", "Upload Custom CSV"],
            index=0,
            help="Choose between the verified synthetic social media dataset or your own custom CSV export.",
        )

        uploaded_bytes = None
        source_name = "default"

        if data_source == "Upload Custom CSV":
            uploaded_file = st.file_uploader(
                "Upload Social Media CSV",
                type=["csv"],
                help="CSV must contain required columns: post_id, platform, username, post_text, hashtags, likes, comments, shares, post_date, topic",
            )
            if uploaded_file is not None:
                uploaded_bytes = uploaded_file.getvalue()
                source_name = uploaded_file.name
            else:
                st.info("Upload your CSV or download the format template below.")
                st.download_button(
                    label="📄 Download CSV Template",
                    data=get_sample_csv_template(),
                    file_name="social_media_sample_template.csv",
                    mime="text/csv",
                )

        # Load & score full dataset
        try:
            full_df = get_processed_data(uploaded_bytes, source_name)
        except Exception as e:
            st.error(f"Error loading dataset: {e}")
            st.stop()

        if full_df.empty:
            st.warning("The dataset contains zero records.")
            st.stop()

        st.markdown("---")
        st.subheader("🔍 Filters")

        # Search Query
        search_query = st.text_input(
            "Search Text / Keywords",
            placeholder="e.g. AI, update, match, concert...",
            help="Filters posts containing the specified term in their text.",
        ).strip().lower()

        # Date Range Filter
        min_date = full_df["post_date"].min().date()
        max_date = full_df["post_date"].max().date()

        if min_date == max_date:
            date_range = (min_date, max_date)
            st.caption(f"Single date available: {min_date}")
        else:
            date_range = st.date_input(
                "Date Range",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date,
            )

        # Platform Filter
        available_platforms = sorted(full_df["platform"].dropna().unique().tolist())
        selected_platforms = st.multiselect(
            "Platform",
            options=available_platforms,
            default=available_platforms,
        )

        # Sentiment Filter
        available_sentiments = ["Positive", "Neutral", "Negative"]
        selected_sentiments = st.multiselect(
            "Sentiment Class",
            options=available_sentiments,
            default=available_sentiments,
        )

        # Topic Filter
        available_topics = sorted(full_df["topic"].dropna().unique().tolist())
        selected_topics = st.multiselect(
            "Topic / Category",
            options=available_topics,
            default=available_topics,
        )

        # Reset Filters Button
        if st.button("🔄 Reset All Filters", use_container_width=True):
            st.rerun()

        st.markdown("---")
        # NLTK Status Notice
        vader_ok, vader_msg = initialize_vader()
        if vader_ok:
            st.success("🟢 VADER Engine: Active", icon="✅")
        else:
            st.warning(f"🟡 VADER: {vader_msg}", icon="⚠️")

    # Apply filters to produce the filtered slice
    filtered = full_df.copy()

    # Search filter
    if search_query:
        filtered = filtered[filtered["cleaned_text"].str.contains(search_query, na=False, case=False)]

    # Date filter
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_dt = pd.to_datetime(date_range[0])
        end_dt = pd.to_datetime(date_range[1]) + pd.Timedelta(days=1, microseconds=-1)
        filtered = filtered[(filtered["post_date"] >= start_dt) & (filtered["post_date"] <= end_dt)]

    # Platform filter
    if selected_platforms:
        filtered = filtered[filtered["platform"].isin(selected_platforms)]
    else:
        filtered = filtered.iloc[0:0]

    # Sentiment filter
    if selected_sentiments:
        filtered = filtered[filtered["sentiment_label"].isin(selected_sentiments)]
    else:
        filtered = filtered.iloc[0:0]

    # Topic filter
    if selected_topics:
        filtered = filtered[filtered["topic"].isin(selected_topics)]
    else:
        filtered = filtered.iloc[0:0]

    return full_df, filtered


# ---------------------------------------------------------
# Main Application Layout
# ---------------------------------------------------------
def main():
    # Load and filter data
    full_df, filtered_df = render_sidebar()

    # Header section
    st.markdown(
        """
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
            <div>
                <h1 style="margin: 0; padding: 0; color: #f8fafc; font-size: 2.2rem; font-weight: 800;">
                    Social Media Sentiment & Trend Analysis
                </h1>
                <p style="margin: 4px 0 0 0; color: #94a3b8; font-size: 0.95rem;">
                    BCA Data Analytics Internship Capstone Project &bull; Real-time NLP & Engagement Analytics
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Empty State Handling
    if filtered_df.empty:
        st.warning(
            "⚠️ No posts match the selected filter criteria. "
            "Please adjust your date range, platform, sentiment, or search keywords in the sidebar, "
            "or click **Reset All Filters**.",
            icon="⚠️",
        )
        return

    # Calculate KPIs
    kpis = calculate_kpi_summary(filtered_df)

    # ---------------------------------------------------------
    # Top KPI Metrics Ribbon
    # ---------------------------------------------------------
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        st.metric(
            label="Total Posts",
            value=f"{kpis['total_posts']:,}",
            delta=f"Active in {kpis['most_active_platform']}",
        )
    with col2:
        st.metric(
            label="Positive Share",
            value=f"{kpis['positive_pct']}%",
            delta=f"{kpis['positive_count']} posts",
            delta_color="normal",
        )
    with col3:
        st.metric(
            label="Negative Share",
            value=f"{kpis['negative_pct']}%",
            delta=f"-{kpis['negative_count']} posts",
            delta_color="inverse",
        )
    with col4:
        st.metric(
            label="Neutral Share",
            value=f"{kpis['neutral_pct']}%",
            delta=f"{kpis['neutral_count']} posts",
            delta_color="off",
        )
    with col5:
        st.metric(
            label="Avg Engagement",
            value=format_number(kpis["avg_engagement"]),
            delta=f"{format_number(kpis['total_engagement'])} total",
        )
    with col6:
        st.metric(
            label="Top Sentiment",
            value=kpis["most_common_sentiment"],
            delta=f"Score: {kpis['avg_sentiment_compound']:+.2f}",
        )

    st.markdown("<div style='margin-bottom: 1.2rem;'></div>", unsafe_allow_html=True)

    # ---------------------------------------------------------
    # Navigation Tabs
    # ---------------------------------------------------------
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📊 Overview & Sentiment",
        "📈 Temporal & Volume",
        "📱 Platform & Topics",
        "🚀 Trending & Keywords",
        "⚡ Engagement Analysis",
        "☁️ Word Cloud",
        "🔍 Post Explorer & Export",
    ])

    # ---------------------------------------------------------
    # TAB 1: Overview & Sentiment
    # ---------------------------------------------------------
    with tab1:
        c1, c2 = st.columns([1, 1])
        with c1:
            st.plotly_chart(
                plot_sentiment_distribution(filtered_df),
                use_container_width=True,
                key="chart_overview_sentiment_dist",
            )
        with c2:
            trends_daily = get_sentiment_trends_over_time(filtered_df, freq="D")
            st.plotly_chart(
                plot_sentiment_trends_over_time(trends_daily),
                use_container_width=True,
                key="chart_overview_sentiment_trends",
            )

        st.markdown("---")
        c3, c4 = st.columns([1, 1])
        with c3:
            topic_summary = get_topic_distribution(filtered_df)
            st.plotly_chart(
                plot_topic_distribution(topic_summary),
                use_container_width=True,
                key="chart_overview_topic_dist",
            )
        with c4:
            plat_summary = get_engagement_by_platform(filtered_df)
            st.plotly_chart(
                plot_platform_comparison(plat_summary),
                use_container_width=True,
                key="chart_overview_platform_comp",
            )

    # ---------------------------------------------------------
    # TAB 2: Temporal & Volume Analysis
    # ---------------------------------------------------------
    with tab2:
        col_ctrl, col_chart = st.columns([1, 3])
        with col_ctrl:
            st.markdown("#### Aggregation Granularity")
            granularity = st.radio(
                "Choose Volume Frequency:",
                options=["Daily Volume", "Weekly Volume"],
                index=0,
            )
            freq_code = "W-MON" if "Weekly" in granularity else "D"
            vol_df = get_post_volume_over_time(filtered_df, freq=freq_code)

            st.markdown("#### Temporal Summary")
            st.write(f"**Total Periods:** {len(vol_df)}")
            if not vol_df.empty:
                max_vol = vol_df.loc[vol_df["post_count"].idxmax()]
                st.write(f"**Peak Period:** {pd.to_datetime(max_vol['date']).strftime('%b %d, %Y')}")
                st.write(f"**Peak Volume:** {int(max_vol['post_count'])} posts")
                st.write(f"**Peak Engagement:** {int(max_vol['total_engagement']):,} interactions")

        with col_chart:
            st.plotly_chart(
                plot_post_volume_timeline(vol_df, title_suffix=granularity.split()[0]),
                use_container_width=True,
                key="chart_temporal_volume_timeline",
            )

        st.markdown("---")
        st.markdown("#### Sentiment Evolution Over Selected Timeline")
        st.plotly_chart(
            plot_sentiment_trends_over_time(trends_daily),
            use_container_width=True,
            key="chart_temporal_sentiment_evolution",
        )

    # ---------------------------------------------------------
    # TAB 3: Platform & Topics
    # ---------------------------------------------------------
    with tab3:
        p1, p2 = st.columns(2)
        with p1:
            st.markdown("#### Platform Comparison")
            plat_data = get_engagement_by_platform(filtered_df)
            st.plotly_chart(
                plot_platform_comparison(plat_data),
                use_container_width=True,
                key="chart_tab3_platform_comparison",
            )
            st.dataframe(
                plat_data.rename(columns={
                    "platform": "Platform",
                    "post_count": "Post Count",
                    "avg_likes": "Avg Likes",
                    "avg_comments": "Avg Comments",
                    "avg_shares": "Avg Shares",
                    "total_engagement": "Total Interactions",
                    "avg_engagement": "Avg Engagement"
                }),
                use_container_width=True,
                hide_index=True,
            )

        with p2:
            st.markdown("#### Category & Topic Distribution")
            topic_data = get_topic_distribution(filtered_df)
            st.plotly_chart(
                plot_topic_distribution(topic_data),
                use_container_width=True,
                key="chart_tab3_topic_distribution",
            )
            st.dataframe(
                topic_data.rename(columns={
                    "topic": "Topic",
                    "post_count": "Post Count",
                    "total_engagement": "Total Engagement",
                    "avg_engagement": "Avg Engagement",
                    "avg_sentiment": "Compound Sentiment"
                }),
                use_container_width=True,
                hide_index=True,
            )

    # ---------------------------------------------------------
    # TAB 4: Trending & Keywords
    # ---------------------------------------------------------
    with tab4:
        st.markdown("### 🚀 Trend Velocity & Emerging Keywords")
        st.info(
            "💡 **Trend Methodology Note**: Simple raw frequency often highlights evergreen generic words. "
            "Our **Velocity Algorithm** evaluates acceleration: it splits the timeline into baseline vs recent periods, "
            "calculating Growth Rate (%) and a weighted Trend Score ($S = F_{recent} \\times \\log_2(1 + \\text{Growth})$). "
            "Keywords are flagged as *Breakout*, *Surging*, or *Stable*.",
            icon="ℹ️"
        )

        col_t1, col_t2 = st.columns([1, 1])
        with col_t1:
            velocity_df = get_trending_keywords_velocity(filtered_df, top_n=10)
            st.plotly_chart(
                plot_trending_keywords_velocity(velocity_df),
                use_container_width=True,
                key="chart_trending_velocity",
            )

        with col_t2:
            st.markdown("#### Trending Hashtags")
            hashtags_df = get_top_hashtags(filtered_df, top_n=10)
            st.plotly_chart(
                plot_top_hashtags(hashtags_df, top_n=10),
                use_container_width=True,
                key="chart_trending_hashtags",
            )

        st.markdown("---")
        st.markdown("#### Top Velocity Trending Keywords Data Table")
        if not velocity_df.empty:
            st.dataframe(
                velocity_df.rename(columns={
                    "keyword": "Keyword",
                    "recent_count": "Recent Count",
                    "baseline_count": "Baseline Count",
                    "growth_pct": "Growth Rate (%)",
                    "trend_score": "Trend Velocity Score",
                    "trend_status": "Status",
                }),
                use_container_width=True,
                hide_index=True,
            )

    # ---------------------------------------------------------
    # TAB 5: Engagement Analysis
    # ---------------------------------------------------------
    with tab5:
        st.markdown("### ⚡ Engagement Dynamics")
        eng_metrics = get_average_engagement_metrics(filtered_df)

        e_col1, e_col2, e_col3, e_col4 = st.columns(4)
        with e_col1:
            st.metric("Avg Likes / Post", format_number(eng_metrics["avg_likes"]))
        with e_col2:
            st.metric("Avg Comments / Post", format_number(eng_metrics["avg_comments"]))
        with e_col3:
            st.metric("Avg Shares / Post", format_number(eng_metrics["avg_shares"]))
        with e_col4:
            st.metric("Avg Total Engagement", format_number(eng_metrics["avg_total_engagement"]))

        st.caption(f"ℹ️ {eng_metrics['engagement_rate_note']}")

        st.markdown("---")
        ec1, ec2 = st.columns(2)
        with ec1:
            st.plotly_chart(
                plot_engagement_by_sentiment(filtered_df),
                use_container_width=True,
                key="chart_engagement_by_sentiment",
            )
        with ec2:
            st.markdown("#### Top 10 Most Viral Posts by Engagement")
            top_posts = get_top_posts(filtered_df, top_n=10)
            if not top_posts.empty:
                display_top = top_posts[[
                    "platform", "username", "post_text", "sentiment_label", "total_engagement"
                ]].copy()
                st.dataframe(display_top, use_container_width=True, hide_index=True)

    # ---------------------------------------------------------
    # TAB 6: Word Cloud
    # ---------------------------------------------------------
    with tab6:
        st.markdown("### ☁️ Linguistic & Word Cloud Analysis")
        wc_sentiment = st.selectbox(
            "Filter Word Cloud by Sentiment Class:",
            options=["All Sentiments", "Positive", "Negative", "Neutral"],
            index=0,
        )

        wc_sub = filtered_df
        if wc_sentiment != "All Sentiments":
            wc_sub = filtered_df[filtered_df["sentiment_label"] == wc_sentiment]

        wc_fig = generate_wordcloud_figure(wc_sub, text_column="cleaned_text", max_words=120)
        if wc_fig is not None:
            st.pyplot(wc_fig, use_container_width=True)
        else:
            st.info("Insufficient textual data to generate word cloud for the chosen filter.")

        st.markdown("---")
        st.markdown("#### Most Frequent Topical Words")
        freq_kw = get_top_keywords(wc_sub, top_n=15)
        if not freq_kw.empty:
            st.dataframe(
                freq_kw.rename(columns={
                    "keyword": "Topical Word",
                    "frequency": "Occurrences",
                    "percentage": "Word Frequency (%)"
                }),
                use_container_width=True,
                hide_index=True,
            )

    # ---------------------------------------------------------
    # TAB 7: Post Explorer & Export Center
    # ---------------------------------------------------------
    with tab7:
        st.markdown("### 🔍 Filterable Social Media Posts Explorer")
        st.caption(f"Displaying {len(filtered_df):,} matching posts out of {len(full_df):,} total records.")

        # Data export buttons
        dcol1, dcol2, dcol3 = st.columns([1, 1, 2])
        with dcol1:
            csv_filtered = filtered_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Filtered Data CSV",
                data=csv_filtered,
                file_name=f"social_media_filtered_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with dcol2:
            csv_full = full_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Full Analyzed Dataset CSV",
                data=csv_full,
                file_name=f"social_media_full_analyzed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True,
            )

        # Presentation Table
        table_cols = [
            "post_id", "platform", "username", "post_text", "topic",
            "sentiment_label", "sentiment_compound", "likes", "comments",
            "shares", "total_engagement", "post_date"
        ]
        available_table_cols = [c for c in table_cols if c in filtered_df.columns]
        
        st.dataframe(
            filtered_df[available_table_cols].sort_values(by="post_date", ascending=False),
            use_container_width=True,
            hide_index=True,
            column_config={
                "sentiment_compound": st.column_config.NumberColumn(
                    "Score", format="%.3f"
                ),
                "total_engagement": st.column_config.NumberColumn(
                    "Total Eng.", format="%d"
                ),
                "post_date": st.column_config.DatetimeColumn(
                    "Date & Time", format="YYYY-MM-DD HH:mm"
                ),
            }
        )


if __name__ == "__main__":
    main()
