# Social Media Sentiment and Trend Analysis

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-FF4B4B.svg)](https://streamlit.io/)
[![Plotly](https://img.shields.io/badge/Plotly-Interactive-3F4F75.svg)](https://plotly.com/)
[![NLTK VADER](https://img.shields.io/badge/NLTK-VADER-green.svg)](https://www.nltk.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A complete, production-grade Data Analytics internship project designed for social media text processing, multi-platform engagement tracking, VADER-powered sentiment classification, and velocity-based trend detection.

---

## 📌 Project Overview

In today's hyper-connected digital landscape, public sentiment shifts rapidly across social media platforms like X, Instagram, Facebook, and YouTube. Organizations, marketing teams, and researchers require automated analytical pipelines to track brand perception, monitor trending topics, detect consumer frustration early, and analyze engagement drivers without relying on expensive, black-box third-party services.

This project delivers an interactive web dashboard that:
1. Ingests social media posts across 8 key industry verticals.
2. Cleanses and sanitizes textual data, extracting hashtags and keywords.
3. Scores sentiment using the **NLTK VADER** rule-based lexicon.
4. Detects viral trends using an **acceleration velocity model** (recent vs. baseline comparison).
5. Visualizes multi-dimensional patterns using high-contrast Plotly charts and Word Clouds.
6. Allows interactive multi-attribute filtering and CSV exports.

---

## 🎯 Problem Statement & Objectives

### Problem Statement
Traditional social media monitoring often suffers from:
- **Misleading Trend Detection:** Treating raw word frequency as proof of a viral trend, which merely surfaces evergreen stop words.
- **Inflexible Scoring:** Fragile sentiment models that crash on offline environments or when encountering missing text and malformed datasets.
- **Engagement Misconceptions:** Conflating interaction counts with standardized engagement rate without considering impressions.

### Key Objectives
- **Automated Text Processing:** Implement a resilient preprocessing pipeline to handle missing values, duplicate entries, malformed dates, and emoji-rich text.
- **Three-Class Sentiment Classification:** Categorize posts into **Positive**, **Negative**, and **Neutral** with continuous compound scores ($-1.0$ to $+1.0$).
- **Velocity-Driven Trend Detection:** Calculate true trending keywords by measuring growth rate between baseline and recent time windows.
- **Executive BI Dashboard:** Provide interactive KPI cards, donut distributions, time-series trajectories, platform benchmarks, word clouds, and data export facilities.

---

## 🛠️ Technology Stack

| Component | Library / Framework | Purpose |
| :--- | :--- | :--- |
| **Language** | Python 3.9 / 3.11+ | Core runtime environment |
| **Data Processing** | Pandas 2.x, NumPy 1.24+ | Vectorized tabular operations and data wrangling |
| **Sentiment Engine**| NLTK VADER | Rule-based valence-aware sentiment scoring |
| **Interactive UI** | Streamlit 1.35+ | Modern, responsive web dashboard framework |
| **Data Visualization** | Plotly Express & Graph Objects | Dark-themed, tooltip-rich interactive charts |
| **Text Visuals** | WordCloud, Matplotlib | High-resolution linguistic word clouds |
| **Quality & Tests** | Pytest 8.x | Automated test suite covering edge cases |

---

## 📂 Architecture and Folder Structure

```text
social-media-sentiment-analysis/
│
├── app.py                      # Main Streamlit web application entry point
├── requirements.txt            # Production dependencies
├── runtime.txt                 # Target deployment runtime configuration
├── README.md                   # Comprehensive internship documentation
├── .gitignore                  # Git tracking exclusion rules
│
├── .streamlit/
│   └── config.toml             # Custom dark navy & blue accent UI theme
│
├── assets/
│   └── .gitkeep                # Directory for architecture diagrams and screenshots
│
├── data/
│   └── social_media_posts.csv  # Verified dataset of 1,200 synthetic social records
│
├── scripts/
│   └── generate_dataset.py     # Reproducible synthetic dataset generator
│
├── src/
│   ├── __init__.py             # Package declaration
│   ├── data_loader.py          # CSV loading, encoding fallback, schema validation
│   ├── preprocessing.py        # Text sanitization, hashtag extraction, data cleaning
│   ├── sentiment_analysis.py   # NLTK VADER sentiment scoring with offline fallback
│   ├── trend_analysis.py       # Temporal aggregations, platform stats, velocity algorithm
│   ├── metrics.py              # Executive KPI computations and number formatting
│   └── visualizations.py       # Plotly charts and Matplotlib word cloud figures
│
└── tests/
    ├── __init__.py             # Test suite declaration
    ├── test_data_loader.py     # Schema validation and CSV loading tests
    ├── test_preprocessing.py   # Cleaning, deduplication, date coercion tests
    ├── test_sentiment.py       # Positive, negative, neutral, and edge case tests
    └── test_trends.py          # Hashtags, volume, velocity, and metric tests
```

---

## 📊 Dataset Description

The project includes a realistic synthetic dataset (`data/social_media_posts.csv`) containing **1,200 unique records** across 4 major platforms and 8 diverse industry topics.

> **Note on Data Provenance:** All records, usernames, and posts are synthetically generated for educational demonstration; no private or copyrighted social media user data is included.

### Dataset Schema

| Column Name | Type | Description | Example |
| :--- | :--- | :--- | :--- |
| `post_id` | String | Unique record identifier | `POST_0001` |
| `platform` | String | Social network (`X`, `Instagram`, `Facebook`, `YouTube`) | `X` |
| `username` | String | Fictional user handle | `@tech_insider` |
| `post_text` | String | Original post text with emojis & punctuation | `"The new AI breakthrough..."` |
| `hashtags` | String | Associated hashtags | `#AI #Innovation` |
| `likes` | Integer | Non-negative like count | `1250` |
| `comments` | Integer | Non-negative comment count | `85` |
| `shares` | Integer | Non-negative share / retweet count | `140` |
| `post_date` | String | ISO 8601 formatted timestamp | `2024-08-15 14:30:00` |
| `topic` | String | Domain category (Technology, Business, Health, Sports, etc.) | `Technology` |

---

## 🔬 Methodology

### 1. Data Cleaning & Text Preprocessing
- **Deduplication:** Posts are deduplicated using `post_id` or unique content hashes.
- **Safe Normalization:** Strips external URLs (`https?://\S+`) and `@mentions`, while preserving original casing in `post_text` for UI rendering. Cleaned lowercase tokens are saved to `cleaned_text`.
- **Date & Engagement Coercion:** Dates are parsed using `pd.to_datetime(errors='coerce')`. Invalid dates are imputed with the median dataset timestamp. Engagement numbers are clipped at a lower bound of `0`.
- **Hashtag Parsing:** Regex `#([A-Za-z0-9_]+)` extracts and normalizes tags to lowercase canonical forms.

### 2. NLTK VADER Sentiment Scoring
The dashboard integrates NLTK's **VADER** (*Valence Aware Dictionary and sEntiment Reasoner*), a gold-standard rule-based model specifically tuned for social media sentiment:
- **Compound Score ($c$):** A normalized metric computed by summing valence scores of each word adjusted for grammatical modifiers (boosters, negations, capitalization), bounded between $-1.0$ and $+1.0$:
  $$c \in [-1.0, +1.0]$$
- **Classification Thresholds:**
  - **Positive:** $c \ge +0.05$
  - **Negative:** $c \le -0.05$
  - **Neutral:** $-0.05 < c < +0.05$
- **Offline Resilience:** If network firewalls or sandbox limits prevent downloading NLTK's external resource bundle, a built-in heuristic fallback dictionary takes over transparently, preventing runtime crashes.

### 3. Trend Velocity Algorithm
Traditional dashboards rank trends purely by raw frequency, which incorrectly highlights static evergreen terms. This project implements a **two-window velocity algorithm**:
1. Partition chronological posts into **Baseline Window** (earlier 50%) and **Recent Window** (latest 50%).
2. Count keyword frequency in both windows: $F_{baseline}$ and $F_{recent}$.
3. Calculate **Growth Rate (%)**:
   $$\text{Growth \%} = \frac{F_{recent} - F_{baseline}}{\max(F_{baseline}, 1)} \times 100$$
4. Compute **Velocity Trend Score ($S$)**:
   $$S = F_{recent} \times \log_2\left(2 + \frac{\max(\text{Growth \%}, 0)}{50}\right)$$
5. Classify trend status:
   - **Breakout 🚀:** Growth $> 50\%$ with zero baseline history.
   - **Surging 📈:** Growth $> 25\%$.
   - **Stable ➡️:** Growth $\ge 0\%$.
   - **Cooling 📉:** Growth $< 0\%$.

### 4. Engagement Rate Clarification
Standard Engagement Rate (ERR) requires private impression data:
$$\text{ERR} = \left(\frac{\text{Likes} + \text{Comments} + \text{Shares}}{\text{Impressions}}\right) \times 100$$
Because public social exports omit impression telemetry, our system calculates **Average Interactions per Post** and explicitly discloses this distinction in the interface.

---

## 💻 Installation & Local Execution

### Prerequisites
- Python 3.9, 3.10, 3.11, or 3.12
- Git

### 1. Clone or Open the Repository
```bash
git clone https://github.com/<your-username>/social-media-sentiment-analysis.git
cd social-media-sentiment-analysis
```

### 2. Create and Activate a Virtual Environment

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows (Command Prompt / PowerShell):**
```cmd
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Download NLTK VADER Lexicon
```bash
python -c "import nltk; nltk.download('vader_lexicon')"
```

### 5. Launch the Streamlit Dashboard
```bash
streamlit run app.py
```
The application will launch automatically in your default browser at:
`http://localhost:8501`

---

## 🧪 Running the Automated Test Suite

The project includes 29 automated unit and regression tests written in `pytest`.

```bash
# Run all tests with verbose output
pytest -v

# Run tests with test summary
pytest -v --tb=short
```

### Test Suite Coverage:
- `tests/test_data_loader.py`: Verifies CSV parsing, schema checking, and missing-column exceptions.
- `tests/test_preprocessing.py`: Validates URL stripping, mention removal, text normalization, deduplication, date coercion, and empty dataset safety.
- `tests/test_sentiment.py`: Tests positive, negative, and neutral classifications, threshold boundaries, empty string handling, and fallback resilience.
- `tests/test_trends.py`: Verifies volume aggregation, top hashtags, engagement summaries, and the recent-vs-baseline velocity algorithm.

---

## 📤 How to Upload a Custom Dataset

1. Prepare a CSV file containing the standard schema headers:
   `post_id,platform,username,post_text,hashtags,likes,comments,shares,post_date,topic`
2. Open the dashboard sidebar and select **Upload Custom CSV**.
3. Drag and drop your file into the file uploader.
4. If your file is missing columns, an actionable error banner will display the missing headers.
5. Click **Download CSV Template** in the sidebar to download a pre-formatted example.

---

## 🚀 Deployment Guide (Streamlit Community Cloud)

Follow these step-by-step instructions to deploy the project online for free:

### Step 1: Create a GitHub Repository
1. Log in to [GitHub](https://github.com).
2. Click **New Repository**.
3. Name the repository: `social-media-sentiment-analysis`.
4. Choose **Public** and leave "Add a README" unchecked (we already have one).
5. Click **Create repository**.

### Step 2: Push Local Code to GitHub
Open your terminal in the `social-media-sentiment-analysis` folder:
```bash
git init
git add .
git commit -m "feat: complete social media sentiment and trend analysis dashboard"
git branch -M main
git remote add origin https://github.com/<your-github-username>/social-media-sentiment-analysis.git
git push -u origin main
```

### Step 3: Deploy to Streamlit Community Cloud
1. Navigate to [share.streamlit.io](https://share.streamlit.io/) and sign in with your GitHub account.
2. Click **New app**.
3. In the deployment configuration dialog:
   - **Repository:** `<your-github-username>/social-media-sentiment-analysis`
   - **Branch:** `main`
   - **Main file path:** `app.py`
4. Click **Deploy!**.

### Step 4: Verify and Troubleshoot
- Streamlit Cloud will install dependencies from `requirements.txt` and boot the app.
- Once deployed, you will receive a public URL (e.g. `https://<your-app>.streamlit.app`).
- If VADER lexicon is needed during cloud startup, `src/sentiment_analysis.py` automatically downloads it or switches to the safe heuristic fallback.

---

## 🔮 Limitations & Future Enhancements

### Current Limitations
- **Language Scope:** VADER is optimized primarily for English-language social posts.
- **Sarcasm Detection:** Lexicon-based methods can occasionally misinterpret nuanced sarcasm or irony without transformer context.
- **Impression Telemetry:** True engagement rate (ERR) requires access to private account impressions.

### Future Enhancements
- **Transformer Fine-Tuning:** Integrate fine-tuned RoBERTa / BERT models for advanced multi-lingual and sarcasm detection.
- **Live Social Media APIs:** Connect official API endpoints (X API v2, YouTube Data API v3) for real-time live streaming ingestion.
- **Geo-Spatial Mapping:** Add geographical location tagging and interactive choropleth maps.
- **Automated Alerts:** Send email or webhook notifications when negative sentiment spikes above threshold.

---

## 📸 Screenshots

*(To add screenshots to your repository, capture images of the running dashboard, save them into `assets/`, and update the markdown links below.)*

| Executive Overview & KPI Metrics | Temporal Trends & Trajectories |
| :---: | :---: |
| `![Overview](assets/screenshot_overview.png)` | `![Temporal](assets/screenshot_temporal.png)` |

| Trending Keywords & Hashtags | Interactive Word Cloud |
| :---: | :---: |
| `![Trending](assets/screenshot_trending.png)` | `![Wordcloud](assets/screenshot_wordcloud.png)` |

---

## 📄 License & Attribution

This project was developed as a BCA Data Analytics internship capstone project. Open-sourced under the [MIT License](LICENSE).
