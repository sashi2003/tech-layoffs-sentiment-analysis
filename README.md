# Tech Layoffs Sentiment Analysis

An end-to-end NLP project analyzing tech layoff trends and media sentiment from 2020 to 2026. 
Combines structured layoff data with transformer-based sentiment analysis and topic modeling 
to examine how the tech industry and media responded to six years of workforce disruption.

## Live Demo

**[Open the dashboard →](https://tech-layoffs-sentiment-analysis-9ey5lhpbwm7juzys4h4hsp.streamlit.app/)**

## Data

- 2,470 layoff events across 1,903 companies (2020–2026)
- 306 news articles with sentiment labels
- US labor market indicators (weekly)

## Findings

**The narrative shifted.** Topic modeling with BERTopic surfaced two dominant themes across the news coverage — corporate restructuring, and AI displacement. Both were largely dormant between 2023 and mid-2025, then resurged sharply, with the AI framing outpacing the restructuring framing. The same layoff spikes were being explained differently depending on the year.

**2023 was the peak.** 170,324 layoffs, driven by Microsoft, Google, and Amazon correcting COVID-era over-hiring. Amazon led all companies with 49,624 total layoffs across the period.

**Sentiment decoupled from layoffs after 2023.** The January 2023 spike coincided with a sharp drop in news sentiment, but from mid-2023 onward the relationship broke down: sentiment recovered while layoffs continued.

**Off-the-shelf sentiment models struggle with business news.** A pretrained RoBERTa sentiment model reached only 44% accuracy against labeled ground truth. It performed well on negative headlines (91/150) but poorly on positive ones (20/119), most often misclassifying them as neutral: a limitation of general-purpose models applied to domain-specific framing.

**Tech employment growth is the strongest correlate.** Year-over-year tech employment change showed the strongest relationship with layoff volume (r = −0.32). Job openings per unemployed worker correlated positively (r = +0.17), suggesting simultaneous hiring and firing — a bifurcated market.

## Method

Four notebooks, run in order:

1. **Data exploration** — cleaning, aggregation, trend analysis
2. **Sentiment analysis** — `cardiffnlp/twitter-roberta-base-sentiment-latest` applied to news headlines, evaluated against ground truth labels
3. **Topic modeling** — BERTopic with stopword filtering, topics-over-time analysis
4. **Correlation analysis** — monthly aggregation, correlation with US labor indicators

## Stack

Python, pandas, Plotly, HuggingFace Transformers, BERTopic, Streamlit, Docker

## Structure

- `notebooks/` — analysis notebooks
- `data/` — layoff events, news sentiment, labor indicators
- `app/` — Streamlit dashboard
- `Dockerfile` — containerized deployment

## Author

Sashi Praneeth Reddy Muthyala  
M.S. Data Analytics and Computational Social Science, UMass Amherst