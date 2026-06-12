import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="Tech Layoffs Sentiment Analysis",
    page_icon="📉",
    layout="wide"
)

# Load data
import os

@st.cache_data
def load_data():
    # Works both locally and on Streamlit Cloud
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    df_layoffs = pd.read_csv(os.path.join(base_path, 'data', 'layoffs_events.csv'))
    df_news = pd.read_csv(os.path.join(base_path, 'data', 'news_sentiment.csv'))
    df_us_labor = pd.read_csv(os.path.join(base_path, 'data', 'us_labor_indicators.csv'))
    
    df_layoffs['date'] = pd.to_datetime(df_layoffs['date'])
    df_news['date'] = pd.to_datetime(df_news['date'])
    df_us_labor['date'] = pd.to_datetime(df_us_labor['date'])
    df_layoffs['pct_workforce'] = df_layoffs['pct_workforce'].str.replace('%', '').astype(float)
    df_layoffs = df_layoffs.dropna(subset=['industry', 'country'])
    
    return df_layoffs, df_news, df_us_labor

df_layoffs, df_news, df_us_labor = load_data()

# Header
st.title("Tech Layoffs & Sentiment Analysis Dashboard")
st.markdown("*Analyzing 2,470 layoff events and media sentiment across 6 years of tech industry data (2020-2026)*")
st.markdown("---")

# Metrics row
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Layoffs Tracked", f"{df_layoffs['layoff_count'].sum():,.0f}")
col2.metric("Companies Affected", df_layoffs['company'].nunique())
col3.metric("Industries Covered", df_layoffs['industry'].nunique())
col4.metric("News Articles Analyzed", len(df_news))

st.markdown("---")

# Sidebar filters
st.sidebar.title("Filters")
industries = ['All'] + sorted(df_layoffs['industry'].dropna().unique().tolist())
selected_industry = st.sidebar.selectbox("Select Industry", industries)

countries = ['All'] + sorted(df_layoffs['country'].dropna().unique().tolist())
selected_country = st.sidebar.selectbox("Select Country", countries)

# Apply filters
filtered_df = df_layoffs.copy()
if selected_industry != 'All':
    filtered_df = filtered_df[filtered_df['industry'] == selected_industry]
if selected_country != 'All':
    filtered_df = filtered_df[filtered_df['country'] == selected_country]

# Add this warning
if len(filtered_df) == 0:
    st.warning("⚠️ No data available for the selected filters. Please adjust your selection.")
    st.stop()

st.sidebar.markdown("---")
st.sidebar.metric("Filtered Records", len(filtered_df))

# Section 1 - Layoffs over time
st.subheader("Layoff Trends Over Time")
monthly = filtered_df.copy()
monthly['month'] = monthly['date'].dt.to_period('M').astype(str)
monthly_layoffs = monthly.groupby('month')['layoff_count'].sum().reset_index()

fig1 = px.line(monthly_layoffs,
               x='month', y='layoff_count',
               title='Monthly Layoffs Over Time',
               labels={'layoff_count': 'Total Laid Off', 'month': 'Month'})
fig1.update_layout(xaxis_tickangle=-45)
st.plotly_chart(fig1, use_container_width=True)

# Section 2 - Top industries and companies side by side
st.subheader("Industry & Company Breakdown")
col1, col2 = st.columns(2)

with col1:
    industry_layoffs = filtered_df.groupby('industry')['layoff_count'].sum()
    industry_layoffs = industry_layoffs.sort_values(ascending=False).head(10).reset_index()
    fig2 = px.bar(industry_layoffs,
                  x='layoff_count', y='industry',
                  orientation='h',
                  title='Top 10 Industries',
                  color='layoff_count',
                  color_continuous_scale='reds')
    fig2.update_layout(yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig2, use_container_width=True)

with col2:
    company_layoffs = filtered_df.groupby('company')['layoff_count'].sum()
    company_layoffs = company_layoffs.sort_values(ascending=False).head(10).reset_index()
    fig3 = px.bar(company_layoffs,
                  x='layoff_count', y='company',
                  orientation='h',
                  title='Top 10 Companies',
                  color='layoff_count',
                  color_continuous_scale='blues')
    fig3.update_layout(yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig3, use_container_width=True)

st.markdown("---")

# Section 3 - News Sentiment
st.subheader("News Sentiment Analysis")

col3, col4 = st.columns(2)

with col3:
    sentiment_counts = df_news['sentiment_cat'].value_counts().reset_index()
    sentiment_counts.columns = ['sentiment', 'count']
    fig4 = px.bar(sentiment_counts,
                  x='sentiment', y='count',
                  title='News Sentiment Distribution',
                  color='sentiment',
                  color_discrete_map={
                      'negative': 'tomato',
                      'neutral': 'gray',
                      'positive': 'steelblue'
                  })
    st.plotly_chart(fig4, use_container_width=True)

with col4:
    df_news['month'] = df_news['date'].dt.to_period('M').astype(str)
    sentiment_time = df_news.groupby(['month', 'sentiment_cat'])['sentiment'].count().reset_index()
    sentiment_time.columns = ['month', 'sentiment', 'count']
    fig5 = px.line(sentiment_time,
                   x='month', y='count',
                   color='sentiment',
                   title='Sentiment Trends Over Time',
                   color_discrete_map={
                       'negative': 'tomato',
                       'neutral': 'gray',
                       'positive': 'steelblue'
                   })
    fig5.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig5, use_container_width=True)

st.markdown("---")

# Section 4 - Layoffs vs Sentiment
st.subheader("Layoffs vs Media Sentiment")

monthly_layoffs_all = df_layoffs.copy()
monthly_layoffs_all['month'] = monthly_layoffs_all['date'].dt.to_period('M').astype(str)
monthly_layoffs_all = monthly_layoffs_all.groupby('month')['layoff_count'].sum().reset_index()

news_monthly = df_news.groupby('month').agg({
    'sentiment': 'mean',
    'title': 'count'
}).reset_index()
news_monthly.columns = ['month', 'avg_sentiment', 'article_count']

combined = monthly_layoffs_all.merge(news_monthly, on='month', how='inner')

fig6 = make_subplots(specs=[[{"secondary_y": True}]])
fig6.add_trace(
    go.Bar(x=combined['month'], y=combined['total_layoffs'] if 'total_layoffs' in combined.columns else combined['layoff_count'],
           name='Total Layoffs', marker_color='tomato', opacity=0.7),
    secondary_y=False
)
fig6.add_trace(
    go.Scatter(x=combined['month'], y=combined['avg_sentiment'],
               name='Avg Sentiment', line=dict(color='steelblue', width=2)),
    secondary_y=True
)
fig6.update_layout(title='Tech Layoffs vs News Sentiment', xaxis_tickangle=-45)
fig6.update_yaxes(title_text='Total Layoffs', secondary_y=False)
fig6.update_yaxes(title_text='Avg Sentiment Score', secondary_y=True)
st.plotly_chart(fig6, use_container_width=True)

st.markdown("---")

# Section 5 - AI vs Non-AI
st.subheader("AI vs Non-AI Company Layoffs")
monthly_layoffs_all2 = filtered_df.copy()
monthly_layoffs_all2['month'] = monthly_layoffs_all2['date'].dt.to_period('M').astype(str)
ai_comparison = monthly_layoffs_all2.groupby(
    ['month', 'is_ai_company'])['layoff_count'].sum().reset_index()

fig7 = px.line(ai_comparison,
               x='month', y='layoff_count',
               color='is_ai_company',
               title='AI vs Non-AI Company Layoffs Over Time',
               labels={'layoff_count': 'Total Laid Off', 'is_ai_company': 'AI Company'})
fig7.update_layout(xaxis_tickangle=-45)
st.plotly_chart(fig7, use_container_width=True)

st.markdown("---")
st.caption("Data source: Tech Layoffs Dataset (2020-2026) | Built by Sashi Praneeth Reddy Muthyala | M.S. DACSS, UMass Amherst")