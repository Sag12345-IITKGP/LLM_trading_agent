import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np

from workflow import workflow

st.set_page_config(
    page_title="Trading Support Framework",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
body, .main, .block-container {
    background: #f4f6fa !important;
    color: #222 !important;
}
.main-header {
    background: linear-gradient(90deg, #3a86ff 0%, #4361ee 100%);
    padding: 2rem 1rem 1.5rem 1rem;
    border-radius: 12px;
    margin-bottom: 2rem;
    text-align: center;
    color: #fff;
    box-shadow: 0 4px 16px rgba(0,0,0,0.10);
}
.metric-card {
    background: #fff;
    padding: 1.2rem;
    border-radius: 10px;
    text-align: center;
    color: #222;
    margin-bottom: 1rem;
    border: 1px solid #e0e6ed;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
}
.analysis-card {
    background: #fff;
    padding: 1.2rem;
    border-radius: 10px;
    margin-bottom: 1rem;
    color: #222;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
}
.news-card {
    background: #fff;
    padding: 1rem;
    border-radius: 8px;
    margin-bottom: 1rem;
    color: #222;
    border-left: 4px solid #3a86ff;
}
.risk-high {background: #ff4d4f; color: #fff;}
.risk-medium {background: #faad14; color: #fff;}
.risk-low {background: #52c41a; color: #fff;}
.stButton > button {
    background: linear-gradient(90deg, #3a86ff 0%, #4361ee 100%);
    color: #fff;
    border: none;
    border-radius: 5px;
    padding: 0.5rem 1rem;
    font-weight: bold;
}
.stButton > button:hover {
    background: linear-gradient(90deg, #4361ee 0%, #3a86ff 100%);
    transform: translateY(-2px);
    box-shadow: 0 4px 8px rgba(0,0,0,0.12);
}
.stSelectbox > div > div > select {
    background-color: #fff;
    color: #222;
}
.stDataFrame, .stTable {background: #fff; color: #222;}
</style>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="main-header"><h1>📈 Professional Trading Support Dashboard</h1>'
    '<p>AI-Powered Multi-Agent Trading Intelligence</p></div>',
    unsafe_allow_html=True
)

# --- Sidebar ---
with st.sidebar:
    st.markdown('<h2 style="color:#fff;">Trading Control Panel</h2>', unsafe_allow_html=True)
    selected_stock = st.selectbox("Select Stock Symbol", ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN", "NVDA", "META"])
    timeframe = st.selectbox("Analysis Timeframe", ["1D", "1W", "1M", "3M", "6M", "1Y"])
    if st.button("🔄 Run Analysis"):
        st.rerun()
    

# --- Run Workflow ---
with st.spinner("Running multi-agent analysis..."):
    results = workflow(selected_stock)

# --- Top Metrics ---
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f'<div class="metric-card"><h4>Current Price</h4><h2>${np.random.uniform(150, 200):.2f}</h2></div>', unsafe_allow_html=True)
with col2:
    st.markdown(f'<div class="metric-card"><h4>Volume</h4><h2>{np.random.randint(10, 100)}M</h2></div>', unsafe_allow_html=True)
with col3:
    st.markdown(f'<div class="metric-card"><h4>Market Cap</h4><h2>${np.random.uniform(1, 3):.2f}T</h2></div>', unsafe_allow_html=True)
with col4:
    st.markdown(f'<div class="metric-card"><h4>AI Confidence</h4><h2>{np.random.randint(70, 98)}%</h2></div>', unsafe_allow_html=True)

# --- Price Chart ---
dates = pd.date_range(end=datetime.today(), periods=120)
prices = 150 + np.cumsum(np.random.randn(len(dates)) * 0.7)
df = pd.DataFrame({'Date': dates, 'Price': prices})

fig = go.Figure()
fig.add_trace(go.Scatter(x=df['Date'], y=df['Price'], mode='lines', name='Price', line=dict(color='#3a86ff', width=2)))
fig.update_layout(
    title=f"{selected_stock} Price Chart",
    xaxis_title="Date",
    yaxis_title="Price ($)",
    template="plotly_dark",
    height=350,
    margin=dict(l=10, r=10, t=40, b=10)
)
st.plotly_chart(fig, use_container_width=True)

# --- Research Report ---
st.markdown('<div class="analysis-card"><h3>📝 Research Report</h3></div>', unsafe_allow_html=True)
st.markdown(f"""
**Bullish Research:**  
{results['analyst_reports']['bullish_report']}

**Bearish Research:**  
{results['analyst_reports']['bearish_report']}
""")

# --- Analysis Section ---
st.markdown('<div class="analysis-card"><h3>🔍 Multi-Agent Analysis</h3></div>', unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    st.markdown("**Technical Analysis**")
    st.info(results['analyst_reports']['technical_report'])
    st.markdown("**Sentiment Analysis**")
    st.info(results['analyst_reports']['sentiment_report'])
with col2:
    st.markdown("**News Analysis**")
    st.info(results['analyst_reports']['news_report'])
    st.markdown("**Fundamental Analysis**")
    st.info(results['analyst_reports']['fundamentals_report'])

# --- Risk Analysis ---
st.markdown('<div class="analysis-card"><h3>⚠️ Risk Analysis</h3></div>', unsafe_allow_html=True)
risk_report = results['risk_report']
if risk_report:
    st.warning(risk_report)
else:
    st.info("No significant risk factors detected for this asset at this time.")

# --- Trading Recommendation ---
st.markdown('<div class="analysis-card"><h3>💡 AI Trading Recommendation</h3></div>', unsafe_allow_html=True)
st.success(f"**Trade Decision:** {results['trader_decision']}")

st.markdown("---")
st.markdown(
    "<small style='color:#888;'>Disclaimer: This dashboard is for demonstration purposes only. "
    "No investment advice is provided.</small>",
    unsafe_allow_html=True
)