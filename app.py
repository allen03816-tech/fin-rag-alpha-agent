import os
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime
from datetime import datetime, timezone
import streamlit as st
from rag_pipeline import TimeFencedFinancialRAG
from agent_core import FinAlphaAgent

st.set_page_config(page_title="Live Point-in-Time FinRAG Agent", layout="wide")
st.title("📈 Live Financial RAG & News Alpha Agent")
st.caption("Cloud-Proof Live Financial RSS + Metadata Time-Fencing to Eliminate Look-Ahead Bias")

SEARCH_QUERY_MAP = {
    "NVDA": "NVDA stock OR Nvidia AI",
    "AAPL": "AAPL stock OR Apple",
    "SPY": "S&P 500 OR Federal Reserve",
    "USD/JPY": "USD JPY OR Bank of Japan",
    "XAU/USD": "Gold price XAU"
}

with st.sidebar:
    st.header("⚙️ Configuration")
    user_key = st.text_input("Groq API Key (Optional if in Secrets)", type="password")
    if user_key:
        os.environ["GROQ_API_KEY"] = user_key
    st.markdown("---")
    st.markdown("**Live Pipeline Features:**\n"
                "- **Data Feed:** Real-time Financial RSS (Cloud-safe)\n"
                "- **Embeddings:** `BAAI/bge-small-en-v1.5`\n"
                "- **Guardrail:** Auto-synced UTC timestamp filtering")

@st.cache_data(ttl=300)
def fetch_live_rss_bundle(query_str: str, display_name: str):
    """Fetches today's/yesterday's real headlines via Google News RSS (never blocked on cloud IPs)."""
    encoded_q = urllib.parse.quote(query_str)
    rss_url = f"https://news.google.com/rss/search?q={encoded_q}&hl=en-US&gl=US&ceid=US:en"
    
    headlines = []
    context_docs = []
    try:
        req = urllib.request.Request(rss_url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            xml_data = resp.read()
        root = ET.fromstring(xml_data)
        
        for idx, item in enumerate(root.findall(".//item")[:10]):
            title = item.findtext("title", "").strip()
            pub_date_raw = item.findtext("pubDate", "")
            source_name = item.findtext("source", "Financial_Wire")
            
            try:
                dt = parsedate_to_datetime(pub_date_raw).astimezone(timezone.utc)
                ts_str = dt.strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                ts_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
                
            if title:
                headlines.append({"title": title, "ts": ts_str, "source": source_name})
                context_docs.append({
                    "text": f"[{display_name} Market Wire | {source_name}]: {title}",
                    "source": f"{source_name.replace(' ', '_')}_{idx+1}",
                    "publish_ts": ts_str
                })
    except Exception:
        pass
        
    return headlines, context_docs

if not os.environ.get("GROQ_API_KEY"):
    st.warning("⚠️ Please enter your free Groq API Key in the sidebar to run live inference.")
    st.stop()

col1, col2 = st.columns([1, 1.2])
with col1:
    st.subheader("1. Live Market Selector")
    ticker = st.selectbox("Asset / Currency Pair", list(SEARCH_QUERY_MAP.keys()))
    
    with st.spinner(f"Pulling today's live {ticker} headlines..."):
        headlines, live_docs = fetch_live_rss_bundle(SEARCH_QUERY_MAP[ticker], ticker)

    if headlines:
        options = [f"[{h['ts']} UTC] {h['title']}" for h in headlines]
        selected = st.selectbox("Select Today's / Yesterday's Live Headline", options)
        idx = options.index(selected)
        headline = st.text_area("Active Breaking Headline (Editable)", value=headlines[idx]["title"])
        as_of_time = st.text_input("As-Of Decision Timestamp (UTC)", value=headlines[idx]["ts"])
    else:
        st.info("RSS feed temporarily unavailable — enter a headline manually below:")
        as_of_time = st.text_input("As-Of Decision Timestamp (UTC)", 
                                   value=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"))
        headline = st.text_area("Breaking Headline", 
                                value="Stock market today: Dow, S&P 500 slip ahead of Anthropic IPO, Nvidia stock extends decline")

    run_btn = st.button("🚀 Generate Live Alpha Signal", type="primary")

with col2:
    st.subheader("2. Live Agent Reasoning & Output")
    if run_btn:
        with st.spinner("Indexing live context & running Point-in-Time inference..."):
            # Always include the user's active headline as a baseline doc so manual input never fails
            active_docs = live_docs.copy()
            active_docs.append({
                "text": f"Primary Breaking Headline for {ticker}: {headline}",
                "source": "Active_Terminal_Headline",
                "publish_ts": as_of_time
            })
            
            rag = TimeFencedFinancialRAG()
            rag.ingest_documents(active_docs)
            agent = FinAlphaAgent(rag)
            
            res = agent.run_pipeline(ticker, headline, as_of_time)
            st.metric(label=f"{res['ticker']} | {res['horizon']}", 
                      value=res['direction'], 
                      delta=f"Confidence: {res['confidence']*100:.0f}%")
            st.markdown(f"**Financial Rationale:** {res['rationale']}")
            st.markdown("**Verified Live Pre-Cutoff Citations:**")
            for c in res["citations"]:
                st.code(c)