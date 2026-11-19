📈 Point-in-Time Financial RAG & News Alpha Agent

A quantitative Retrieval-Augmented Generation (RAG) and structured news-alpha system designed to generate short-term directional trading signals (LONG / SHORT / NEUTRAL) from live financial news wires while enforcing strict Point-in-Time (PIT) metadata time-fencing to eliminate look-ahead bias.

🎯 Problem Statement & Quantitative Motivation

Standard LLM-based financial RAG pipelines suffer from a critical flaw in quantitative research and backtesting: look-ahead bias (future data leakage). When querying a vector database during historical simulation or intraday event studies, standard similarity search retrieves documents published after the decision timestamp, invalidating out-of-sample signal evaluation.

This project solves this by decoupling semantic retrieval from causal availability:

Unix Timestamp Metadata Indexing: Every ingested financial wire, filing, or macro release is tagged with its exact UTC publication timestamp (ts_unix) at ingestion time.

Pre-Inference Causal Filtering (ts_unix <= cutoff_unix): At query time, candidate chunks retrieved via FAISS cosine similarity are strictly filtered against the simulated or live decision timestamp before entering the LLM context window.

Deterministic Structured Output: Enforces a strict Pydantic JSON schema (TradingSignal) with calibrated confidence scores and verifiable pre-cutoff source citations.

🏗️ System Architecture

┌─────────────────────────────┐      ┌──────────────────────────────────┐
│   Live Financial RSS Wire   │      │    Historical Macro / Filings    │
│ (Equities, FX, Commodities) │      │   (Central Bank & 10-Q Notes)    │
└──────────────┬──────────────┘      └────────────────┬─────────────────┘
               │                                      │
               └──────────────────┬───────────────────┘
                                  ▼
               ┌──────────────────────────────────────┐
               │   Chunking & BGE Vector Embedding    │
               │     (BAAI/bge-small-en-v1.5)         │
               └──────────────────┬───────────────────┘
                                  ▼
               ┌──────────────────────────────────────┐
               │       FAISS Local Vector Store       │
               │   + Unix Timestamp Metadata Fence    │
               │       (ts_unix <= cutoff_unix)       │
               └──────────────────┬───────────────────┘
                                  ▼
               ┌──────────────────────────────────────┐
               │    LangChain Structured JSON Agent   │
               │     (Groq Low-Latency Inference)     │
               └──────────────────┬───────────────────┘
                                  ▼
               ┌──────────────────────────────────────┐
               │   Structured Alpha Signal Output     │
               │  Direction | Confidence | Citations  │
               └──────────────────────────────────────┘


📂 Repository Structure

fin-rag-alpha-agent/
├── app.py              # Streamlit interactive UI & cloud-safe live RSS news ingestion
├── agent_core.py       # LangChain structured-output agent & Groq model auto-routing
├── rag_pipeline.py     # FAISS vector store, BGE embeddings & Unix timestamp time-fence
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation


🚀 Quickstart (Local & Cloud Deployment)

1. Local Installation

git clone https://github.com/<your-username>/fin-rag-alpha-agent.git
cd fin-rag-alpha-agent
pip install -r requirements.txt


Set your free Groq API Key as an environment variable and launch the Streamlit dashboard:

# macOS / Linux
export GROQ_API_KEY="your_groq_api_key_here"
streamlit run app.py

# Windows (CMD)
set GROQ_API_KEY=your_groq_api_key_here
streamlit run app.py


2. Streamlit Community Cloud Deployment

Push this repository to GitHub with app.py at the repository root.

Connect the repository on share.streamlit.io.

Add GROQ_API_KEY = "your_groq_api_key_here" under Advanced Settings → Secrets and deploy.

🛠️ Tech Stack

LLM Orchestration: LangChain (langchain-core, langchain-groq), Pydantic v2 (json_mode structured validation)

Embeddings & Vector Search: HuggingFace BAAI/bge-small-en-v1.5, sentence-transformers, faiss-cpu

Market Data Ingestion: Cloud-resilient XML/RSS financial news parsing (urllib, xml.etree) with UTC timestamp normalization

Frontend: Streamlit

⚠️ Disclaimer

This software is built for quantitative research, financial engineering demonstrations, and educational purposes only. Generated signals do not constitute investment or financial advice.
