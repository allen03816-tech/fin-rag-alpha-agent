import os
from groq import Groq
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate

class TradingSignal(BaseModel):
    ticker: str = Field(description="交易标的，如 USD/JPY 或 NVDA")
    horizon: str = Field(description="交易周期: Overnight 或 Intraday")
    direction: str = Field(description="交易方向: LONG, SHORT, 或 NEUTRAL")
    confidence: float = Field(description="置信度 0.0 到 1.0")
    rationale: str = Field(description="结合历史基本面与突发新闻的英文金融逻辑分析")
    citations: list[str] = Field(description="引用的文档名称与时间戳")

def get_available_groq_model(api_key: str) -> str:
    """Queries Groq's live model list and picks the best active text model."""
    client = Groq(api_key=api_key)
    active_ids = [m.id for m in client.models.list().data]
    
    # Preferred models in priority order (skipping audio/whisper/guard models)
    preferred = [
        "llama-3.3-70b-versatile",
        "meta-llama/llama-4-maverick-17b-128e-instruct",
        "meta-llama/llama-4-scout-17b-16e-instruct",
        "qwen/qwen3-32b",
        "llama-3.1-8b-instant",
    ]
    for model_id in preferred:
        if model_id in active_ids:
            return model_id
            
    # Fallback: pick any available chat model
    chat_models = [m for m in active_ids if not any(x in m.lower() for x in ["whisper", "guard", "tts"])]
    return chat_models[0]

class FinAlphaAgent:
    def __init__(self, rag_engine):
        self.rag = rag_engine
        api_key = os.environ.get("GROQ_API_KEY")
        self.model_name = get_available_groq_model(api_key)

        llm = ChatGroq(
            model=self.model_name,
            temperature=0,
            api_key=api_key
        )
        self.structured_llm = llm.with_structured_output(
            TradingSignal,
            method="json_mode",
        )

    def run_pipeline(self, ticker: str, headline: str, as_of_time: str) -> dict:
        # Step 1 [THINK & ACT]: 检索决策时点之前的历史宏观/财报依据
        query = f"{ticker} monetary policy earnings guidance risk: {headline}"
        context_docs = self.rag.retrieve_as_of(query=query, as_of_time=as_of_time, top_k=3)
        
        context_str = "\n".join([
            f"[{d.metadata['publish_ts']} | {d.metadata['source']}]: {d.page_content}"
            for d in context_docs
        ])

        # Step 2 [OBSERVE & DECIDE]: 生成防幻觉交易信号
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an institutional quantitative macro & equity analyst. "
                       "Generate a short-term trading signal strictly grounded in the provided "
                       "Point-in-Time Context and Breaking Headline. Do NOT use external post-cutoff knowledge. "
                       "If context is insufficient, output NEUTRAL. "
                       "Return only a valid JSON object with these fields: ticker (string), "
                       "horizon (Overnight or Intraday), direction (LONG, SHORT, or NEUTRAL), "
                       "confidence (number from 0.0 to 1.0), rationale (string), "
                       "and citations (array of strings). Do not use Markdown fences."),
            ("human", "Decision Timestamp (As-Of): {as_of_time}\n"
                      "Ticker: {ticker}\n"
                      "Breaking Headline: {headline}\n\n"
                      "Point-in-Time Retrieved Context:\n{context}")
        ])

        chain = prompt | self.structured_llm
        result: TradingSignal = chain.invoke({
            "as_of_time": as_of_time,
            "ticker": ticker,
            "headline": headline,
            "context": context_str if context_str else "No pre-cutoff documents available."
        })
        return result.model_dump()


       