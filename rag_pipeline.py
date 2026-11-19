#pip install -r requirements.txt
from datetime import datetime
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

class TimeFencedFinancialRAG:
    def __init__(self):
        # 免费轻量级开源金融/英文语义模型，首次加载约 130MB，完全免费无需 API Key
        self.embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en-v1.5",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
        self.vector_store = None

    def ingest_documents(self, raw_docs: list[dict]):
        """
        写入带有严格发布时间戳的财报/央行纪要片段
        """
        splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=80)
        docs = []
        for item in raw_docs:
            ts_unix = datetime.strptime(item["publish_ts"], "%Y-%m-%d %H:%M:%S").timestamp()
            for chunk in splitter.split_text(item["text"]):
                docs.append(Document(
                    page_content=chunk,
                    metadata={
                        "source": item["source"],
                        "publish_ts": item["publish_ts"],
                        "ts_unix": ts_unix  # 核心：Point-in-Time 时间戳锁
                    }
                ))
        self.vector_store = FAISS.from_documents(docs, self.embeddings)

    def retrieve_as_of(self, query: str, as_of_time: str, top_k: int = 3) -> list[Document]:
        """
        仅检索严格早于或等于 as_of_time 的历史文档，彻底阻断 Look-ahead Bias
        """
        cutoff_unix = datetime.strptime(as_of_time, "%Y-%m-%d %H:%M:%S").timestamp()
        candidates = self.vector_store.similarity_search(query, k=top_k * 5)
        valid_docs = [doc for doc in candidates if doc.metadata["ts_unix"] <= cutoff_unix]
        return valid_docs[:top_k]