"""
Tiện ích để tải và xử lý dữ liệu cho RAG pipeline.

Cách dùng:
    from utils.data_loader import load_knowledge_base, split_text, build_vectorstore

    text        = load_knowledge_base()
    chunks      = split_text(text, chunk_size=500, chunk_overlap=50)
    vectorstore = build_vectorstore(chunks, embeddings)
"""
import re
import time
from pathlib import Path


_GEMINI_EMBEDDING_BATCH_SIZE = 100
_MAX_RATE_LIMIT_RETRIES = 3


def _get_retry_delay(error: Exception) -> float | None:
    """Extract Gemini's suggested retry delay from a rate-limit error."""
    message = str(error)
    for pattern in (
        r"retry in\s+(\d+(?:\.\d+)?)s",
        r"retryDelay.*?(\d+(?:\.\d+)?)s",
    ):
        match = re.search(pattern, message, flags=re.IGNORECASE)
        if match:
            return float(match.group(1))
    return None


def _embed_gemini_documents(chunks: list[str], embeddings) -> list[list[float]]:
    """Embed Gemini documents in quota-sized batches with bounded 429 retries."""
    vectors = []

    for start in range(0, len(chunks), _GEMINI_EMBEDDING_BATCH_SIZE):
        batch = chunks[start : start + _GEMINI_EMBEDDING_BATCH_SIZE]

        for attempt in range(_MAX_RATE_LIMIT_RETRIES + 1):
            try:
                vectors.extend(embeddings.embed_documents(batch))
                break
            except Exception as error:
                message = str(error)
                retry_delay = _get_retry_delay(error)
                is_rate_limited = "429" in message or "RESOURCE_EXHAUSTED" in message

                if (
                    not is_rate_limited
                    or retry_delay is None
                    or attempt == _MAX_RATE_LIMIT_RETRIES
                ):
                    raise

                # Add a small buffer so the next request is outside the quota window.
                wait_seconds = int(retry_delay) + 2
                print(
                    f"⏳ Gemini embedding đạt giới hạn quota. "
                    f"Thử lại sau {wait_seconds} giây ..."
                )
                time.sleep(wait_seconds)

    return vectors


def load_knowledge_base(path: str = None) -> str:
    """
    Đọc file knowledge base và trả về nội dung dạng chuỗi.

    Args:
        path: đường dẫn tới file text.
              Mặc định: data/knowledge_base.txt (thư mục gốc của project)

    Returns:
        Nội dung file dưới dạng str
    """
    if path is None:
        path = Path(__file__).parent.parent.parent / "data" / "knowledge_base.txt"
    return Path(path).read_text(encoding="utf-8")


def split_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> list:
    """
    Chia văn bản thành các đoạn nhỏ (chunks) để index.

    Dùng RecursiveCharacterTextSplitter — tách ưu tiên theo đoạn văn, câu, rồi ký tự.

    Args:
        text         : văn bản cần chia
        chunk_size   : số ký tự tối đa mỗi chunk (mặc định: 500)
        chunk_overlap: số ký tự chồng lên nhau giữa 2 chunks liên tiếp (mặc định: 50)

    Returns:
        list[str] — danh sách các chuỗi chunk
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_text(text)


def build_vectorstore(chunks: list, embeddings):
    """
    Tạo FAISS vectorstore từ danh sách chunks và embeddings.

    Args:
        chunks    : list[str] — danh sách text chunks đã chia
        embeddings: Embeddings instance (từ get_embeddings())

    Returns:
        FAISS vectorstore đã được index và sẵn sàng dùng để retrieve
    """
    from langchain_community.vectorstores import FAISS

    print(f"🔨 Đang tạo FAISS index từ {len(chunks)} chunks ...")

    is_gemini = embeddings.__class__.__module__.startswith("langchain_google_genai")
    if is_gemini and len(chunks) > _GEMINI_EMBEDDING_BATCH_SIZE:
        vectors = _embed_gemini_documents(chunks, embeddings)
        vectorstore = FAISS.from_embeddings(zip(chunks, vectors), embeddings)
    else:
        vectorstore = FAISS.from_texts(chunks, embeddings)

    print("✅ FAISS vectorstore đã sẵn sàng.")
    return vectorstore
