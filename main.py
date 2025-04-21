from fastapi import FastAPI, HTTPException, Query, Body
from pydantic import BaseModel, Field
import uvicorn
from typing import List, Optional
import os
import time


class QueryRequest(BaseModel):
    query: str = Field(..., description="Câu truy vấn cần tìm kiếm thông tin")
    top_k: Optional[int] = Field(5, description="Số lượng tài liệu tương tự nhất")
    threshold: Optional[float] = Field(0.0, description="Ngưỡng điểm tương đồng tối thiểu")


class SearchResult(BaseModel):
    id: int
    document: str
    similarity: float


class QueryResponse(BaseModel):
    answer: str
    retrieved_documents: List[SearchResult]
    processing_time: float


app = FastAPI(
    title="CAG API với SentenceTransformer & FAISS HNSW",
    description="API truy xuất và trả lời dựa trên tài liệu với SentenceTransformer và FAISS HNSW",
    version="1.0.0"
)

# Đường dẫn đến index và documents
INDEX_PATH = os.environ.get("INDEX_PATH", "data/hnsw.index")
DOCUMENTS_PATH = os.environ.get("DOCUMENTS_PATH", "data/documents.pkl")
MODEL_NAME = os.environ.get("MODEL_NAME", "paraphrase-multilingual-MiniLM-L12-v2")
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4o")


# Khởi tạo các thành phần
@app.on_event("startup")
async def startup_event():
    global retriever, cag_system

    # Khởi tạo retriever
    from modules.retriever import SentenceTransformerRetriever
    retriever = SentenceTransformerRetriever(
        index_path=INDEX_PATH,
        documents_path=DOCUMENTS_PATH,
        model_name=MODEL_NAME
    )

    # Khởi tạo CAG system
    from modules.cag import CAGWithSentenceTransformer
    cag_system = CAGWithSentenceTransformer(
        retriever=retriever,
        model_name=LLM_MODEL
    )

    print(f"Đã khởi tạo CAG system với {retriever.index.ntotal} tài liệu")


@app.get("/health")
async def health_check():
    """Kiểm tra trạng thái hoạt động của API"""
    return {"status": "healthy", "timestamp": time.time()}


@app.post("/search", response_model=List[SearchResult])
async def search_documents(request: QueryRequest):
    """Tìm kiếm tài liệu tương tự với truy vấn"""
    start_time = time.time()

    try:
        results = retriever.retrieve(
            query=request.query,
            top_k=request.top_k,
            threshold=request.threshold
        )

        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """Xử lý truy vấn và trả lời dựa trên tài liệu"""
    start_time = time.time()

    try:
        # Truy xuất tài liệu
        retrieved_docs = retriever.retrieve(
            query=request.query,
            top_k=request.top_k,
            threshold=request.threshold
        )

        # Xử lý truy vấn
        answer = cag_system.process_query(request.query)

        # Đo thời gian xử lý
        processing_time = time.time() - start_time

        return {
            "answer": answer,
            "retrieved_documents": retrieved_docs,
            "processing_time": processing_time
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
async def get_metrics():
    """Lấy thông tin metrics của hệ thống"""
    return {
        "total_documents": retriever.index.ntotal,
        "cag_metrics": cag_system.metrics
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)