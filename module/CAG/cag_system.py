import time
import tiktoken
import openai
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os
import faiss
import pickle
import numpy as np

load_dotenv()

EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL')

class DocumentCAG:
    def __init__(self, index_path, mappings_path, model_name=EMBEDDING_MODEL):
        self.model = SentenceTransformer(model_name)

        # Tải FAISS index
        self.index = faiss.read_index(index_path)

        # Tải mappings
        with open(mappings_path, 'rb') as f:
            self.mappings = pickle.load(f)

        self.index_to_chunk = self.mappings["index_to_chunk"]
        self.chunk_metadata = self.mappings["chunk_metadata"]
        self.texts = self.mappings["texts"]

        # Cài đặt efSearch cho tìm kiếm
        self.index.hnsw.efSearch = 40

        # Khởi tạo bộ nhớ đệm cho các trích dẫn cuối cùng
        self.last_citations = []
        self.last_retrieved_chunks = {}

    def retrieve_relevant_chunks(self, query, top_k=10, threshold=0.0):
        """Truy xuất các chunk liên quan nhất"""
        # Tạo embedding cho query
        query_vector = self.model.encode(query)
        query_vector = np.array([query_vector], dtype=np.float32)

        # Tìm kiếm các chunk liên quan
        distances, indices = self.index.search(query_vector, top_k)

        # Xử lý kết quả
        results = []
        retrieved_chunks = {}
        document_tracker = set()  # Theo dõi tài liệu đã truy xuất

        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx == -1:
                continue

            chunk_id = self.index_to_chunk[idx]
            metadata = self.chunk_metadata[chunk_id]
            text = self.texts[idx]

            # Tính điểm tương đồng
            similarity = 1 / (1 + float(dist))

            if similarity < threshold:
                continue

            # Tạo document_id duy nhất nếu chunk thuộc cùng tài liệu
            doc_id = metadata["document_id"]
            parent_document = metadata.get("parent_document", "")

            # Theo dõi các đoạn thuộc cùng một tài liệu
            if doc_id not in document_tracker:
                document_tracker.add(doc_id)
                citation_id = len(document_tracker)
            else:
                # Tìm citation_id của tài liệu đã có
                for res in results:
                    if res["metadata"]["document_id"] == doc_id:
                        citation_id = res["citation_id"]
                        break

            result = {
                "chunk_id": chunk_id,
                "citation_id": citation_id,
                "text": text,
                "similarity": similarity,
                "metadata": metadata
            }

            results.append(result)
            retrieved_chunks[chunk_id] = result

        # Lưu trữ truy vấn cuối cùng
        self.last_retrieved_chunks = retrieved_chunks

        # Tạo danh sách trích dẫn
        citations = []
        cited_docs = {}

        for result in results:
            doc_id = result["metadata"]["document_id"]
            if doc_id not in cited_docs:
                cited_docs[doc_id] = {
                    "citation_id": result["citation_id"],
                    "document_id": doc_id,
                    "title": result["metadata"].get("title", ""),
                    "author": result["metadata"].get("author", ""),
                    "source": result["metadata"].get("source", ""),
                    "date": result["metadata"].get("date", ""),
                    "category": result["metadata"].get("category", ""),
                }
                citations.append(cited_docs[doc_id])

        self.last_citations = citations

        return results, citations

    def format_context_with_citations(self, retrieved_chunks, citations, max_tokens=8000):
        """Định dạng ngữ cảnh với trích dẫn"""
        context = ""

        # Sắp xếp theo độ tương đồng
        sorted_chunks = sorted(retrieved_chunks, key=lambda x: x["similarity"], reverse=True)

        # Sắp xếp lai theo tài liệu
        doc_chunks = {}
        for chunk in sorted_chunks:
            doc_id = chunk["metadata"]["document_id"]
            if doc_id not in doc_chunks:
                doc_chunks[doc_id] = []
            doc_chunks[doc_id].append(chunk)

        # Tạo ngữ cảnh
        for doc_id, chunks in doc_chunks.items():
            citation_id = chunks[0]["citation_id"]

            # Sắp xếp chunk theo thứ tự xuất hiện trong tài liệu
            chunks.sort(key=lambda x: x["metadata"].get("chunk_index", 0))

            doc_header = f"\n--- Document [{citation_id}]: {chunks[0]['metadata'].get('title', '')} ---\n"
            context += doc_header

            # Thêm nội dung từng chunk
            for chunk in chunks:
                context += chunk["text"] + "\n\n"

        return context

    def format_references(self, citations):
        """Định dạng danh sách tài liệu tham khảo"""
        references = "\n\nTài liệu tham khảo:\n"

        for citation in sorted(citations, key=lambda x: x["citation_id"]):
            ref = f"[{citation['citation_id']}] "

            if citation['title']:
                ref += f"\"{citation['title']}\""

            if citation['author']:
                ref += f", {citation['author']}"

            if citation['source']:
                ref += f", {citation['source']}"

            if citation['date']:
                ref += f", {citation['date']}"

            references += ref + "\n"

        return references

    def process_query(self, query, max_tokens=8000):
        """Xử lý truy vấn và trả về ngữ cảnh và trích dẫn"""
        # Truy xuất chunks liên quan
        retrieved_chunks, citations = self.retrieve_relevant_chunks(query, top_k=15)

        # Định dạng ngữ cảnh
        context = self.format_context_with_citations(retrieved_chunks, citations, max_tokens)

        # Định dạng danh sách tham khảo
        references = self.format_references(citations)

        return {
            "context": context,
            "references": references,
            "retrieved_chunks": retrieved_chunks,
            "citations": citations
        }