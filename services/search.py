from sentence_transformers import SentenceTransformer, util
import faiss
import numpy as np

model = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS')

def search_query(documents, query, top_k=3):
    # 문서 임베딩 생성
    document_embeddings = model.encode(documents, convert_to_tensor=True)
    
    # 문서 임베딩 인덱스에 추가
    dimension = document_embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)  # L2 거리 인덱스 사용
    index.add(document_embeddings.cpu().numpy())
    
    # 쿼리 임베딩 생성
    query_embedding = model.encode(query, convert_to_tensor=True)
    
    # 유사도 계산
    distances, indices = index.search(query_embedding.cpu().numpy(), top_k)
    print("Query:", query)
    print("\nTop 3 most similar documents:")
    for i, idx in enumerate(indices[0]):
        print(f"Document {i + 1}: {documents[idx]} (Distance: {distances[0][i]})")
