import faiss
import numpy as np

# FAISS 인덱스 로드
index = faiss.read_index('faiss_index.bin')

# 메타데이터 로드
with open('metadata.json', 'r', encoding='utf-8') as meta_file:
    metadata = json.load(meta_file)

# 검색할 쿼리
query = "검색하고 싶은 텍스트"
query_embedding = embedding_model.encode(query, convert_to_tensor=False)

# 유사 문서 검색
top_k = 5  # 상위 5개 결과
distances, indices = index.search(np.array([query_embedding]), top_k)

print("Query:", query)
print("\nTop 5 most similar documents:")
for i, idx in enumerate(indices[0]):
    print(f"\nDocument {i + 1}: {documents[idx]} (Distance: {distances[0][i]})")
    print(f"File: {metadata[idx]['file']}")
