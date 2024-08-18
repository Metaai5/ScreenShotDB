import os
import json
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# 임베딩 모델 초기화
embedding_model = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS')

# JSON 파일이 저장된 경로
json_folder_path = 'C:/ScreenShotDB/Image_data'

# 텍스트 데이터를 저장할 리스트
documents = []

# 모든 JSON 파일을 순회하면서 텍스트를 수집
for root, _, files in os.walk(json_folder_path):
    for file in files:
        if file.endswith('_ocr_results.json'):  # OCR 결과 파일만 처리
            json_path = os.path.join(root, file)
            print(f"Reading file: {json_path}")  # 디버깅 메시지 추가
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        documents.extend(data)
                    else:
                        print(f"File {json_path} does not contain a list. Skipping.")
            except Exception as e:
                print(f"Error reading {json_path}: {e}")

# 문서가 비어 있지 않은지 확인
if not documents:
    print("No documents found for embedding. Exiting.")
else:
    # 텍스트를 임베딩
    print("Generating embeddings...")
    document_embeddings = embedding_model.encode(documents, convert_to_tensor=False)

    # FAISS 인덱스 생성
    dimension = document_embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)  # L2 거리 인덱스 사용
    index.add(np.array(document_embeddings).astype('float32'))

    # 인덱스를 파일로 저장
    faiss.write_index(index, "faiss_index.bin")

    print("FAISS index saved successfully.")
