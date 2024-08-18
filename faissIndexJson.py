import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer

# 임베딩 모델 초기화
embedding_model = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS')

# JSON 파일이 저장된 경로
json_folder_path = 'C:/ScreenShotDB/Image_data'

# 텍스트 데이터를 저장할 리스트
documents = []
metadata = []

# 모든 JSON 파일을 순회하면서 텍스트를 수집
for root, _, files in os.walk(json_folder_path):
    for file in files:
        if file.endswith('_ocr_results.json'):  # OCR 결과 파일만 처리
            json_path = os.path.join(root, file)
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    documents.extend(data)
                    metadata.extend([{"file": json_path, "text": text} for text in data])

# 텍스트를 임베딩
print("Generating embeddings...")
document_embeddings = embedding_model.encode(documents, convert_to_tensor=False).tolist()  # 리스트 형식으로 변환

# 임베딩과 텍스트를 함께 JSON으로 저장
output_data = []
for text, embedding, meta in zip(documents, document_embeddings, metadata):
    output_data.append({
        "text": text,
        "embedding": embedding,
        "metadata": meta
    })

with open('embeddings_with_texts.json', 'w', encoding='utf-8') as json_file:
    json.dump(output_data, json_file, ensure_ascii=False, indent=4)

print("Embeddings and texts saved successfully as JSON.")
