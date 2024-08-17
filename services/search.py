from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import faiss
import numpy as np
from dependencies.model_factory import embedding_model
from config.path import STORAGE_FILE_PATH
import logging

def search_with_just_keyword(keyword):

    # 전체 로드
    df = pd.read_csv(STORAGE_FILE_PATH)

    logging.error(df['text'])
    # 1. 키워드로 필터링
    # text, summary, tags 컬럼에서 키워드가 포함된 문서를 필터링
    filtered_df = df[df['text'].str.contains(keyword, na=False) |
                    df['summary'].str.contains(keyword, na=False) |
                    df['tags'].str.contains(keyword, na=False)].copy()

    search_result = []
    # 필터링된 결과 출력
    if not filtered_df.empty:
        for num, data in enumerate(filtered_df, start=1):
            cur_row = {}
            for index, row in filtered_df.iterrows():
                cur_row['file_names'] = row['file_names']
                cur_row['summary'] = row['summary']
                cur_row['tags'] = row['tags']
            search_result.append(cur_row)

    return search_result    
    
    
def search_document(documents, query, top_k=3):
    # 문서 임베딩 생성
    document_embeddings = embedding_model.encode(documents, convert_to_tensor=True)
    
    # 문서 임베딩 인덱스에 추가
    dimension = document_embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)  # L2 거리 인덱스 사용
    index.add(document_embeddings.cpu().numpy())
    
    # 쿼리 임베딩 생성
    query_embedding = embedding_model.encode(query, convert_to_tensor=True)
    
    # 유사도 계산
    distances, indices = index.search(query_embedding.cpu().numpy(), top_k)
    print("Query:", query)
    print("\nTop 3 most similar documents:")
    for i, idx in enumerate(indices[0]):
        print(f"Document {i + 1}: {documents[idx]} (Distance: {distances[0][i]})")

