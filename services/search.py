import faiss
import pandas as pd
# from services.model import embedding_model
from model import embedding_model

def search_query(query, top_k=3):
    df = pd.read_csv('data/result_texts.csv')
    document_text = df['text'].tolist()
    # 문서 임베딩 생성
    document_embeddings = embedding_model.encode(document_text, convert_to_tensor=True)
    
    # 문서 임베딩 인덱스에 추가
    dimension = document_embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)  # L2 거리 인덱스 사용
    index.add(document_embeddings.cpu().numpy())
    
    # 쿼리 임베딩 생성
    query_embedding = embedding_model.encode([query], convert_to_tensor=True)
    
    # 유사도 계산
    D, I = index.search(query_embedding.cpu().numpy(), top_k)
    results = [(df.iloc[i]['uuid'], D[0][idx]) for idx, i in enumerate(I[0])]
    return results

def show_search_result(results):
    # 유사도가 낮은 결과들은 보여주지 않음
    for result in results:
        uuid, distance = result
        if distance > 500:
            pass
        else:
            return uuid
            
        
print(search_query('클라우드 컴퓨팅', 3))