import json
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
from pathlib import Path

# 사전 훈련된 모델 로드
model = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS')

# Hugging Face의 요약 모델 로드
summarizer = pipeline("summarization", model="gogamza/kobart-summarization")

# 태그 파일 경로 설정
tag_file = Path("data/tag.json")

# 태그 로드 함수
def load_tags():
    if tag_file.exists():
        with open(tag_file, 'r') as file:
            return json.load(file)
    return {}

# 태그 저장 함수
def save_tags(tags):
    with open(tag_file, 'w') as file:
        json.dump(tags, file, indent=4)


def tag_document(document):
    # 태그 로드
    tags = load_tags()
    
    # 문서 임베딩 생성
    document_embedding = model.encode(document, convert_to_tensor=True)
    
    # 기존 태그 중 가장 유사한 태그를 찾기
    best_tag = None
    best_similarity = -1  # 초기 유사도 값
    
    for tag_name, tag_info in tags.items():
        tag_embedding = tag_info['embedding']
        similarity = util.pytorch_cos_sim(document_embedding, tag_embedding)
        
        if similarity > best_similarity:
            best_similarity = similarity
            best_tag = tag_name
    
    # 임계값을 설정하여 문서가 기존 태그에 들어갈 수 있는지 판단 (예: 유사도 0.7 이상)
    if best_similarity > 0.7:
        print(f"문서를 '{best_tag}' 태그에 추가합니다. (유사도: {best_similarity})")
        return best_tag
    else:
        # 적절한 태그가 없으면 새로운 태그 생성
        new_tag_name = generate_new_tag_name(document)
        tags[new_tag_name] = {
            'embedding': document_embedding.cpu().numpy().tolist()
        }
        save_tags(tags)  # 새로운 태그 저장
        print(f"새로운 태그 '{new_tag_name}' 생성 및 저장.")
        return new_tag_name

def generate_new_tag_name(document):
    # LLM을 사용해 문서 내용을 요약하여 태그 이름으로 사용
    summary = summarizer(document, max_length=10, min_length=2, do_sample=False)[0]['summary_text']
    return summary.strip()  # 요약된 텍스트를 태그명으로 반환

