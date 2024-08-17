from langchain.prompts import PromptTemplate
from sentence_transformers import util
from models.llm_models import LLMModel
from dependencies.model_factory import gpt_chat_model, device, embedding_model
import torch
import json
from pathlib import Path
from collections import defaultdict
import pandas as pd
from config.path import TAG_FILE_PATH

tag_file = Path(TAG_FILE_PATH)

def load_tags():
    if tag_file.exists():
        with open(tag_file, 'r') as file:
            return json.load(file)
    return defaultdict(dict)

tags = load_tags()
tags_str = ','.join(tags)



def make_models():
    topic_classification_prompt = '''
                                    당신은 세계에서 제일 유능한 토픽 분류 전문가입니다.
                                    답변에는 카테고리만 제시합니다.
                                '''
    topic_classification_user_prompt_template = PromptTemplate.from_template('{text}라는 내용에 대해 카테고리로 분류하세요. \
                                                                            기존의 카테고리는' + tags_str + ' 입니다\
                                                                            최대한 기존의 카테고리 내에 속하게 해주세요.\
                                                                            카테고리는 최대 다섯개, 각 카테고리는 유일합니다.\
                                                                            카테고리에 불필요한 기호, 문구 등은 덧붙이지 않습니다.\
                                                                            대분류부터 소분류 순으로 나열하고 구분은 ,로 하세요.'
                                                                            )
    
    topic_classification_model = LLMModel(gpt_chat_model, None, device, topic_classification_prompt, topic_classification_user_prompt_template)

    return topic_classification_model

topic_classification_model = make_models()

def generate_new_tag(text):
    new_tags = []
    classification = topic_classification_model.exec(text)
    splitted_tags = classification.split(',')

    for cur_tag in splitted_tags:
        cur_tag = cur_tag.strip()  # 공백 제거
        if cur_tag not in tags:  # 이미 존재하지 않는 태그라면 추가
            tags[cur_tag] = {'embedding':None}
            new_tags.append(cur_tag)
            
    # print("Classification:", classification)
    
    return new_tags

def save_tags(tags):
    with open(TAG_FILE_PATH, 'w') as f:
        json.dump(tags, f, indent=4)
            
def tag_document(text):
    # 문서 임베딩 생성
    document_embedding = embedding_model.encode(text, convert_to_tensor=True)
    
    # 기존 태그 중 가장 유사한 태그를 찾기
    best_tag = None
    best_similarity = -1  # 초기 유사도 값
    
    for tag_name, tag_info in tags.items():
        tag_embedding = torch.tensor(tag_info['embedding']).to(device)
        similarity = util.pytorch_cos_sim(document_embedding, tag_embedding)
        
        if similarity > best_similarity:
            best_similarity = similarity
            best_tag = tag_name
    
    # 임계값을 설정하여 문서가 기존 태그에 들어갈 수 있는지 판단 (유사도 0.7 이상)
    if best_similarity > 0.7:
        print(f"문서를 '{best_tag}' 태그에 추가합니다. (유사도: {best_similarity})")
        return best_tag
    else:
        # 적절한 태그가 없으면 새로운 태그 생성
        new_tag_names = generate_new_tag(text)
        for new_tag_name in new_tag_names:
            tag_embedding = embedding_model.encode(new_tag_name, convert_to_tensor=True)
            
            tags[new_tag_name] = {
                'embedding':tag_embedding.cpu().numpy().tolist()
            }
            save_tags(tags)  # 새로운 태그 저장
            print(f"새로운 태그 '{new_tag_name}' 생성 및 저장.")
        return new_tag_names
    


if __name__ =='__main__':
    text = '''
    sys.path.append(): 코드에서 직접 Python 경로를 추가합니다.
절대 경로 import: from utils import paddle_ocr를 사용하고, 프로젝트 루트 디렉토리에서 실행합니다.
상대 경로 import: 패키지 구조를 고려해 상대 경로로 모듈을 import합니다.
가상 환경 확인: 가상 환경이 활성화된 상태인지 확인합니다.
'''
    tag_document(text)
