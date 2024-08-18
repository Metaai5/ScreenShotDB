from langchain.prompts import PromptTemplate
from sentence_transformers import util
from models.llm_models import LLMModel
from dependencies.model_factory import device, embedding_model
import torch
import json
from pathlib import Path
from collections import defaultdict
import pandas as pd
from config.path import TAG_FILE_PATH
from langchain.chat_models import ChatOpenAI
import logging


tag_file = Path(TAG_FILE_PATH)

def load_tags():
    if tag_file.exists():
        with open(tag_file, 'r') as file:
            return json.load(file)
    return defaultdict(dict)

tags = load_tags()
tags_str = ','.join(tags)
print(tags_str)
#  TODO:태그 프롬프트 변경
def make_models():
    topic_classification_prompt = '''
                                    당신은 세계에서 제일 유능한 토픽 분류 전문가입니다.
                                    답변에는 카테고리만 제시합니다.
                                '''
    topic_classification_user_prompt_template = PromptTemplate.from_template('{text}라는 내용에 대해 카테고리로 분류하세요. \
                                                                            기존의 카테고리는' + tags_str + ' 입니다\
                                                                            최대한 기존의 카테고리 내에 속하게 해주세요.\
                                                                            카테고리는 최대 세개, 각 카테고리는 유일합니다.\
                                                                            카테고리에 불필요한 기호, 문구 등은 덧붙이지 않습니다.\
                                                                            대분류부터 소분류 순으로 나열하고 구분은 ,로 하세요.'
                                                                            )
    
    topic_classification_model = LLMModel(ChatOpenAI(model='gpt-4o-mini'), None, device, topic_classification_prompt, topic_classification_user_prompt_template)

    return topic_classification_model

topic_classification_model = make_models()

def save_tags(tags):
    with open(TAG_FILE_PATH, 'w') as f:
        json.dump(tags, f, indent=4)
        
def tag_document(text):
    # 새로운 태그 저장
    new_tags = []
    
    # 유사한 기존 태그 저장
    exist_tags = []
    similarity_threshold = 0.85  # 임계값
    
    classification = topic_classification_model.exec(text)
    splitted_tags = classification.split(',')
        
    for cur_tag in splitted_tags:
        cur_tag = cur_tag.strip()  # 공백 제거
        cur_tag_embedding = embedding_model.encode(cur_tag, device=device, convert_to_tensor=True)
        cur_simuilarity = False
        for tag_name, tag_info in tags.items():
            tag_embedding = torch.tensor(tag_info['embedding']).to(device)
            similarity = util.pytorch_cos_sim(cur_tag_embedding, tag_embedding).item()
            if similarity > similarity_threshold: 
                exist_tags.append(tag_name)
                cur_simuilarity = True
                # 여러 개 비슷한 경우도 전부 추가, 대신 임계값을 높임
        if not cur_simuilarity:       
            new_tags.append(cur_tag)
            tags[cur_tag] = {
                'embedding': cur_tag_embedding
            }
    
    save_tags(tags)
            
    # print("Classification:", classification)
    result = new_tags + exist_tags
    return result

if __name__ =='__main__':
    text = '''
    sys.path.append(): 코드에서 직접 Python 경로를 추가합니다.
절대 경로 import: from utils import paddle_ocr를 사용하고, 프로젝트 루트 디렉토리에서 실행합니다.
상대 경로 import: 패키지 구조를 고려해 상대 경로로 모듈을 import합니다.
가상 환경 확인: 가상 환경이 활성화된 상태인지 확인합니다.
'''
    
