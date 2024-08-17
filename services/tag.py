from langchain.chat_models import ChatOpenAI, ChatOllama
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain.prompts import PromptTemplate
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer, util
import torch
import json
from pathlib import Path
from collections import defaultdict
import os
from dotenv import load_dotenv

load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')

llama_chat_model = ChatOllama(model="llama3.1:latest", temperature=0.1)
gpt_chat_model = ChatOpenAI(model='gpt-4o-mini')
tokenizer = AutoTokenizer.from_pretrained('MLP-KTLim/llama-3-Korean-Bllossom-8B')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
embedding_model = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS')

tag_file = Path("data/tags.json")

def load_tags():
    if tag_file.exists():
        with open(tag_file, 'r') as file:
            return json.load(file)
    return defaultdict(dict)

tags = load_tags()
tags_str = ','.join(tags)

class LLMModel():
    def __init__(self, model, tokenizer, device, prompt, user_prompt_template):
        self.prompt = prompt
        self.user_prompt_template = user_prompt_template
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def exec(self, text):
        llm_prompt_result = self.user_prompt_template.format(text=text)
        messages = [SystemMessage(content=self.prompt) ,
                    HumanMessage(content=llm_prompt_result)]

        return self.model(messages).content

class CustomModel(LLMModel):
    def __init__(self, model, tokenizer, device, prompt, user_prompt_template):
        super().__init__(model, tokenizer, device, prompt, user_prompt_template)

    def exec(self, text):
        llm_prompt_result = self.user_prompt_template.format(text=text)

        # 메시지를 직접 토큰화
        messages = f"{self.prompt}\n{llm_prompt_result}"
        input_ids = self.tokenizer(messages, return_tensors="pt").input_ids.to(self.device)

        outputs = self.model.generate(
            input_ids,
            max_new_tokens=256,
            eos_token_id=self.tokenizer.eos_token_id,  # 단일 정수로 설정
            do_sample=True,
            temperature=0.01,
            top_p=0.7, # 상위 n% 확률을 가진 토큰들만 샘플링에 포함.
            repetition_penalty=1.1 # 반복에 대한 페널티
        )

        return self.tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)

def make_models():
    summary_prompt = '''
                        당신은 도움이 되는 텍스트분석전문가입니다.
                        답변에는 요약한 내용만을 제시합니다.
                    '''
    summary_user_prompt_template = PromptTemplate.from_template('''
                                                                다음 텍스트를 요약하되, 주요 포인트에 중점을 두고 아래와 같은 형식으로 요약하세요:
                                                                요약은 간단하게 나타내고 주요 포인트에 집중해주세요
                                                                용량 과 단위도 나타내세요

                                                                1. 요약:
                                                                2. 주요 포인트:
                                                                    - 포인트 1
                                                                    - 포인트 2
                                                                    - 포인트 3

                                                                요약할 내용이 없다면, 요약을 생략해도 됩니다.
                                                                다음은 요약할 텍스트입니다:
                                                                {text}                                         
                                                                ''')
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
    summary_model = LLMModel(gpt_chat_model, tokenizer, device, summary_prompt, summary_user_prompt_template)
    topic_classification_model = LLMModel(gpt_chat_model, tokenizer, device, topic_classification_prompt, topic_classification_user_prompt_template)

    return summary_model,topic_classification_model

summary_model, topic_classification_model = make_models()

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
    with open('data/tags.json', 'w') as f:
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
    
def make_summary(text):
    summary = summary_model.exec(text)
    print(summary)
    return summary

if __name__ =='__main__':
    text = '''
    sys.path.append(): 코드에서 직접 Python 경로를 추가합니다.
절대 경로 import: from utils import paddle_ocr를 사용하고, 프로젝트 루트 디렉토리에서 실행합니다.
상대 경로 import: 패키지 구조를 고려해 상대 경로로 모듈을 import합니다.
가상 환경 확인: 가상 환경이 활성화된 상태인지 확인합니다.
'''
    tag_document(text)
    print(make_summary(text))