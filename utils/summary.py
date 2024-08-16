import os
from paddleocr import PaddleOCR
import re
import string
import json
from langchain.chat_models import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain

# PaddleOCR 초기화 (한국어 모델 사용)
ocr = PaddleOCR(lang='korean')

# 텍스트 요약을 위한 프롬프트 템플릿 정의
prompt_template = """
당신은 도움이 되는 텍스트분석전문가입니다.

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

{input_text}
"""

# 프롬프트와 LLM 모델을 사용하여 요약 체인 생성
prompt = ChatPromptTemplate.from_template(prompt_template)
model = ChatOllama(model='llama3.1:latest')
llm_chain = LLMChain(llm=model, prompt=prompt)

def clean_text(text):
    # 특수 문자 제거
    pattern = f"[{re.escape(string.punctuation)}]"
    cleaned_text = re.sub(pattern, '', text)
    # 연속된 공백을 하나의 공백으로 대체하고 양쪽 공백 제거
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    return cleaned_text

# 이미지 요약 반환
def process_image(image):
    # 이미지를 임시로 저장
    temp_path = "temp_image.jpg"
    image.save(temp_path)
    
    # OCR 수행
    result = ocr.ocr(temp_path, cls=False)
    ocr_result = result[0]
    
    # 텍스트와 좌표 추출
    y_threshold = 20
    sorted_results = sorted([(item[1][0], item[0][0][0], item[0][0][1]) for item in ocr_result], key=lambda x: x[2])
    
    # y 좌표를 기준으로 그룹화
    groups = []
    current_group = [sorted_results[0]]
    for item in sorted_results[1:]:
        if abs(item[2] - current_group[-1][2]) <= y_threshold:
            current_group.append(item)
        else:
            groups.append(current_group)
            current_group = [item]
    groups.append(current_group)
    
    # 그룹 내에서 정렬하고 텍스트 정제
    result = []
    for group in groups:
        sorted_group = sorted(group, key=lambda x: x[1])
        line_data = [{'text': clean_text(item[0]), 'coordinates': item[1]} for item in sorted_group]
        result.append(line_data)
    
    # 모든 텍스트 결합
    full_text = ' '.join([' '.join([item['text'] for item in line]) for line in result])
    
    # 텍스트 요약
    summary = llm_chain.run(input_text=full_text)
    
    # 임시 파일 삭제
    os.remove(temp_path)
    
    return summary