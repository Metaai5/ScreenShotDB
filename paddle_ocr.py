from paddleocr import PaddleOCR
from PIL import Image
import re
import string
import json

ocr = PaddleOCR(lang='korean')
 
def clean_text(text):
    pattern = f"[{re.escape(string.punctuation)}]"
    cleaned_text = re.sub(pattern, '', text)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    return cleaned_text

def make_wordlist(img_path):
    result = ocr.ocr(img_path, cls=False)
    word_list = []
    for i in result[0]:
        word_list.append(clean_text(i[1][0]))
    return word_list

def make_json(img_path):
    ocr_result = ocr.ocr(img_path, cls=False)[0]
    # 텍스트와 좌표를 추출하고 y 좌표로 정렬
    y_threshold = 10
    sorted_results = sorted([(item[1][0], item[0][0][0], item[0][0][1]) for item in ocr_result], key=lambda x: x[2])

    # y 좌표를 기준으로 그룹화 : 같은 줄
    groups = []
    current_group = [sorted_results[0]]

    for item in sorted_results[1:]:
        if abs(item[2] - current_group[-1][2]) <= y_threshold:
            current_group.append(item)
        else:
            groups.append(current_group)
            current_group = [item]

    groups.append(current_group)
    # 각 그룹 내에서 x 좌표로 정렬하고 텍스트 연결 (기호 제거 적용)
    result = []
    for group in groups:
        sorted_group = sorted(group, key=lambda x: x[1])
        line_data = [{'text': clean_text(item[0]), 'coordinates': item[1]} for item in sorted_group]
        result.append(line_data)

    # 결과를 JSON 파일로 저장
    output_data = {
        "lines": result
    }
    
    # json파일 저장
    with open('ocr_result_with_coordinates.json', 'w', encoding='utf-8') as f: 
        json.dump(output_data, f, ensure_ascii=False, indent=4)

# def read_json(file_path):
# # JSON 파일 불러오기 및 텍스트 추출 
#     texts = []
#     for line in output_data['lines']:
#         line_text = ' '.join([item['text'] for item in line])
#         texts.append(line_text)

        