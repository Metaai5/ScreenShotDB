import uuid
from PIL import Image
from pathlib import Path
import os
from services.tag import tag_document 
from services.summary import make_summary
from utils.paddle_ocr import make_wordlist
import pandas as pd

data_path = './data'  # 데이터 디렉토리
organized_file_path = f'{data_path}/organized_images'  # 태그 및 요약이 완료된 이미지가 위치할 디렉토리
storage_data_file_path = f'{data_path}/result_texts.csv'  # DB대용의 파일

# 필요한 모든 디렉토리 생성
os.makedirs(data_path, exist_ok=True)
os.makedirs(organized_file_path, exist_ok=True)


# 이미지 저장
def save_image(image_path, save_path):
    print(image_path)
    image = Image.open(image_path)
    image.save(save_path)
    return save_path

#text, file_path, uuid, tag, df 생성 
def make_dataframe(document_data):
    result_texts = {}
    result_texts[document_data['uuid_str']] = {
        'text': document_data['text'],
        'file_path': document_data['file_path'],
        'tags': document_data['tags'],
        'summary': document_data['summary']
    }
    
    df = pd.DataFrame(result_texts).T.reset_index()
    df.columns = ['uuid', 'text', 'file_path', 'tags', 'summary']
    df = df[['uuid', 'file_path', 'text', 'tags', 'summary']]
    if not os.path.exists(storage_data_file_path):
        df.to_csv(storage_data_file_path, index=False)
    else:
        base_df = pd.read_csv(storage_data_file_path)
        base_df = pd.concat([base_df, df], ignore_index=True)
        base_df.to_csv(storage_data_file_path, index=False)
            
# Gradio - pipe 역할
def load_image(image_paths):
    for image_path in image_paths:
        allowed_formats = {'.jpg', '.jpeg', '.png'}
        image_format = Path(image_path).suffix.lower()
        uuid_str = str(uuid.uuid4()) 
        save_path = os.path.join(organized_file_path, f'{uuid_str}{image_format}')
        if image_format in allowed_formats:
            try:
                file_path = save_image(image_path, save_path)
                text = ' '.join(make_wordlist(file_path))
                tags = tag_document(text)
                summary = make_summary(text)
                document_data = {'uuid_str':uuid_str, 'text':text, 'file_path': image_path, 'tags':tags, 'summary' : summary}
                make_dataframe(document_data)
            except Exception as e:
                return f'이미지 파일 .jpg, .jpeg, .png만 업로드 가능합니다.'
        
