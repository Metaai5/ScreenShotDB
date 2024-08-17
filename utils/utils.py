import uuid
from PIL import Image
from pathlib import Path
import os
from services.tag import tag_document 
from services.summary import make_summary
from utils.paddle_ocr import make_wordlist
import pandas as pd

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
    dataframe_path = 'data/result_texts.csv'
    df = pd.DataFrame(result_texts).T.reset_index()
    df.columns = ['uuid', 'text', 'file_path', 'tags', 'summary']
    df = df[['uuid', 'file_path', 'text', 'tags', 'summary']]
    if not os.path.exists(dataframe_path):
        df.to_csv(dataframe_path, index=False)
    else:
        base_df = pd.read_csv(dataframe_path)
        base_df = pd.concat([base_df, df], ignore_index=True)
        base_df.to_csv(dataframe_path, index=False)
            
# Gradio - pipe 역할
def load_image(image_paths):
    for image_path in image_paths:
        allowed_formats = {'.jpg', '.jpeg', '.png'}
        image_format = Path(image_path).suffix.lower()
        uuid_str = str(uuid.uuid4()) 
        save_path = os.path.join('data/organized_images/', f'{uuid_str}{image_format}')
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
        