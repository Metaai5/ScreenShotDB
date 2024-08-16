import uuid
from PIL import Image
from pathlib import Path
from utils.paddle_ocr import clean_text, make_wordlist, make_json
import os

def save_image(image_paths):
    for image_path in image_paths:
        allowed_formats = ['.jpg', '.jpeg', '.png']
        image_format = Path(image_path).suffix.lower()
        if image_format in allowed_formats:
            try:
                file_path = os.path.join('data/organized_images/', f'{str(uuid.uuid4())}{image_format}')
                image = Image.open(image_path)
                image.save(file_path)
                make_json(file_path)
            except Exception as e:
                return f'허용되지 않는 확장자입니다. 이미지 파일 .jpg, .jpeg, .png만 업로드 가능합니다.'
    
    