### ScreenshotDB


맨위에 추가
from summary import process_image

# 이미지 및 요약 표시 함수
def display_images_and_summary(image_path):
    try:
        img = Image.open(image_path)
        summary = process_image(img) <<<<<<<<<<<<<<<<<<<<<<<<< 이거바꾸기
        categories = ", ".join(example_categories)
        chatbot_message = Chatbot
        return img, summary, categories, chatbot_message
    except Exception:
        return None, "이미지를 찾을 수 없습니다.", "", ""