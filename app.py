import gradio as gr
import os
import shutil
from PIL import Image

# 예제 요약 및 카테고리 데이터
example_summary = "모카빵 반죽을 250그램씩 9개로 나누고, 중간 발효를 15분 진행합니다. 토핑은 100그램씩 분할하여, 반죽 위에 밀대로 펼쳐 덮습니다."
example_categories = ['음식', '레시피']
image_path = os.path.abspath("C:/mtoc/이미지셋/레시피1.jpg")

# 이미지 정리 함수
def organize_by_category(image_path, categories):
    base_dir = './organized_images'
    os.makedirs(base_dir, exist_ok=True)
    
    for category in categories:
        category_folder = os.path.join(base_dir, category)
        os.makedirs(category_folder, exist_ok=True)
        shutil.copy(image_path, os.path.join(category_folder, os.path.basename(image_path)))
    
    return f"Image organized into folders: {', '.join(categories)}"

# 이미지 및 요약 표시 함수
def display_images_and_summary(image_path):
    if not image_path or image_path == "검색된 이미지가 없습니다.":
        return None, "이미지를 찾을 수 없습니다."
    
    try:
        print(f"Loading image from path: {image_path}")
        img = Image.open(image_path)
        summary = example_summary
        return img, summary
    except Exception as e:
        print(f"Error: {e}")
        return None, "이미지를 찾을 수 없습니다."

# 카테고리별 이미지 검색 함수
def search_images_by_category(search_query):
    base_dir = './organized_images'
    matched_images = []
    seen_filenames = set()
    
    search_query = search_query.lower()
    for category in example_categories:
        if search_query in category.lower():
            for category_folder in os.listdir(base_dir):
                category_folder_path = os.path.join(base_dir, category_folder)
                if os.path.isdir(category_folder_path):
                    for img_file in os.listdir(category_folder_path):
                        if img_file.endswith(('.png', '.jpg', '.jpeg')):
                            if img_file not in seen_filenames:
                                img_path = os.path.join(category_folder_path, img_file)
                                matched_images.append(img_path)
                                seen_filenames.add(img_file)
    
    return matched_images if matched_images else ["검색된 이미지가 없습니다."]

# 선택된 이미지에 대한 요약 및 이미지 표시 함수
def update_image_and_summary(selected):
    if isinstance(selected, list) and len(selected) > 0:
        selected_image_path = selected[0][0]
        print(f"Selected image: {selected_image_path}")

        if selected_image_path and os.path.exists(selected_image_path):
            img, summary = display_images_and_summary(selected_image_path)
            return img, summary
    return None, "이미지를 찾을 수 없습니다."

# Gradio 인터페이스 구성
with gr.Blocks() as app:
    with gr.Column():
        gr.Markdown("### 이미지 검색/요약")
        
        search_input = gr.Textbox(label="카테고리 검색", placeholder="카테고리 이름 입력...")
        search_button = gr.Button("검색")
        search_results = gr.Gallery(label="검색 결과 이미지", elem_id="gallery", columns=4, height="500px")
        search_button.click(fn=search_images_by_category, inputs=search_input, outputs=search_results)
    
    with gr.Row():
        with gr.Column(scale=1):
            selected_image_display = gr.Image(label="이미지", width=480, height=552)
        with gr.Column(scale=2):
            selected_summary_display = gr.Textbox(label="요약", interactive=False, lines=20)
            tags_display = gr.Textbox(value=", ".join(example_categories), label="태그", interactive=False)

        search_results.select(fn=update_image_and_summary, inputs=search_results, outputs=[selected_image_display, selected_summary_display])

    with gr.Row():
        tags_display

app.launch(share = True)
