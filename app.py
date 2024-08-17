import gradio as gr
import os
from PIL import Image
# from services.search import search_query
from utils.utils import save_image
from utils.summary import process_image

# 예제 요약 및 카테고리 데이터
example_summary = "모카빵 반죽을 250그램씩 9개로 나누고, 중간 발효를 15분 진행합니다. 토핑은 100그램씩 분할하여, 반죽 위에 밀대로 펼쳐 덮습니다."
example_categories = ['음식', '레시피']
image_path = os.path.abspath("data/")
Chatbot = "OOO하는 방식도 추천드립니다."

# 전역 변수
previous_results = []

# 이미지 및 요약 표시 함수
def display_images_and_summary(image_path):
    try:
        if isinstance(image_path, tuple):
            # 튜플의 첫 번째 요소가 이미지 경로입니다
            img_path = image_path[0]
            img = Image.open(img_path)
        elif isinstance(image_path, str):
            img = Image.open(image_path)
        elif isinstance(image_path, Image.Image):
            img = image_path
        summary = process_image(img)
        categories = ", ".join(example_categories)
        chatbot_message = Chatbot
        return img, summary, categories, chatbot_message
    except Exception:
        return None, "이미지를 찾을 수 없습니다.", "", ""

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
                        if img_file.endswith(('.png', '.jpg', '.jpeg', '.gif')):
                            if img_file not in seen_filenames:
                                img_path = os.path.join(category_folder_path, img_file)
                                matched_images.append(img_path)
                                seen_filenames.add(img_file)
    
    return matched_images if matched_images else []

# 선택된 이미지에 대한 요약 및 이미지 표시 함수
# def update_image_and_summary(selected):
#     if isinstance(selected, list) and len(selected) > 0:
#         selected_image_path = selected[0][0]  # 튜플에서 경로만 추출
            
#         if selected_image_path and os.path.exists(selected_image_path):
#             img, summary, categories, chatbot_message = display_images_and_summary(selected_image_path)
#             return img, summary, categories, chatbot_message
    
#     return None, "이미지를 찾을 수 없습니다.", "", ""

# 선택된 이미지에 대한 요약 및 이미지 표시 함수
def update_image_and_summary(evt: gr.SelectData, images): # 그라디오 선택 이벤트 객체 어떤이미지선택했는지
    if evt.index < len(images): # 유효한지 확인
        selected_image = images[evt.index]
        return display_images_and_summary(selected_image)
    return None, "이미지를 찾을 수 없습니다.", "", ""

# 이미지 업로드 처리 함수
def handle_image_upload(files, current_images):
    new_images = [file.name for file in files]
    all_images = current_images + new_images if current_images is not None else new_images
    return all_images

# Gradio 인터페이스 구성
with gr.Blocks() as app:
    with gr.Column():
        gr.Markdown("### 이미지 검색/요약")      
        with gr.Row():
            search_input = gr.Textbox(label="카테고리 검색", placeholder="카테고리 이름 입력...", scale=7)
            search_button = gr.Button("검색")
            image_input = gr.File(label="이미지 업로드", file_count="multiple", scale=1)
            image_input.change(save_image, inputs=image_input)
        search_input = gr.Textbox(label="태그 검색", placeholder="태그 입력")
        search_button = gr.Button("검색")
        
        # 초기에는 빈 문자열로 설정
        gallery_info = gr.Markdown(value="")

        def handle_search(search_query):
            global previous_results  # 전역 변수 사용 선언
            results = search_images_by_category(search_query)
            if results == previous_results:
                return gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
            
            previous_results = results
            if results:
                return results, "검색된 이미지 표시", None, "", "", ""
            else:
                return [], "검색 결과가 없습니다. 다른 태그로 검색해보세요.", None, "이미지를 찾을 수 없습니다.", "검색된 결과가 없습니다", ""
        
        
        search_results = gr.Gallery(label="검색 결과 이미지", elem_id="gallery", columns=5, height="480px",allow_preview=False)
             
    
    with gr.Row():
        with gr.Column(scale=1):
            selected_image_display = gr.Image(label="이미지", width=480, height=650)
        with gr.Column(scale=2):
            tags_display = gr.Textbox(label="태그", interactive=False)
            selected_summary_display = gr.Textbox(label="요약", interactive=False, lines=10)
            chatbot_display = gr.Textbox(label="Chatbot", interactive=False, lines=10)

        search_results.select(fn=update_image_and_summary, inputs=search_results, outputs=[selected_image_display, selected_summary_display, tags_display, chatbot_display])

    with gr.Row():
        tags_display

    # search_button.click(fn=handle_search, inputs=search_input, outputs=[search_results, gallery_info, selected_image_display, selected_summary_display, tags_display, chatbot_display])

    
    search_results.select(fn=update_image_and_summary, inputs=[search_results], outputs=[selected_image_display, selected_summary_display, tags_display, chatbot_display])
    
    image_input.change(fn=handle_image_upload, inputs=[image_input, search_results], outputs=search_results)

    search_button.click(fn=handle_search, inputs=search_input, outputs=[search_results, gallery_info, selected_image_display, selected_summary_display, tags_display])

app.launch(share=True)