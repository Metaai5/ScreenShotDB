import torch
from langchain.chat_models import ChatOllama, ChatOpenAI
from sentence_transformers import SentenceTransformer
from models.ocr_models import OCRModel  # OCRModel을 정의한 라이브러리로 교체

# 모델 초기화
llama_chat_model = ChatOllama(model="llama3.1:latest", temperature=0.1)
gpt_chat_model = ChatOpenAI(model='gpt-4o-mini')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
embedding_model = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS')
ocr_model = OCRModel()
