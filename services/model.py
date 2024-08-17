from langchain.chat_models import ChatOpenAI, ChatOllama
from langchain.schema import HumanMessage, SystemMessage
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
import torch
from dotenv import load_dotenv

load_dotenv()

llama_chat_model = ChatOllama(model="llama3.1:latest", temperature=0.1)
gpt_chat_model = ChatOpenAI(model='gpt-4o-mini')
tokenizer = AutoTokenizer.from_pretrained('MLP-KTLim/llama-3-Korean-Bllossom-8B')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
embedding_model = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS')

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