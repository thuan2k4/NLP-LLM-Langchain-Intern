#pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124
from langchain_community.llms import LlamaCpp
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# Cấu hình
model_file = "models/vinallama-7b-chat-Q5_0.gguf"

def load_file(model_file):
    # Load model với hỗ trợ GPU tối ưu cho GeForce 940M
    llm = LlamaCpp(
        model_path=model_file,
        model_type="llama",
        n_ctx=1024,           # Giảm context length để tiết kiệm bộ nhớ
        max_tokens=256,       # Giảm số token tối đa
        temperature=0.1,      # Độ ngẫu nhiên của đầu ra
        n_gpu_layers=5,       # Offload ít layer lên GPU do VRAM thấp (thử 0-10)
        n_batch=128,          # Giảm batch size để tránh lỗi bộ nhớ
        verbose=True          # In thông tin debug
    )
    return llm

def create_prompt(template):
    # Tạo prompt
    prompt = PromptTemplate(
        input_variables=["question"],
        template=template
    )
    return prompt

def create_simple_chain(prompt, llm):
    # Tạo chuỗi đơn giản
    chain = LLMChain(
        llm=llm,
        prompt=prompt
    )
    return chain

# Demo chain
template = """<|im_start|>system
Bạn là một trợ lý AI hữu ích. Hãy trả lời người dùng một cách chính xác <|im_end|>
<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant
"""

prompt = create_prompt(template)
llm = load_file(model_file)
chain = create_simple_chain(prompt, llm)

# Chạy câu hỏi
question = "Một cộng một bằng bao nhiêu?"
response = chain.invoke({"question": question})
print(response["text"])  # In ra câu trả lời từ mô hình