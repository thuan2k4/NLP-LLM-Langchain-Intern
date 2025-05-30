from langchain_community.llms import CTransformers
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

#Cau hinh
model_file = "models/vinallama-7b-chat-Q5_0.gguf"

def load_file(model_file):
    # Load model
    llm = CTransformers(
        model=model_file, 
        model_type="llama", 
        max_new_tokens=512,  # Số lượng token tối đa cho mỗi lần sinh
        tempurature=0.1,  # Độ ngẫu nhiên của đầu ra
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

#demo chain

template = """<|im_start|>system
Bạn là một trợ lý AI hữu ích. Hãy trả lời người dùng 1 cách chính xác <|im_end|>
<|im_start|>user
{question}<|im_end|>
<|im_start|>assistantnsdkfnsdfkjn
"""

prompt = create_prompt(template)
llm = load_file(model_file)
chain = create_simple_chain(prompt, llm)

question = "Thiền Tông là gì?"
response = chain.invoke({"question": question})
print(response)  # In ra câu trả lời từ mô hình