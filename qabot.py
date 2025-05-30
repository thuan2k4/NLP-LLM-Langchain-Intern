from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.vectorstores import FAISS


#Cau hinh
model_file = "models/vinallama-7b-chat-Q5_0.gguf"
vector_db_path = "vectorstore/db_faiss"

def load_llm(model_file):
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
        input_variables=["context", "question"], #Thêm yếu tố ngữ cảnh cảu văn bản
        template=template
    )
    return prompt

def create_qa_chain(prompt, llm, db):
    # Tạo chuỗi đơn giản
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # Sử dụng chuỗi "stuff" để kết hợp ngữ cảnh và câu hỏi
        retriever=db.as_retriever(search_kwargs={"k": 3}),  # Sử dụng bộ truy xuất từ vector DB
        # đưa ra k văn bản gần nhất với query
        return_source_documents=False,  # Có/Không trả về tài liệu nguồn
        chain_type_kwargs={
            "prompt": prompt  # Sử dụng prompt đã tạo
        }
    )
    return chain


# read from vectordb
def read_vector_db():
    # Embedding
    embedding = GPT4AllEmbeddings(model_file="models/all-MiniLM-L6-v2.gguf")
    db = FAISS.load_local(vector_db_path, embedding, allow_dangerous_deserialization=True)
    return db

#demo
db = read_vector_db()
llm = load_llm(model_file)

#create prompt
template = """<|im_start|>system
Sử dụng thông tin sau đây để trả lời câu hỏi. Nếu bạn không biết câu trả lời, hãy nói không biết, đừng cố tạo ra câu trả lời\n
{context}<|im_end|>\n
<|im_start|>user\n
{question}<|im_end|>\n
<|im_start|>assistant
"""

prompt = create_prompt(template)
# create chain
chain = create_qa_chain(prompt, llm, db)

#run chain
question = "Thiền Tông là gì?"
response = chain.invoke({"query": question})
print(response)  # In ra câu trả lời từ mô hình