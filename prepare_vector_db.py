from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter 
# Recur là chia theo dấu câu, Character là chia theo (số lượng) ký tự
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
# Load file
from langchain_community.vectorstores import FAISS
# Lưu trữ vector 
from langchain_community.embeddings import GPT4AllEmbeddings
# Sử dụng model local
pdf_data_path = "data"
vector_db_path = "vectorstore/db_faiss"

# Tao vector db từ 1 đoạn text
def create_vector_db_from_text():
    raw_text = """Thiền Tông (hay Zen trong tiếng Nhật) là một trường phái của 
    Phật giáo Đại thừa, nhấn mạnh vào việc thực hành thiền định để đạt được giác ngộ. 
    Thiền Tông tập trung vào trải nghiệm trực tiếp và sự hiểu biết bản thân thông qua 
    thiền định, thay vì dựa vào kinh điển hay nghi lễ phức tạp."""
    
    # Chia nhỏ văn bản
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=256,  # Kích thước mỗi đoạn văn bản
        chunk_overlap=50,  # Số lượng ký tự chồng lấn giữa các đoạn
        length_function=len  # Hàm tính độ dài của đoạn văn bản
    )
    
    chunks = text_splitter.split_text(raw_text)
    
    #Embedding
    embedding = GPT4AllEmbeddings(model_file="models/all-MiniLM-L6-v2.gguf")
    
    # Đưa vào Faiss Vector DB
    db = FAISS.from_texts(texts=chunks, embedding=embedding) 
    # bien cac doan chunk thành vector
    db.save_local(vector_db_path)
    return db

def create_vector_db_from_pdf():
    # Quet toan bo file pdf
    loader = DirectoryLoader(pdf_data_path, glob="*.pdf", loader_cls=PyPDFLoader)
    docs = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,  # Kích thước mỗi đoạn văn bản
        chunk_overlap=100,  # Số lượng ký tự chồng lấn giữa các đoạn
    )
    
    chunks = text_splitter.split_documents(docs)

    #Embedding
    embedding = GPT4AllEmbeddings(model_file="models/all-MiniLM-L6-v2.gguf")
    
    # Đưa vào Faiss Vector DB
    db = FAISS.from_documents(chunks, embedding) 
    # bien cac doan chunk thành vector
    db.save_local(vector_db_path)
    return db

create_vector_db_from_text()
create_vector_db_from_pdf()