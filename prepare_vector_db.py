from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter 
# Recur là chia theo dấu câu, Character là chia theo (số lượng) ký tự
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
# Load file
from langchain_community.vectorstores import Faiss
# Lưu trữ vector 
from langchain_community.embeddings import GPT4AllMiniLMEmbedder
from langchain_community.embeddings import HuggingFaceEmbeddings

pdf_data_path = ""
vector_db_path = "vectorstore/db_faiss"

# Tao vector db từ 1 đoạn text
def create_vector_db_from_text(text):
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
    embedding = GPT4AllMiniLMEmbedder()