# Research LLM

## LLM
- LLM là 1 thuật toán sử dụng AI mà trong đó nó sử dụng các công nghệ về Deep Learning và được train với 1 lượng lớn
dataset để hiểu, tóm tắt, sinh văn bản và dự đoán văn bản.
- LLM làm được:
    + Dịch
    + Tóm tắt
    + Q & A
    + Sentiment Analyst
- Kiến trúc Transformer
    + Position Encoding: đánh dấu từ
    + Attention: 
    + Self-Attention

- LLM Application & Finetuning
    + Finetuning là lấy 1 model pre-trained sau đó tối ưu tham số của nó
    -> Ví dụ: Dataset -> (pre-training) -> Base LLM ->  (Finetuning) -> Finetuned LLM <-> (Query, Response) User

- Các bước Finetuning:
    + Self-supervised
    + Supervised
    + Reinforcement Learning

- 03 options for model Parameter training
    1/ Retrain all parameters
    2/ Transfer Learning
    3/ Parameter Efficient Fine-Tuning (PEFT)

## Langchain & RAG
- Langchain là 1 framework xử lý các bài toán liên quan đến LLM
- Form basic langchain
    + LLM -> Prompt -> Data sources -> LLM -> Answer
- RAG (Retrieval Augmented Generation)
- Vector DB: chia nhỏ văn bản thành thành các đoạn nhỏ, mỗi đoạn sẽ đi qua 1 model embedding, để sinh ra 1 vector đặc trưng, sau đó vector đó sẽ được đưa vào DB