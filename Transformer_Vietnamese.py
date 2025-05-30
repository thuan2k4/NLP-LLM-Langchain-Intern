from transformers import AutoTokenizer, AutoModelForCausalLM
import re

# Load mô hình và tokenizer
tokenizer = AutoTokenizer.from_pretrained("NlpHUST/gpt2-vietnamese")
model = AutoModelForCausalLM.from_pretrained("NlpHUST/gpt2-vietnamese")

# Prompt
prompt = "Thiền tông là gì"

# Tokenize
inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)

# Sinh văn bản
outputs = model.generate(
    **inputs,
    max_length=200,
    num_beams=5,
    do_sample=True,
    top_p=0.95,
    no_repeat_ngram_size=3,
    temperature=0.3,
    pad_token_id=tokenizer.eos_token_id
)

# Giải mã và hậu xử lý
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(f"Văn bản sinh ra: {generated_text}")