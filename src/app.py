import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import PyPDF2

from flask import Flask, request, jsonify

###############################################
# 1. Hàm load toàn bộ nội dung PDF từ thư mục pdf_docs
def load_pdf_texts(folder_path):
    texts = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".pdf"):
            file_path = os.path.join(folder_path, filename)
            try:
                with open(file_path, "rb") as f:
                    reader = PyPDF2.PdfReader(f)
                    text = ""
                    for page in reader.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + " "
                    texts.append(text)
            except Exception as e:
                print(f"Lỗi khi đọc {filename}: {e}")
    return "\n".join(texts)

# Sử dụng biến môi trường để lấy đường dẫn chứa PDF, mặc định là thư mục pdf_docs
pdf_folder = os.environ.get("PDF_FOLDER", "./Document")
loaded_text = load_pdf_texts(pdf_folder)
if loaded_text.strip():
    sample_text = loaded_text
else:
    sample_text = "No PDF content found."

###############################################
# 2. Hàm chia nhỏ văn bản thành các đoạn (chunks)
def split_text(text, chunk_size=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunks.append(" ".join(words[i:i+chunk_size]))
    return chunks

chunks = split_text(sample_text, chunk_size=50)

###############################################
# 3. Khởi tạo model, tokenizer, generator và FAISS index
def initialize_models():
    global embed_model, tokenizer, generator, faiss_index, chunk_embeddings

    print("Khởi tạo SentenceTransformer...")
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    chunk_embeddings = embed_model.encode(chunks, convert_to_numpy=True)
    dimension = chunk_embeddings.shape[1]
    faiss_index = faiss.IndexFlatL2(dimension)
    faiss_index.add(np.array(chunk_embeddings))

    print("Đang load mô hình Qwen2.5-7B...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True)
    generator = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True)
    if torch.cuda.is_available():
        generator.half().to("cuda")
    print("Khởi tạo xong.")

###############################################
# 4. Hàm kiểm duyệt phản hồi: đo cosine similarity giữa phản hồi và context
def filter_response(response, context, threshold=0.7):
    resp_emb = embed_model.encode([response])[0]
    ctx_emb = embed_model.encode([context])[0]
    norm_resp = resp_emb / np.linalg.norm(resp_emb)
    norm_ctx = ctx_emb / np.linalg.norm(ctx_emb)
    similarity = np.dot(norm_resp, norm_ctx)
    if similarity < threshold:
        return "Thông tin không rõ ràng. Bạn có thể cung cấp thêm chi tiết?"
    return response

###############################################
# 5. Định nghĩa pipeline CRAG với bước corrective retrieval bổ sung
class CRAGPipeline:
    def __init__(self, max_hops=2, top_k=3):
        self.max_hops = max_hops
        self.top_k = top_k

    def retrieve_chunks(self, query):
        query_embedding = embed_model.encode([query], convert_to_numpy=True)
        distances, indices = faiss_index.search(np.array(query_embedding), self.top_k)
        retrieved_text = "\n".join([chunks[i] for i in indices[0]])
        return retrieved_text

    def correct_retrieve(self, query):
        query_embedding = embed_model.encode([query], convert_to_numpy=True)
        new_top_k = self.top_k + 2
        distances, indices = faiss_index.search(np.array(query_embedding), new_top_k)
        retrieved_text = "\n".join([chunks[i] for i in indices[0]])
        return retrieved_text

    def generate(self, prompt, max_new_tokens=1000):
        messages = [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(generator.device)
        generated_ids = generator.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens
        )
        generated_ids = [
            output_ids[len(model_inputs.input_ids[i]):]
            for i, output_ids in enumerate(generated_ids)
        ]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response.strip()

    def hop_pipeline(self, initial_query):
        query = initial_query
        full_reasoning = f"Truy vấn ban đầu: {initial_query}\n"
        for hop in range(self.max_hops):
            context = self.retrieve_chunks(query)
            prompt = (
                f"Truy vấn: {query}\n\n"
                f"Thông tin liên quan:\n{context}\n\n"
                "Hãy xác định danh mục bệnh phù hợp ('Khoa nội','Khoa nội tiêu hoá','Khoa sản') "
                "Phân tích dựa trên truy vấn của người dùng. "
                "Chỉ sử dụng thông tin từ dữ liệu đã cho. "
                "Nếu thông tin chưa đủ, hãy yêu cầu làm rõ. "
                "Nếu triệu chứng chưa rõ ràng thì hãy phân khoa dựa trên các triệu chứng đang có hoặc đưa vào Nội Tổng Hợp"
            )
            response = self.generate(prompt)
            filtered_response = filter_response(response, context)
            full_reasoning += f"\n{filtered_response}\n"

            if "Thông tin không rõ ràng" in filtered_response:
                corrective_context = self.correct_retrieve(query)
                new_prompt = prompt + "\nThông tin bổ sung:\n" + corrective_context
                response = self.generate(new_prompt)
                filtered_response = filter_response(response, context + "\n" + corrective_context)
                full_reasoning += f"\nPhản hồi sau bổ sung: {filtered_response}\n"
                if "Thông tin không rõ ràng" not in filtered_response:
                    break
                else:
                    query = filtered_response  # Cập nhật truy vấn cho vòng lặp tiếp theo
                    full_reasoning += f"\nVẫn chưa đủ, chuyển truy vấn thành: {query}\n"
            else:
                break
        return full_reasoning

###############################################
# 6. Khởi tạo các model (chỉ chạy 1 lần)
initialize_models()
pipeline = CRAGPipeline(max_hops=3, top_k=4)

###############################################
# 7. Xây dựng API với Flask
app = Flask(__name__)

@app.route("/query", methods=["POST"])
def query():
    data = request.get_json()
    if not data or "query" not in data:
        return jsonify({"error": "Request phải có key 'query'"}), 400
    user_query = data["query"]
    result = pipeline.hop_pipeline(user_query)
    return jsonify({"result": result})

if __name__ == "__main__":
    # Sử dụng biến môi trường PORT do Replicate cung cấp
    port = int(os.environ.get("PORT", 5000))
    print("Chạy ứng dụng tại cổng:", port)
    app.run(host="0.0.0.0", port=port)
