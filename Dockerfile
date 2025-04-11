# Sử dụng image Python 3.9 để match với cog.yaml
FROM python:3.9-slim

# Đặt thư mục làm việc trong container
WORKDIR /src

# Cài đặt system packages
RUN apt-get update && apt-get install -y git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Cài đặt Python packages từ cog.yaml
RUN pip install --no-cache-dir \
    flask \
    pyngrok \
    faiss-cpu \
    numpy \
    sentence-transformers \
    transformers \
    PyPDF2 \
    torch==2.3.1 \
    Pillow

# Copy toàn bộ mã nguồn của bạn vào container
COPY . .

# Mở cổng 5000 (nếu ứng dụng của bạn chạy trên cổng này)
EXPOSE 5000

# Thiết lập lệnh khởi chạy khi container chạy
CMD ["python", "app.py"]
