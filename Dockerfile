# Sử dụng image Python tối giản phù hợp
FROM python:3.9-slim

# Cài đặt các gói hệ thống cần thiết
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Đặt thư mục làm việc cho container
WORKDIR /app

# Sao chép file requirements.txt và cài đặt các thư viện Python
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Sao chép toàn bộ mã nguồn và thư mục pdf_docs vào container
COPY . .

# Expose cổng mặc định của ứng dụng (5000)
EXPOSE 5000

# Lệnh chạy ứng dụng
CMD ["python", "app.py"]
