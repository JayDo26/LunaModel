# Sử dụng image Python 3.12 với phiên bản cụ thể cho tính ổn định
FROM python:3.12.2-slim@sha256:abc123def456ghi789jkl012mno345pqrs678tuv901wxyz234567890abcdef

# Thiết lập timezone
RUN apt-get update && apt-get install -y --no-install-recommends tzdata && \
    ln -fs /usr/share/zoneinfo/Asia/Ho_Chi_Minh /etc/localtime && \
    dpkg-reconfigure -f noninteractive tzdata && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Thiết lập biến môi trường Python
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Đặt thư mục làm việc trong container
WORKDIR /src

# Copy requirements.txt từ thư mục hiện tại (không phải thư mục cha)
COPY requirements.txt .

# Cài đặt các dependencies từ requirements.txt (giảm kích thước image)
RUN pip install --no-cache-dir -r requirements.txt

# Tạo user không phải root
RUN adduser --disabled-password --gecos '' appuser

# Copy toàn bộ mã nguồn của bạn vào container
COPY . .

# Chuyển quyền sở hữu thư mục cho user không phải root
RUN chown -R appuser:appuser /src

# Chuyển sang user không phải root
USER appuser

# Mở cổng 5000 (nếu ứng dụng của bạn chạy trên cổng này)
EXPOSE 5000

# Thiết lập lệnh khởi chạy khi container chạy
CMD ["python", "app.py"]
