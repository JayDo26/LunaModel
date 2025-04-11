# Sử dụng image Python 3.12 (phiên bản slim có kích thước nhỏ hơn)
FROM python:3.12-slim

# Đặt thư mục làm việc trong container
WORKDIR /src

# Copy file requirements.txt vào container
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy toàn bộ mã nguồn của bạn vào container
COPY . .

# Mở cổng 5000 (nếu ứng dụng của bạn chạy trên cổng này)
EXPOSE 5000

# Thiết lập lệnh khởi chạy khi container chạy
CMD ["python", "app.py"]
