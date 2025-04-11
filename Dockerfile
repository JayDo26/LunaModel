# Sử dụng image Python 3.9 để match với cog.yaml
FROM python:3.9-slim

# Đặt thư mục làm việc trong container
WORKDIR /src

COPY requirements.txt


RUN apt-get update && apt-get install -y libpq-dev build-essential

RUN pip install -r requirements.txt
 
COPY . .
EXPOSE 5000

# Thiết lập lệnh khởi chạy khi container chạy
CMD ["python", "app.py"]
