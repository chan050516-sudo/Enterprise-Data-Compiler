# 使用官方轻量级镜像
FROM python:3.12-slim

# 设置工作目录
WORKDIR /app

# 1. 安装系统级编译依赖（针对 hdbscan, scipy 等）
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# 2. 利用 Docker 缓存机制：先单独复制依赖文件
COPY requirements.txt .

# 3. 安装 Python 依赖，禁用缓存以缩减镜像体积
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 4. 固化 NLP 资产：提前下载 Presidio 所需的 SpaCy 模型
RUN python -m spacy download en_core_web_lg

# 5. 复制全部业务代码 (会避开 .dockerignore 中的文件)
COPY . .

# 6. 暴露服务端口（根据你的框架调整，如 FastAPI 默认 8000）
EXPOSE 8000

# 7. 启动指令（假设你的入口是 main.py，或者按需替换为 uvicorn/pytest 等）
CMD ["python", "main.py"]