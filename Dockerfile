# 使用官方轻量级镜像
FROM python:3.12-slim

# 设置工作目录
WORKDIR /app

# 1. 安装系统级编译依赖（针对 hdbscan, scipy 等）
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 2. 利用 Docker 缓存机制：先单独复制依赖文件
COPY requirements.txt .

# 3. 安装 Python 依赖，禁用缓存以缩减镜像体积
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 4. 固化 NLP 资产：提前下载 Presidio 所需的 SpaCy 模型
#    (en_core_web_lg 约 700MB，若网络慢可改为 en_core_web_sm)
RUN python -m spacy download en_core_web_lg

# 5. 复制全部业务代码（此时代码已就绪）
COPY . .

# 6. 【关键】在构建阶段预加载所有 Detector，避免运行时首次调用卡顿
#    这一步会触发 pandas-type-detector / Presidio 的完整初始化，
#    并将模型加载到内存中，后续运行将直接使用缓存。
#    若预加载失败（例如网络超时），构建会立刻报错，便于提前定位问题。
RUN python -c "from app.schema.semantic_profiler import SemanticProfiler; \
               print('Preloading detectors...'); \
               SemanticProfiler.preload_all_detectors(); \
               print('Preload completed successfully.')"

# 7. 暴露服务端口（根据你的框架调整，如 FastAPI 默认 8000）
EXPOSE 8000

# 8. 启动指令（假设你的入口是 main.py）
CMD ["python", "main.py"]