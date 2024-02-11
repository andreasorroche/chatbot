FROM python:3.9-slim
WORKDIR /app

COPY docker/requirements.txt ./
RUN pip install --upgrade pip
RUN pip install -r requirements.txt --no-cache-dir

COPY config/ ./config
COPY src/ ./src
COPY data/ ./data
COPY app.py ./

EXPOSE 8501

CMD ["streamlit", "run", "app.py"]
