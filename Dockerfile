FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 STREAMLIT_SERVER_ADDRESS=0.0.0.0 STREAMLIT_SERVER_PORT=8501
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends wget && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app ./app

EXPOSE 8501
HEALTHCHECK CMD wget -qO- http://localhost:8501/_stcore/health || exit 1
CMD ["streamlit", "run", "app/app.py"]
