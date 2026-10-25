# language: Dockerfile, target: Render Free Web Service, Python 3.11
FROM python:3.11-slim

# системные зависимости:
#   ffmpeg            — извлечение кадров и сборка .webm
#   tesseract-ocr     — движок OCR
#   tesseract-ocr-rus — русские traineddata (в коде lang='rus+eng')
#   libgl1            — OpenCV (даже headless иногда требует)
#   libgomp1          — OpenMP для numpy/opencv
#   fonts-dejavu-core — шрифт для PIL (в коде путь /usr/share/fonts/truetype/dejavu/...)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    tesseract-ocr \
    tesseract-ocr-rus \
    libgl1 \
    libgomp1 \
    fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY bot.py .

# Render передаёт порт через $PORT; health_server() в bot.py читает os.environ["PORT"].
# EXPOSE не обязателен, но фиксирует намерение.
EXPOSE 8080

CMD ["python", "bot.py"]