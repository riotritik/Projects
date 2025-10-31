FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY . /app
ENV PORT 7860
CMD gunicorn brain_tumor_detection.wsgi --bind 0.0.0.0:$PORT
