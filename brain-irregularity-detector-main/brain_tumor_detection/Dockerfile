FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY . /app
# Port 7860 is mandatory for Hugging Face Spaces
ENV PORT 7860
# CRITICAL FIX: Use the 'sh -c' wrapper for Gunicorn to ensure the command executes correctly
CMD ["sh", "-c", "gunicorn brain_tumor_detection.wsgi --bind 0.0.0.0:$PORT"]