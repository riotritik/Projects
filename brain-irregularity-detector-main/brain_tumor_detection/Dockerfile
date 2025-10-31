FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY . /app
# Allow execution of the startup script
RUN chmod +x /app/entrypoint.sh
# Port 7860 is mandatory for Hugging Face Spaces
ENV PORT 7860
# NEW START COMMAND: Use the entrypoint script
CMD ["/app/entrypoint.sh"]