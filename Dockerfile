FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py .
COPY ringbench ./ringbench
COPY static ./static
ENV RINGBENCH_DATA_DIR=/data RINGBENCH_HOST=0.0.0.0 PORT=18940 PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=2
EXPOSE 18940
HEALTHCHECK --interval=8s --timeout=3s --start-period=4s --retries=3 CMD python3 -B -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:18940/api/health')"
CMD ["python3", "-B", "app.py"]
