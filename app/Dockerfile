FROM python:3.13
WORKDIR /app
COPY requirements.txt .
COPY models/ .
COPY logs/ .
COPY main.py .
COPY app.py .
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 8000 8501
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port 8000 --reload & streamlit run app.py --server.port 8501"]
