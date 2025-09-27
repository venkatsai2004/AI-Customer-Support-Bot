FROM python:3.9-slim
WORKDIR /app
COPY streamlit_requirements.txt .
RUN pip install --no-cache-dir -r streamlit_requirements.txt
COPY . .
RUN mkdir -p log
EXPOSE 8501
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]