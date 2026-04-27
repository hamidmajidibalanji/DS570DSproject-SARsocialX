FROM python:3.10

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

CMD ["streamlit", "run", "app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]