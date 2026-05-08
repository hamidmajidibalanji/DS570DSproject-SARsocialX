FROM python:3.8

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

# Run data pipeline before app starts  
# RUN python src/data/download_data.py   
RUN python src/preprocessing/preprocess.py

CMD ["streamlit", "run", "app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]