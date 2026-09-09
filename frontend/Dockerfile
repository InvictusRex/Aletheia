FROM python:3.12-slim

WORKDIR /code

COPY frontend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY frontend/app.py frontend/api.py ./
COPY frontend/.streamlit ./.streamlit

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address", "0.0.0.0", "--server.port", "8501"]
