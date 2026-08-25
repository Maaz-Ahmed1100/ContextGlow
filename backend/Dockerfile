# we use python 311 here
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

# install python packages
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# run the fastapi server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
