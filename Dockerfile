FROM python:3.11-slim

WORKDIR /app

# 1️⃣ Copy requirements FIRST (for proper layer caching)
COPY requirements.txt .

# 2️⃣ Install deps
RUN pip install --no-cache-dir -r requirements.txt

# 3️⃣ Copy application code
COPY src/api/ .

# 4️⃣ Copy trained artifacts
RUN mkdir -p models/trained
COPY models/trained/ models/trained/

EXPOSE 8000 9100

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]


