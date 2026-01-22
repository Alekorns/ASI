FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir \
        "fastapi>=0.110" \
        "uvicorn[standard]>=0.29" \
        "pandas>=2.3.3" \
        "scikit-learn>=1.6.1" \
        "joblib>=1.3.0"

COPY src /app/src

RUN mkdir -p /app/models

ENV PYTHONPATH=/app
EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
