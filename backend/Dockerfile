FROM python:3.13-slim

WORKDIR /app

COPY pyproject.toml README.md requirements.txt ./
COPY backend ./backend

RUN pip install --no-cache-dir -e .[pdf]

EXPOSE 8000

CMD ["uvicorn", "diagnostician.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
