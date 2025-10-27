FROM python:3.13.5-slim-bookworm

WORKDIR /app

COPY pyproject.toml .

COPY uv.lock .

COPY . .

RUN pip install uv && uv sync --locked

RUN uv --version

# Copy the rest of your application code
COPY . .

# Expose the port your FastAPI application will run on
EXPOSE 8000

CMD ["uvicorn", "fastapi_app:app", "--host", "0.0.0.0", "--port", "8000"]