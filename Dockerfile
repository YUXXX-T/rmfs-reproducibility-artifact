FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg

WORKDIR /artifact
COPY requirements-artifact.txt pyproject.toml README.md LICENSE ./
COPY src ./src
RUN python -m pip install --no-cache-dir -r requirements-artifact.txt && \
    python -m pip install --no-cache-dir -e .
COPY . .
CMD ["python", "scripts/verify_artifacts.py"]
