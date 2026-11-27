FROM python:3.12-slim@sha256:5e2bdb1b0d542d5039cffb8181c65930093949194a65c365dfb1b0fe13a4fa5b

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/app/src \
    HOME=/tmp MPLCONFIGDIR=/tmp/matplotlib MLFLOW_DISABLE_AGENT_HINT=1

WORKDIR /app
COPY requirements.lock.txt .
RUN pip install --no-cache-dir -c requirements.lock.txt \
    fastapi uvicorn mlflow pandas numpy scikit-learn skops PyYAML matplotlib \
    && useradd --system --uid 10001 --no-create-home novaria
COPY src/novaria/ /app/src/novaria/

USER 10001
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "novaria.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
