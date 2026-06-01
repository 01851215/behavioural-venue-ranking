FROM python:3.11-slim

WORKDIR /app

# Install only what the API needs (not the full ML stack)
RUN pip install --no-cache-dir fastapi uvicorn[standard] pandas numpy pydantic

COPY serve.py .
COPY uk_fsq_venue_scores.csv .
COPY uk_fsq_businesses.csv .
COPY london_birank_venue_scores.csv .
COPY london_businesses.csv .

ENV MODEL=uk_fsq
ENV DATA_DIR=/app
ENV PORT=8000

EXPOSE 8000

CMD ["python3", "serve.py"]
