# Use official Python runtime as a parent image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    pypi_index_url=https://pypi.org/simple

# Set output to show logging immediately
ENV PORT=8080

WORKDIR /app

# Install system dependencies (needed for some python packages like psycopg)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Create a non-root user for security (best practice)
RUN useradd -m appuser && chown -R appuser /app
USER appuser

# Expose the port
EXPOSE 8080

# Run the application with uvicorn
# We use the shell form to allow variable expansion if needed, but array form is safer.
# Using 'sh -c' allows using the $PORT env var in the command args.
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT} --workers 1 --proxy-headers"]
