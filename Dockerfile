FROM apache/spark:3.5.0

USER root

# Install Python pip
RUN apt-get update && apt-get install -y python3-pip && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy and install Python dependencies
COPY requirements.txt /app/requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application files
COPY src/ /app/src/
COPY data/ /app/data/

# Create output and models directories
RUN mkdir -p /app/output /app/models

# Set environment variables
ENV SPARK_HOME=/opt/spark
ENV PYTHONPATH=/app

WORKDIR /app