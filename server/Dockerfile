# Use an official Python image with Debian
FROM python:3.9-slim-bullseye

# Install system dependencies including FFmpeg and necessary libraries
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    curl \
    iputils-ping \
    net-tools \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the rest of your application code
COPY . .

# Create a directory for storing logs and data
RUN mkdir -p /app/logs /app/data

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Expose the port your app runs on - we'll actually use Render's PORT variable
EXPOSE 8000

# Run the FastAPI app, using the PORT environment variable provided by Render
CMD ["sh", "-c", "python main.py"]