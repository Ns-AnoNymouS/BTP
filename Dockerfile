# Use an official Python image
FROM python:3.12-slim

# Install FFmpeg and other dependencies
RUN apt-get update && apt-get install -y ffmpeg && apt-get clean

# Set the working directory
WORKDIR /app

# Copy the application files
COPY . /app

ENV PYTHONUNBUFFERED=1

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Run the application
CMD ["python", "server.py"]
