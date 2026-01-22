# Use a Python base image
FROM python:3.10-slim

# Set the working directory
WORKDIR /app
ENV PIP_DEFAULT_TIMEOUT=1000

# Copy the requirements file
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir --no-input -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the port for the ingestion service
EXPOSE 8000

# Run the main application
CMD ["python", "main.py"]