
FROM --platform=linux/amd64 python:3.9-slim
# Use the official Streamlit image
#FROM python:3.12-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install the dependencies
RUN pip install -r requirements.txt

# Copy the rest of the application code
COPY app app/.

# Expose the port where Streamlit runs
EXPOSE 8080

HEALTHCHECK CMD curl --fail http://localhost:8080/_stcore/health

ENV PYTHONPATH=/app

ENTRYPOINT ["streamlit", "run", "app/main.py", "--server.port=8080", "--server.address=0.0.0.0"]

