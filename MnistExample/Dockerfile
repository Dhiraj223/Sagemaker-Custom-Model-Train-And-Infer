# Base Image with TensorFlow
FROM tensorflow/tensorflow:2.17.0

# Install system-level dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    dos2unix \
    nginx \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt /opt/ml/code/requirements.txt
RUN python3 -m pip install --upgrade pip && \
    pip3 install --no-cache-dir -r /opt/ml/code/requirements.txt

# Create directories for SageMaker paths
RUN mkdir -p /opt/ml/code /opt/ml/model /opt/ml/input/data

# Copy training and inference scripts
COPY training_script.py /opt/ml/code/train.py
COPY inference.py /opt/ml/code/inference.py
COPY wsgi.py /opt/ml/code/wsgi.py
COPY nginx.conf /opt/ml/code/nginx.conf

# Copy and setup the serve and train scripts
COPY serve /usr/bin/serve
COPY train /usr/bin/train
RUN dos2unix /usr/bin/train /usr/bin/serve && \
    chmod +x /usr/bin/serve /usr/bin/train

# Configure nginx
RUN rm /etc/nginx/sites-enabled/default

# Expose port 8080 for serving
EXPOSE 8080

# Define the default entrypoint
ENTRYPOINT ["bash", "-c"]

# CMD argument determines whether to run train or inference
CMD ["if [ \"$1\" = 'serve' ]; then /usr/bin/serve; else /usr/bin/train; fi"]
