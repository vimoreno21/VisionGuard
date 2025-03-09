# Use Ubuntu 20.04 as base (Jetson-compatible)
FROM ubuntu:20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/lib/aarch64-linux-gnu/hdf5/serial:${LD_LIBRARY_PATH}"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    python3-opencv \
    libopencv-dev \
    libsm6 libxext6 libxrender-dev \
    libhdf5-dev libhdf5-serial-dev \
    hdf5-tools ffmpeg curl wget unzip git \
    && rm -rf /var/lib/apt/lists/*

# Make sure HDF5 is linked correctly
RUN ln -s /usr/lib/aarch64-linux-gnu/hdf5/serial/libhdf5.so /usr/lib/libhdf5.so

# Install dependencies in the correct order (this sequence worked in testing)
# Step 1: Install Cython first (needed for h5py)
RUN pip3 install --no-cache-dir cython==0.29.37

# Step 2: Install numpy with specific version
RUN pip3 install --no-cache-dir numpy==1.23.5

# Step 3: Install h5py with proper HDF5 linkage
RUN HDF5_DIR=/usr/lib/aarch64-linux-gnu/hdf5/serial pip3 install --no-cache-dir h5py==3.8.0 --no-build-isolation

# Step 4: Install Jetson-specific TensorFlow 
RUN pip3 install --no-cache-dir --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v60 tensorflow==2.10.0

# Step 5: Install Keras with matching version
RUN pip3 install --no-cache-dir keras==2.10.0

# Step 6: Install remaining dependencies (using system OpenCV instead of pip version)
RUN pip3 install --no-cache-dir \
    deepface==0.0.93 \
    flask==3.0.3 \
    matplotlib==3.5.3 \
    pandas==2.0.3 \
    gdown==5.2.0 \
    tqdm==4.67.1 \
    flask-cors==5.0.0 \
    retina-face==0.0.17 \
    fire==0.7.0 \
    gunicorn==23.0.0

# Step 7: Install MTCNN after TensorFlow to ensure compatibility
RUN pip3 install --no-cache-dir mtcnn==0.1.1

# Set working directory (matching your mount point)
WORKDIR /VisionGuard

# Copy application files into the container
COPY . /VisionGuard

# Default command
CMD ["python3"]