# Use the JetPack 6 PyTorch base image
FROM dustynv/l4t-pytorch:r36.2.0

# Set environment variables for non-interactive installs and CUDA paths
ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/lib/aarch64-linux-gnu/hdf5/serial:/usr/local/cuda/lib64"
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,video,utility

# Install system dependencies
RUN apt-get update && apt-get install -y \
    bash \
    python3-pip \
    python3-dev \
    python3-opencv \
    libopencv-dev \
    libsm6 libxext6 libxrender-dev \
    libhdf5-dev libhdf5-serial-dev \
    hdf5-tools ffmpeg curl wget unzip git \
    && rm -rf /var/lib/apt/lists/*

# Ensure HDF5 is linked correctly
RUN ln -s /usr/lib/aarch64-linux-gnu/hdf5/serial/libhdf5.so /usr/lib/libhdf5.so

# --------- Begin Python package installations ---------
# Step 1: Install Cython (needed for h5py)
RUN pip3 install --no-cache-dir cython==0.29.37

# Step 2: Install numpy (specific version)
RUN pip3 install --no-cache-dir numpy==1.23.5

# Step 3: Install h5py with proper HDF5 linkage
RUN HDF5_DIR=/usr/lib/aarch64-linux-gnu/hdf5/serial pip3 install --no-cache-dir h5py==3.8.0 --no-build-isolation

# Step 4: Install PyTorch-based dependencies and other project packages
RUN pip3 install --no-cache-dir --ignore-installed \
    facenet-pytorch \
    flask==3.0.3 \
    matplotlib==3.5.3 \
    pandas==2.0.3 \
    gdown==5.2.0 \
    tqdm==4.67.1 \
    flask-cors==5.0.0 \
    retina-face==0.0.17 \
    fire==0.7.0 \
    gunicorn==23.0.0

# Optionally install deep_sort_realtime if your project needs it:
RUN pip3 install --no-cache-dir deep_sort_realtime

# Set working directory for your project
WORKDIR /VisionGuard

# Default command (can be overridden by docker-compose)
CMD ["python3"]
