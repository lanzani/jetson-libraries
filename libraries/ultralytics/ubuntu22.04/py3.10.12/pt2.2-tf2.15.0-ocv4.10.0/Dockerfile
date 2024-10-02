FROM ghcr.io/lanzani/opencv:ubuntu22.04-py3.10.12-ocv4.10.0

# === Pytorch ==========================================================================================================

# Install necessary packages
RUN apt update && apt install -y \
    python3 \
    python3-pip

# Upgrade pip
RUN pip3 install --upgrade pip

# Install Python packages
RUN pip3 install numpy scipy

# Install PyTorch and torchvision
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

RUN pip3 install ultralytics