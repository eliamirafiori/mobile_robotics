# Use Debian-slim instead of Alpine for glibc compatibility with MuJoCo & NVIDIA GPUs
FROM python:3.11-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install OpenGL, EGL (for headless GPU), GLFW (for visual GUI), and X11 libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libgl1-mesa-glx \
    libegl1 \
    libegl1-mesa \
    libglfw3 \
    libglew2.2 \
    libosmesa6 \
    libxrandr2 \
    libxinerama1 \
    libxcursor1 \
    libxi6 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Expose NVIDIA GPU capabilities for both compute and OpenGL/EGL rendering
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics,display

# Upgrade pip and install latest MuJoCo, Gymnasium-Robotics, and scientific stack
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    mujoco \
    gymnasium \
    gymnasium-robotics \
    numpy \
    scipy \
    imageio[ffmpeg] \
    matplotlib

WORKDIR /workspace

CMD ["bash"]