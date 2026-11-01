FROM python:3.11-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MUJOCO_GL=osmesa \
    PYOPENGL_PLATFORM=osmesa

# System libraries needed by MuJoCo for headless (software) rendering
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    libgl1 \
    libglfw3 \
    libosmesa6 \
    libglew-dev \
    libegl1 \
    patchelf \
    && rm -rf /var/lib/apt/lists/*

# The `mujoco` pip wheel bundles the MuJoCo binaries, so no manual download is needed
RUN pip install --upgrade pip \
    && pip install --upgrade mujoco gymnasium-robotics

WORKDIR /app

CMD ["python"]