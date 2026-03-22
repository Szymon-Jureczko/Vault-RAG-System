FROM python:3.12-bookworm

# System dependencies
# - tesseract-ocr + language packs for OCR processing
# - libgl1 / libglib2.0-0 for PyMuPDF / OpenCV
# - sqlite3 for state_db CLI access
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    wget \
    vim \
    build-essential \
    libpq-dev \
    # Tesseract OCR
    tesseract-ocr \
    tesseract-ocr-eng \
    libtesseract-dev \
    # PyMuPDF / image processing
    libgl1 \
    libglib2.0-0 \
    # SQLite CLI
    sqlite3 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip tooling
RUN pip install --upgrade pip setuptools wheel

# Create a non-root user matching common host UIDs
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=${USER_UID}

RUN groupadd --gid ${USER_GID} ${USERNAME} \
    && useradd --uid ${USER_UID} --gid ${USER_GID} -m ${USERNAME} \
    && apt-get update \
    && apt-get install -y sudo \
    && echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/${USERNAME} \
    && chmod 0440 /etc/sudoers.d/${USERNAME} \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /home/${USERNAME}/.cache && chown -R ${USERNAME}:${USERNAME} /home/${USERNAME}/.cache

USER ${USERNAME}

# Add user-local bin to PATH (for pip-installed CLIs)
ENV PATH="/home/${USERNAME}/.local/bin:${PATH}"

# Persist pip cache across rebuilds (speeds up postCreateCommand)
ENV PIP_CACHE_DIR=/home/${USERNAME}/.cache/pip

WORKDIR /workspace
