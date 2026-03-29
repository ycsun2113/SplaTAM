###############################################################################
#  Ubuntu 22.04  +  CUDA 12.1 
###############################################################################
ARG BASE_IMAGE=nvidia/cuda:12.1.1-devel-ubuntu22.04
FROM ${BASE_IMAGE}


###############################################################################
# 0 ─ Global settings & utilities
###############################################################################
ARG USER_NAME=splatam
ARG USER_ID=1000


# Prevent anything requiring user input
ENV DEBIAN_FRONTEND=noninteractive
ENV TERM=linux

ENV TZ=America
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Use bash for every RUN so that "source" works
SHELL ["/bin/bash", "-c"]

###############################################################################
# 1 ─ Base APT packages
###############################################################################

# Basic packages
RUN apt-get -y update \
    && apt-get -y install \
    python3-pip \
    sudo \
    vim \
    wget \
    curl \
    software-properties-common \
    doxygen \
    git \
    tmux \
    g++ \
    gcc \
    build-essential \
    checkinstall \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get -y update \
    && apt-get -y install \
    cmake \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y --no-install-recommends locales && \
    locale-gen en_US.UTF-8
ENV LANG=en_US.UTF-8

###############################################################################
# 2 - CUDA environment (provided by the devel base image)
###############################################################################
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Set CUDA architectures for building extensions without GPU detection (A6000 = 8.6)
ENV TORCH_CUDA_ARCH_LIST="8.6"

###############################################################################
# 3 - Install GCC/G++ 11
###############################################################################
RUN add-apt-repository ppa:ubuntu-toolchain-r/test -y && \
    apt-get update && \
    apt-get install -y gcc-11 g++-11 && \
    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 100 && \
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 100


###############################################################################
# 4 - Install Graphics / build dependencies
###############################################################################
RUN apt-get -y update \
    && apt-get -y install \
    libglew-dev \
    libassimp-dev \
    libgtk-3-dev \
    libglfw3-dev \
    libavdevice-dev \
    libavcodec-dev \
    libeigen3-dev \
    libxxf86vm-dev \
    libembree-dev \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y --no-install-recommends \
    libffi-dev \
    libbz2-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libfreetype6-dev \
    libssl-dev \
    libsqlite3-dev \
    zlib1g-dev \
    libyaml-dev \
    libwebp-dev \
    ffmpeg \
    python3-dev \
    libxt6 \
    libxt-dev \
    && rm -rf /var/lib/apt/lists/*


###############################################################################
# 5 ─ Install Miniconda (installed system‑wide in /opt/conda)
###############################################################################
ENV CONDA_DIR=/opt/conda
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-py311_24.3.0-0-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    mkdir -p $CONDA_DIR && \
    bash /tmp/miniconda.sh -b -u -p $CONDA_DIR && \
    rm /tmp/miniconda.sh && \
    ln -s $CONDA_DIR/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    $CONDA_DIR/bin/conda clean -afy
ENV PATH=$CONDA_DIR/bin:$PATH
# Make conda usable in non‑login shells
RUN echo ". $CONDA_DIR/etc/profile.d/conda.sh" >> /etc/bash.bashrc

###############################################################################
# 6 - Create SplaTAM conda environment
###############################################################################
# Create env with Python 3.10 only (no conda pytorch — avoids iJIT symbol conflict)
RUN $CONDA_DIR/bin/conda create -n splatam python=3.10 -y && \
    $CONDA_DIR/bin/conda clean -afy

# Install PyTorch via pip (compatible with CUDA 12.1)
RUN $CONDA_DIR/bin/conda run -n splatam \
        pip install --no-cache-dir \
        torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 \
        --index-url https://download.pytorch.org/whl/cu121

# Install regular pip packages (excluding the git+ CUDA extension)
COPY requirements.txt /tmp/requirements.txt
RUN $CONDA_DIR/bin/conda run -n splatam \
        pip install --no-cache-dir $(grep -v '^git+' /tmp/requirements.txt | tr '\n' ' ') && \
    rm /tmp/requirements.txt

# Clone and install diff-gaussian-rasterization with --no-build-isolation
RUN cd /tmp && \
    git clone https://github.com/JonathonLuiten/diff-gaussian-rasterization-w-depth.git && \
    cd diff-gaussian-rasterization-w-depth && \
    git checkout cb65e4b86bc3bd8ed42174b72a62e8d3a3a71110 && \
    git submodule update --init --recursive && \
    $CONDA_DIR/bin/conda run -n splatam \
        pip install --no-cache-dir --no-build-isolation . && \
    rm -rf /tmp/diff-gaussian-rasterization-w-depth


###############################################################################
# 7 - User Setup & Entrypoint
###############################################################################
RUN useradd -m -l -u ${USER_ID} -s /bin/bash ${USER_NAME} \
    && usermod -aG video ${USER_NAME} \
    && export PATH=$PATH:/home/${USER_NAME}/.local/bin

# Give them passwordless sudo
RUN echo "${USER_NAME} ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers


# Switch to user to run user-space commands
USER ${USER_NAME}
WORKDIR /home/${USER_NAME}

RUN sudo chown -R ${USER_NAME} /home/${USER_NAME}

COPY ./entrypoint.sh /entrypoint.sh
RUN sudo chmod +x /entrypoint.sh
ENTRYPOINT [ "/entrypoint.sh" ]