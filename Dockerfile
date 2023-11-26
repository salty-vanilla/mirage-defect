ARG PYTHON_VERSION=3.9.18
ARG CUDA_TAG=11.8.0-devel-ubuntu22.04

FROM python:${PYTHON_VERSION}-slim-bullseye as python-build-stage

FROM nvidia/cuda:${CUDA_TAG} as base

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

COPY --from=python-build-stage /usr/local /usr/local
RUN apt-get update && \
    apt-get -y install --no-install-recommends software-properties-common && \
    apt-get update && \
    apt-get -y install --no-install-recommends \
    curl \
    git \
    libgl1-mesa-dev \
    libpq-dev \
    poppler-data \
    poppler-utils && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /tmp
RUN curl -O http://nz2.archive.ubuntu.com/ubuntu/pool/main/o/openssl/libssl1.1_1.1.1f-1ubuntu2.20_amd64.deb && \
    dpkg -i libssl1.1_1.1.1f-1ubuntu2.20_amd64.deb && \
    curl -O http://archive.ubuntu.com/ubuntu/pool/universe/libf/libffi7/libffi7_3.3-5ubuntu1_amd64.deb && \
    dpkg -i libffi7_3.3-5ubuntu1_amd64.deb

ARG CUDNN_URL=https://developer.download.nvidia.com/compute/redist/cudnn/v8.6.0/local_installers/11.8/cudnn-linux-x86_64-8.6.0.163_cuda11-archive.tar.xz
RUN curl -o cudnn.tar.xz $CUDNN_URL && \
    mkdir cudnn && \
    tar xvf cudnn.tar.xz -C cudnn --strip-components=1&& \
    mv cudnn/include/* /usr/local/cuda/include/ && \
    mv cudnn/lib/* /usr/local/cuda/lib64 && \
    echo LD_LIBRARY_PATH=/usr/local/cuda/lib64 >> /etc/environment

ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64

WORKDIR /
RUN rm -rf /tmp/*


FROM base as builder
ENV POETRY_HOME=/opt/poetry
ENV PATH=${POETRY_HOME}/bin:${PATH}
RUN curl -sSL https://install.python-poetry.org | python3 - && \
    echo PATH="${PATH}" > /etc/environment


FROM builder as dev
RUN apt-get update && \
    apt-get install --no-install-recommends -y \
    bash-completion \
    curl \
    git \
    nano \
    postgresql-client \
    unzip && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

RUN poetry completions bash >> /etc/bash_completion.d/poetry && \
    echo source /etc/bash_completion.d/poetry >> /etc/bash.bashrc


FROM dev as dev-user
ARG USER=user-name-goes-here
ARG UID=1000
ARG GID=$UID
ENV TZ=Asia/Tokyo

# Create the user
RUN groupadd --gid $GID $USER \
    && useradd --uid $UID --gid $GID -m $USER --shell /bin/bash \
    #
    # [Optional] Add sudo support. Omit if you don't need to install software after connecting.
    && apt-get update \
    && apt-get install -y sudo procps gdb \
    && echo $USER ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USER \
    && chmod 0440 /etc/sudoers.d/$USER

# ********************************************************
# * Anything else you want to do like clean up goes here *
# ********************************************************

# [Optional] Set the default user. Omit if you want to keep the default as root.
USER $USER
