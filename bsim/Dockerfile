FROM ubuntu:focal

RUN apt-get update && apt-get install -y \
    git curl \
    python3 python3-pip python3-venv python3-wheel python3-setuptools \
    ripgrep htop vim

RUN curl -fsSL https://deb.nodesource.com/setup_19.x | bash \
    && apt-get install -y nodejs

# install miniconda
RUN curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-py39_4.12.0-Linux-x86_64.sh -o /tmp/miniconda.sh \
    && echo "78f39f9bae971ec1ae7969f0516017f2413f17796670f7040725dd83fcff5689  /tmp/miniconda.sh" | sha256sum -c - \
    && bash /tmp/miniconda.sh -b -p /opt/miniconda \
    && rm /tmp/miniconda.sh

# FIXME: use conda environment.yml instead
RUN conda install -c conda-forge slycot

# custom dotfiles
ARG GITHUB_READONLY_TOKEN
ARG GITHUB_USER
RUN cd /opt \
    && git clone https://${GITHUB_USER:?}:${GITHUB_READONLY_TOKEN:?}@github.com/ben-z/dot-conf.git \
    && cd dot-conf \
    && python3 setup.py install \
    && cd - \
    && git clone https://${GITHUB_USER:?}:${GITHUB_READONLY_TOKEN:?}@github.com/ben-z/shell-conf.git
RUN cd /opt/shell-conf \
    && ls -alh  \
    && dot-conf .conf.yaml
