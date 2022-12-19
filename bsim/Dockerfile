FROM ubuntu:focal

RUN apt-get update && apt-get install -y \
    git curl \
    python3 python3-pip python3-venv python3-wheel python3-setuptools \
    ripgrep htop vim

RUN curl -fsSL https://deb.nodesource.com/setup_19.x | bash \
    && apt-get install -y nodejs

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
