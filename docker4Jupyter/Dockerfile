FROM ubuntu:16.04

LABEL maintainer="Zenodia"

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
RUN mkdir ~/share2host
# Install Ubuntu packages
RUN apt-get update && apt-get install -y \
    wget \
    bzip2 \
    ca-certificates \
    build-essential \
    curl \
    git-core \
    htop \
    pkg-config \
    unzip \
    unrar \
    tree
# Clean up
RUN apt-get clean && rm -rf /var/lib/apt/lists/*
# Install Jupyter config
RUN mkdir ~/.ssh && touch ~/.ssh/known_hosts
RUN ssh-keygen -F github.com || ssh-keyscan github.com >> ~/.ssh/known_hosts
RUN mkdir ~/.jupyter
COPY jupyter_config.py ~/.jupyter/

#install anaconda
RUN echo 'export PATH=/opt/conda/bin:$PATH' > /etc/profile.d/conda.sh
RUN wget --quiet https://repo.anaconda.com/archive/Anaconda3-5.2.0-Linux-x86_64.sh -O ~/anaconda.sh
RUN /bin/bash ~/anaconda.sh -b -p /opt/conda
ENV PATH /opt/conda/bin:$PATH
# Update Anaconda
RUN conda update conda && conda update anaconda && conda update --all
# Install Jupyter theme
RUN pip install msgpack jupyterthemes
RUN jt -t grade3
# Install other Python packages
RUN pip install scikit-learn scikit-image pillow numpy pandas scipy matplotlib
RUN pip install opencv-contrib-python-headless \
    tensorflow==1.10.0 \
    keras==2.2.2
#Configure access to Jupyter
WORKDIR /root/share2host
EXPOSE 8888
CMD jupyter lab --no-browser --ip=0.0.0.0 --allow-root --NotebookApp.token=''