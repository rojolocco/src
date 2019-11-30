FROM nvidia/cuda:9.2-cudnn7-devel-ubuntu18.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    curl \
    vim \
    ca-certificates \
    libjpeg-dev \
    libpng-dev \
    sudo \
    apt-utils \
    man \
    tmux \
    less \
    wget \
    iputils-ping \
    zsh \
    htop \
    software-properties-common \
    locales \
    openssh-server \
    xauth \
    rsync &&\
    rm -rf /var/lib/apt/lists/*

RUN curl -o ~/miniconda.sh -O  https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p /opt/conda && \     
    rm ~/miniconda.sh

ENV PATH /opt/conda/bin:$PATH

RUN conda install python=3.6 jupyter pip=9.0.1

RUN conda install pytorch=1.0.1 torchvision=0.2.2 cudatoolkit=9.0 -c pytorch

RUN conda clean -ya 

RUN echo "export PATH=/opt/conda/bin:\$PATH" > /etc/profile.d/conda.sh

RUN pip install scipy scikit-learn pandas cython tensorflow visdom matplotlib

RUN pip install --upgrade pip

RUN rm -rf ~/.cache/pip

ENV PYTHONUNBUFFERED=1

WORKDIR /root

CMD ["jupyter", "notebook", "--allow-root", "--ip=0.0.0.0", "--NotebookApp.custom_display_url='http://localhost:8888'"]