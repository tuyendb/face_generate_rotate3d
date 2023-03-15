FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04

ENV LANG C.UTF-8
ENV DEBIAN_FRONTEND noninteractive
ENV MPLLOCALFREETYPE 1

RUN apt-get update && apt-get install -y software-properties-common

RUN apt-get update && apt-get install -y \
    cmake\
    build-essential \
    git \
    wget \
    vim \
    curl \
    zip \
    zlib1g-dev \
    unzip \
    pkg-config \
    libgl-dev \
    libblas-dev \
    liblapack-dev \
    python3-tk \
    python3-wheel \
    graphviz \
    libhdf5-dev \
    python3.8 \
    python3.8-dev \
    python3.8-distutils \
    swig \
    apt-transport-https \
    lsb-release \
    libpng-dev \
    ca-certificates &&\
    apt-get clean &&\
    ln -s /usr/bin/python3.8 /usr/local/bin/python &&\
    ln -s /usr/bin/python3.8 /usr/local/bin/python3 &&\
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py &&\
    python3 get-pip.py &&\
    rm get-pip.py &&\
    # best practice to keep the Docker image lean
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

WORKDIR /

RUN pip3 --no-cache-dir install --index-url https://mirror-python.t-idc.net/simple/ --default-timeout=1000 torch==1.7.1+cu101 torchvision==0.8.2+cu101 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip3 --no-cache-dir install --index-url https://mirror-python.t-idc.net/simple/ --default-timeout=1000 iopath
RUN pip3 --no-cache-dir install --index-url https://mirror-python.t-idc.net/simple/ --default-timeout=1000 fvcore
RUN pip3 --no-cache-dir install --index-url https://mirror-python.t-idc.net/simple/ --default-timeout=1000 dlib
RUN pip3 --no-cache-dir install --index-url https://mirror-python.t-idc.net/simple/ --default-timeout=1000 kornia==0.5.5
RUN pip3 --no-cache-dir install --index-url https://mirror-python.t-idc.net/simple/ --default-timeout=1000 dominate==2.6.0
RUN pip3 --no-cache-dir install --index-url https://mirror-python.t-idc.net/simple/ --default-timeout=1000 trimesh==3.9.20
RUN pip3 --no-cache-dir install --index-url https://mirror-python.t-idc.net/simple/ --default-timeout=1000 opencv-python==4.6.0.66
RUN pip3 --no-cache-dir install --index-url https://mirror-python.t-idc.net/simple/ --default-timeout=1000 scipy==1.4.1
RUN pip3 --no-cache-dir install --index-url https://mirror-python.t-idc.net/simple/ --default-timeout=1000 scikit-image==0.16.2
RUN pip3 --no-cache-dir install --index-url https://mirror-python.t-idc.net/simple/ --default-timeout=1000 tensorflow==2.3.0
RUN pip3 --no-cache-dir install --index-url https://mirror-python.t-idc.net/simple/ --default-timeout=1000 tf-slim==1.1.0
RUN pip3 install --no-cache-dir --default-timeout=1000 mediapipe==0.8.11
RUN pip3 --no-cache-dir install --index-url https://mirror-python.t-idc.net/simple/ --default-timeout=1000 protobuf==3.9.2
RUN pip3 --no-cache-dir install --index-url https://mirror-python.t-idc.net/simple/ --default-timeout=1000 cached_property
RUN pip3 --no-cache-dir install --index-url https://mirror-python.t-idc.net/simple/ --default-timeout=1000 aiohttp
# RUN pip3 uninstall -y pytorch3d
ENV LD_LIBRARY_PATH /usr/local/cuda/extras/CUPTI/lib64:${LD_LIBRARY_PATH}

COPY . .
RUN cp -r /libs/pytorch3d /usr/local/lib/python3.8/dist-packages
RUN cp -r /libs/pytorch3d-0.6.2-py3.8.egg-info /usr/local/lib/python3.8/dist-packages
RUN cd libs && python3 -m pip --no-cache-dir install --index-url https://mirror-python.t-idc.net/simple/ --default-timeout=1000 *.whl
RUN rm -r libs

RUN export PYTHONPATH=./
CMD ["python", "app_dev.py"]
