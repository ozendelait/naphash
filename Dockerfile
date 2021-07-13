# fastai2 v0.0.30 with all pillow/libjpeg performance improvements 
FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-devel

RUN apt-get update && apt-get install -y wget git nasm zlib1g-dev software-properties-common && \
    add-apt-repository ppa:ubuntu-toolchain-r/test && \
    apt update && \
    apt -y install  gcc-9  g++-9 && \
    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 60 --slave /usr/bin/g++ g++ /usr/bin/g++-9 && \
    apt-get clean
    #rm -rf /var/lib/apt/lists/*```

RUN conda install -c pytorch -c fastai fastai pytorch jupyter
RUN conda install -c fastai nvidia-ml-py3
RUN pip install nbdev timm
RUN conda install -y "pylint<2.0.0" rope

RUN cd /workspace && git clone --recurse-submodules https://github.com/fastai/fastai && \ 
    cd /workspace/fastai && git checkout 8eef914b721a22bd0e3f53e1404d8afde395c5af
RUN cd /workspace && git clone https://github.com/fastai/fastcore && \
    cd /workspace/fastcore && git checkout 6ede1f85f823d2bf30ecda74857ed8536c61f46f

RUN conda uninstall -y --force pillow pil jpeg libtiff libjpeg-turbo
RUN pip   uninstall -y         pillow pil jpeg libtiff libjpeg-turbo
RUN conda install -yc conda-forge libjpeg-turbo
RUN CC="cc -mavx2" pip install --no-cache-dir -U --force-reinstall pillow-simd
RUN cd /workspace/fastcore && pip install -e "."
RUN cd /workspace/fastai && pip install -e "."


RUN pip install onnx==1.7.0 ipyexperiments==0.1.17 onnxruntime-gpu==1.4.0 image-similarity-measures==0.3.5 mysql-connector-python==8.0.18
RUN apt-get install -y libgl1-mesa-glx
RUN pip install "opencv-python>=4.4.0.0" pylibjpeg-libjpeg

#starting jupyter notebook with the password naphash
CMD /bin/bash -c "jupyter notebook --ip=0.0.0.0 --port=8889 --allow-root --no-browser --NotebookApp.password='sha1:26c148c4c3d7:f3fb45584ca8f933451d8e05e6861b54f3fe7822'"

##switch to non-root user with sudo privileges
#RUN apt-get update && apt-get install sudo && \
#    adduser --disabled-password --gecos "" udocker && \
#    adduser udocker sudo && echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
#USER udocker
#RUN sudo chown -R udocker:udocker /workspace

