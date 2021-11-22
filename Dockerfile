# fastai2 v0.0.30 with all pillow/libjpeg performance improvements 
FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-devel

RUN apt-get update && apt-get install -y wget git nasm zlib1g-dev software-properties-common cmake && \
    add-apt-repository ppa:ubuntu-toolchain-r/test && \
    apt update && \
    apt -y install  gcc-9  g++-9 && \
    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 60 --slave /usr/bin/g++ g++ /usr/bin/g++-9 && \
    apt-get clean
    #rm -rf /var/lib/apt/lists/*```

RUN conda install -c pytorch -c fastai fastai==2.0.19 jupyter==1.0.0 pybind11==2.7.0
RUN conda install -c fastai nvidia-ml-py3==7.352.0
RUN pip install nbdev==1.1.23 timm==0.4.12
RUN conda install -y "pylint<2.0.0" rope==0.21.1

RUN cd /workspace && git clone --recurse-submodules https://github.com/fastai/fastai && \ 
    cd /workspace/fastai && git checkout 8eef914b721a22bd0e3f53e1404d8afde395c5af
RUN cd /workspace && git clone https://github.com/fastai/fastcore && \
    cd /workspace/fastcore && git checkout 6ede1f85f823d2bf30ecda74857ed8536c61f46f

RUN conda uninstall -y --force pillow pil jpeg libtiff libjpeg-turbo
RUN pip   uninstall -y         pillow pil jpeg libtiff libjpeg-turbo
RUN conda install -yc conda-forge libjpeg-turbo==2.1.1
RUN CC="cc -mavx2" pip install --no-cache-dir -U --force-reinstall pillow-simd==7.0.0.post3
RUN cd /workspace/fastcore && pip install -e "."
RUN cd /workspace/fastai && pip install -e "."


RUN pip install onnx==1.7.0 ipyexperiments==0.1.17 onnxruntime-gpu==1.4.0 image-similarity-measures==0.3.5 mysql-connector-python==8.0.18
RUN apt-get install -y libgl1-mesa-glx
RUN conda install -yc conda-forge "opencv==4.5.3"

#starting jupyter notebook with the password naphash
CMD /bin/bash -c "jupyter notebook --ip=0.0.0.0 --port=8889 --allow-root --no-browser --NotebookApp.password='sha1:26c148c4c3d7:f3fb45584ca8f933451d8e05e6861b54f3fe7822'"

##switch to non-root user with sudo privileges (run with --user 1000:1000 or equivalent)
RUN apt-get update && apt-get install sudo && \
    adduser --disabled-password --gecos "" udocker && \
    adduser udocker sudo && echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
USER udocker
RUN sudo chown -R udocker:udocker /workspace

# example docker commands:
# docker build -t naphash .
# docker run -p 8889:8889 --shm-size=28G -u $UID:$UID -v $PWD:/workspace/local -it naphash
