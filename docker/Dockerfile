FROM nvidia/cuda:11.0-cudnn8-devel-ubuntu18.04
 
SHELL [ "/bin/bash", "--login", "-c" ]
 
# install utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        curl \
        ca-certificates \
        sudo \
        bzip2 \
        libx11-6 \
        git \
        wget \
        libjpeg-dev \
        libpng-dev && \
    rm -rf /var/lib/apt/lists/*

# Create a non-root user
ARG username=yue3
ARG uid=1000
ARG gid=100
ENV USER $username
ENV UID $uid
ENV GID $gid
ENV HOME /home/$USER

RUN adduser --disabled-password \
   --gecos "Non-root user" \
   --uid $UID \
   --gid $GID \
   --home $HOME \
   $USER

# switch to that user
USER $USER
 
# install miniconda
ENV MINICONDA_VERSION py37_4.8.2
# if you want a specific version (you shouldn't) replace "latest" with that, e.g. ENV MINICONDA_VERSION py38_4.8.3
 
ENV CONDA_DIR $HOME/miniconda3
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-$MINICONDA_VERSION-Linux-x86_64.sh -O ~/miniconda.sh && \
   chmod +x ~/miniconda.sh && \
   ~/miniconda.sh -b -p $CONDA_DIR && \
   rm ~/miniconda.sh
 
# add conda to path (so that we can just use conda install <package> in the rest of the dockerfile)
ENV PATH=$CONDA_DIR/bin:$PATH
 
# make conda activate command available from /bin/bash --login shells
RUN echo ". $CONDA_DIR/etc/profile.d/conda.sh" >> ~/.profile
 
# make conda activate command available from /bin/bash --interative shells
RUN conda init bash
 
# create a project directory inside user home
ENV PROJECT_DIR $HOME/app
RUN mkdir $PROJECT_DIR
WORKDIR $PROJECT_DIR

# build the conda environment
ENV ENV_PREFIX $PROJECT_DIR/env
RUN conda update --name base --channel defaults conda && \
   conda create --name dino && \
   conda clean --all --yes
RUN conda activate dino
RUN conda install -c pytorch torchvision=0.8.2 cudatoolkit=11.0 pytorch=1.7.1
RUN pip install torchsummary
RUN pip install timm==0.3.2

RUN conda install -c conda-forge jupyterlab
RUN conda install -c conda-forge matplotlib
RUN conda install -c anaconda pandas
RUN conda install -c anaconda numpy
RUN conda install -c conda-forge pydicom
RUN conda install -c conda-forge opencv
RUN conda install -c conda-forge scikit-learn
RUN conda install -c conda-forge scikit-image
RUN conda install -c fastchan fastai
RUN pip install kaggle

USER root
RUN apt-get update -y
RUN conda install -c anaconda pyyaml
RUN conda install -c conda-forge vim
RUN apt-get -y install vim nano
RUN pip install pyclean
RUN env > /root/env.txt #&& cron -f
RUN echo 'alias jupyter="jupyter notebook --ip 0.0.0.0 --no-browser --allow-root --port 8084"' >> ~/.bashrc
