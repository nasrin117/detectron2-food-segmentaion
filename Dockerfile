FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get install -y \
	python3-opencv ca-certificates git wget sudo

RUN ln -sv /usr/bin/python3 /usr/bin/python
RUN pip install tensorboard 
RUN pip install cmake
RUN pip install onnx

RUN pip install fvcore

RUN git clone https://github.com/facebookresearch/detectron2 detectron2_repo
ENV FORCE_CUDA="1"
ARG TORCH_CUDA_ARCH_LIST="Kepler;Kepler+Tesla;Maxwell;Maxwell+Tegra;Pascal;Volta;Turing"
ENV TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}"

RUN pip install -e detectron2_repo
ENV FVCORE_CACHE="/tmp"
WORKDIR /detectron2_repo
RUN pip install opencv-python

