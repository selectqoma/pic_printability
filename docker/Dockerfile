FROM python:3.8

# Install required packages
RUN apt-get update && \
    apt-get install -y \
    cmake \
    gcc \
    g++ \
    build-essential \
    wget \
    bzip2 \
    libgl1-mesa-glx

# Install Python packages
RUN pip install \
    dlib==19.24.0 \
    opencv-python \
    Pillow \
    scikit-image \
    matplotlib \
    streamlit \
    onnx \
    tf2onnx \
    nnabla==1.36.0 \
    onnxruntime



# Set work directory
WORKDIR /app

# Copy source code to the work directory
COPY ./src /app

# Expose Streamlit port
EXPOSE 8501

# Download and decompress models
RUN cd /app && \
    wget https://nnabla.org/pretrained-models/nnabla-examples/face-alignment/2DFAN4_NNabla_model.h5 && \
    wget http://dlib.net/files/mmod_human_face_detector.dat.bz2 && \
    bzip2 -d mmod_human_face_detector.dat.bz2

# Run Streamlit app
CMD ["streamlit", "run", "ui.py"]
