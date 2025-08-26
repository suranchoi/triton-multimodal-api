FROM nvcr.io/nvidia/tritonserver:25.07-vllm-python-py3

# Install FFmpeg
RUN apt-get update && apt-get install -y ffmpeg

COPY . . 
RUN pip install --no-cache-dir -r requirements.txt