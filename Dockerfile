FROM nvcr.io/nvidia/tritonserver:25.07-vllm-python-py3

COPY . . 

RUN pip install --no-cache-dir -r requirements.txt