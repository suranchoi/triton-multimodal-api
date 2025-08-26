# Triton Multimodal

A multimodal AI model serving system utilizing Triton Inference Server. Provides LLM (Large Language Model) services capable of processing various multimodal inputs including images, audio, and video.

## 🚀 Key Features

- **Multimodal Processing**: Handles images, audio, and video along with text
- **Triton Inference Server**: Leverages NVIDIA Triton for high-performance model serving
- **FastAPI**: Easy access through RESTful API
- **Various Model Support**: 
  - Gemma-3n multimodal model
  - vLLM backend support
- **Embedding Service**: Text embedding generation functionality

## 📁 Project Structure

```
triton-multimodal/
├── README.md                 # Project documentation
├── fastapi/                 # FastAPI web server
│   └── app.py              # Main API server file
├── model_repository/        # Triton model repository
│   ├── llm/                # LLM model configuration
│   └── embedding/          # Embedding model configuration
```

## 💻 Setup

### 0. Prerequisites

#### Install FFmpeg in Docker Environment

For video processing, you need to modify the Dockerfile to install FFmpeg:

```dockerfile
FROM nvcr.io/nvidia/tritonserver:25.07-vllm-python-py3

# Install FFmpeg
RUN apt-get update && apt-get install -y ffmpeg

COPY . . 
RUN pip install --no-cache-dir -r requirements.txt
```

#### Configure Model Paths

Before starting the Triton server, you need to configure the model paths in the following files:

1. **LLM Model Configuration** (`model_repository/llm/1/model.py`):
```python
# Update GEMMA3N_MODEL environment variable or modify this path directly
self.model_path = os.environ.get("GEMMA3N_MODEL", "/data2/huggingface/hub/gemma-3n-E4B-it")
```

2. **Embedding Model Configuration** (`model_repository/embedding/1/model.py`):
```python
# Update HF_HOME environment variable or modify this path
HF_HOME = os.getenv("HF_HOME", "/data2/huggingface/")
```

Make sure to:
- Replace the paths with your local model paths
- Ensure the models are downloaded from HuggingFace
- Set appropriate environment variables if using them
- Check model file permissions

### 1. Start Triton Server

```bash
# 1. Git clone
git clone https://github.com/suranchoi/triton-multimodal-api
cd triton-multimodal-api

# 2. Build Docker image (from the directory containing Dockerfile)
docker build . -t triton-multimodal:latest

# 3. Run Triton server
docker run -d -it --name triton-multimodal-api \
--gpus all \
--shm-size=1G \
--ulimit memlock=-1 --ulimit stack=67108864 \
-p 8000:8000 -p 8001:8001 -p 8002:8002 \
-v $(pwd)/model_repository:/models \
triton-multimodal:latest \
tritonserver --model-repository=/models
```

### 2. Start FastAPI Server

```bash
cd fastapi
python app.py
```

Or

```bash
uvicorn fastapi.app:app --host 0.0.0.0 --port 8080
```

## 🌐 API Endpoints

The FastAPI server provides the following endpoints:

### Multimodal Generation API

```http
POST /multimodal/generate
```

**Requests:**
```bash
# 1. Image 
curl -s http://localhost:8080/multimodal/generate \
  -H "Content-Type: application/json" \
  -d '{
    "input_text": "Describe this image.",
    "image_data": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg"
  }'

# 2. Video
curl -s http://localhost:8080/multimodal/generate \
  -H "Content-Type: application/json" \
  -d '{
    "input_text": "Describe the scene transitions in this video.",
    "video_data": "/data2/llm/triton-multimodal-api/data/dks_llm.mp4"
  }'

# 3. Audio
curl -s http://localhost:8080/multimodal/generate \
  -H "Content-Type: application/json" \
  -d '{
    "input_text": "Transcribe this audio clip:",
    "audio_data": "https://huggingface.co/microsoft/Phi-4-multimodal-instruct/resolve/main/examples/what_is_shown_in_this_image.wav"
  }'
```

### Embedding Generation API

```http
POST /embeddings/{model_name}
```

**Request:**
```bash
curl -X POST "http://localhost:8080/embeddings/sentence-transformers/all-MiniLM-L6-v2" \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello world"}'
```

## 🔍 Troubleshooting

### Common Issues

1. **CUDA Memory Shortage**
   ```bash
   # Adjust model parallel processing settings
   tensor_parallel_size=2
   max_model_len=2048
   gpu_memory_utilization=0.8
   ```

2. **FFmpeg Installation Required**
   ```bash
   # Ubuntu/Debian
   sudo apt-get install ffmpeg
   
   # macOS
   brew install ffmpeg
   ```

3. **Model Path Error**
   - Verify that the model path is correct
   - Ensure model files are downloaded

### Log Checking

```bash
# Check Triton server logs
docker logs -f <triton_container_id>
``` 