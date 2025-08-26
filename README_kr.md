# Triton Multimodal

Triton Inference Server를 활용한 멀티모달 AI 모델 서빙 시스템입니다. 이미지, 오디오, 비디오를 포함한 다양한 멀티모달 입력을 처리할 수 있는 LLM(Large Language Model) 서비스를 제공합니다.


## 🚀 주요 기능

- **멀티모달 처리**: 이미지, 오디오, 비디오를 텍스트와 함께 처리
- **Triton Inference Server**: 고성능 모델 서빙을 위한 NVIDIA Triton 활용
- **FastAPI**: RESTful API를 통한 간편한 접근
- **다양한 모델 지원**: 
  - Gemma-3n 멀티모달 모델
  - vLLM 백엔드 지원
- **임베딩 서비스**: 텍스트 임베딩 생성 기능

## 📁 프로젝트 구조

```
triton-multimodal/
├── README.md                 # 프로젝트 설명서
├── fastapi/                 # FastAPI 웹 서버
│   └── app.py              # API 서버 메인 파일
├── model_repository/        # Triton 모델 저장소
│   ├── llm/                # LLM 모델 설정
│   └── embedding/          # 임베딩 모델 설정
```


## 💻 사용법

### 1. Triton 서버 시작

```bash
# 1. Git clone
git clone https://github.com/suranchoi/triton-multimodal-api
cd triton-multimodal-api

# 2. 도커 이미지 빌드 (Dockerfile 이 있는 경로에서)
docker build . -t triton-multimodal:latest

# 3. Triton 서버 실행
docker run -d -it --name triton-multimodal-api \
--gpus all \
--shm-size=1G \
--ulimit memlock=-1 --ulimit stack=67108864 \
-p 8000:8000 -p 8001:8001 -p 8002:8002 \
-v $(pwd)/model_repository:/models \
triton-multimodal:latest \
tritonserver --model-repository=/models
```

### 2. FastAPI 서버 시작

```bash
cd fastapi
python app.py
```

또는

```bash
uvicorn fastapi.app:app --host 0.0.0.0 --port 8080
```


## 🌐 API 엔드포인트

FastAPI 서버는 다음과 같은 엔드포인트를 제공합니다:

### 멀티모달 생성 API

```http
POST /multimodal/generate
```

**요청:**
```bash
# 1. 이미지 
curl -s http://localhost:8080/multimodal/generate \
  -H "Content-Type: application/json" \
  -d '{
    "input_text": "이미지를 묘사해줘.",
    "image_data": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg"
  }'

# 2. 비디오
curl -s http://localhost:8080/multimodal/generate \
  -H "Content-Type: application/json" \
  -d '{
    "input_text": "이 영상의 장면 전환을 설명해줘.",
    "video_data": "/data2/llm/triton-multimodal-api/data/dks_llm.mp4"
  }'

# 3. 오디오
curl -s http://localhost:8080/multimodal/generate \
  -H "Content-Type: application/json" \
  -d '{
    "input_text": "Transcribe this audio clip:",
    "audio_data": "https://huggingface.co/microsoft/Phi-4-multimodal-instruct/resolve/main/examples/what_is_shown_in_this_image.wav"
  }'
```

### 임베딩 생성 API

```http
POST /embeddings/{model_name}
```

**요청:**
```bash
curl -X POST "http://localhost:8080/embeddings/sentence-transformers/all-MiniLM-L6-v2" \
  -H "Content-Type: application/json" \
  -d '{"input": "안녕하세요"}'
```


## 🔍 문제 해결

### 일반적인 문제

1. **CUDA 메모리 부족**
   ```bash
   # 모델 병렬 처리 설정 조정
   tensor_parallel_size=2
   max_model_len=2048
   gpu_memory_utilization=0.8
   ```

2. **FFmpeg 설치 필요**
   ```bash
   # Ubuntu/Debian
   sudo apt-get install ffmpeg
   
   # macOS
   brew install ffmpeg
   ```

3. **모델 경로 오류**
   - 모델 경로가 올바른지 확인
   - 모델 파일이 다운로드되었는지 확인


### 로그 확인

```bash
# Triton 서버 로그 확인
docker logs -f <triton_container_id>
```