import os
import uvicorn
import requests
from typing import Optional, List, Literal, Union
from fastapi import FastAPI, HTTPException, Path
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware

# =========================
# Env
# =========================
TRITON_URL = os.environ.get("TRITON_URL", "http://localhost:8000")
TRITON_MODEL_GEN = os.environ.get("TRITON_MODEL_GEN", "llm")           # 멀티모달 생성 모델 이름
TRITON_MODEL_EMB = os.environ.get("TRITON_MODEL_EMB", "embedding")     # 임베딩 모델 이름
TRITON_TIMEOUT = float(os.environ.get("TRITON_TIMEOUT", "120"))

TRITON_INFER_GEN = f"{TRITON_URL}/v2/models/{TRITON_MODEL_GEN}/infer"
TRITON_BASE_EMB = f"{TRITON_URL}/v2/models"  # /{model}/infer 로 접근

# =========================
# FastAPI init
# =========================
app = FastAPI(title="Multimodal + Embeddings API (FastAPI → Triton)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "OPTIONS"],
    allow_headers=["*"],
)

# =========================
# utils
# =========================
def _bytes_tensor(name: str, data: Union[str, List[str]]):
    if isinstance(data, list):
        return {"name": name, "shape": [len(data)], "datatype": "BYTES", "data": data}
    else:
        return {"name": name, "shape": [1], "datatype": "BYTES", "data": [data]}

# =========================
# 1) Multimodal Generate API
# =========================
class GenerateRequest(BaseModel):
    input_text: str = Field(..., description="프롬프트 텍스트")
    image_data: Optional[str] = Field(None, description="이미지 경로 or URL")
    audio_data: Optional[str] = Field(None, description="오디오 경로 or URL")
    video_data: Optional[str] = Field(None, description="비디오 경로(로컬)")
    # config.pbtxt 수정 없이 per-request 제어 → 매직 코멘트로 input_text에 삽입
    fps: Optional[int] = Field(None, description="비디오 샘플링 FPS")
    frames: Optional[int] = Field(None, description="추출 프레임 수 상한")

class GenerateResponse(BaseModel):
    text: str

@app.post("/multimodal/generate", response_model=GenerateResponse)
def multimodal_generate(req: GenerateRequest):
    # 매직 코멘트 구성
    header = []
    if req.fps is not None:
        header.append(f"#fps={int(req.fps)}")
    if req.frames is not None:
        header.append(f"#frames={int(req.frames)}")
    input_text = ((" ".join(header) + " ") if header else "") + req.input_text

    inputs = [_bytes_tensor("input_text", input_text)]
    if req.image_data:
        inputs.append(_bytes_tensor("image_data", req.image_data))
    if req.audio_data:
        inputs.append(_bytes_tensor("audio_data", req.audio_data))
    if req.video_data:
        inputs.append(_bytes_tensor("video_data", req.video_data))

    payload = {"inputs": inputs, "outputs": [{"name": "generated_text"}]}
    try:
        r = requests.post(TRITON_INFER_GEN, json=payload, timeout=TRITON_TIMEOUT)
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Triton connection error: {e}")
    if r.status_code != 200:
        raise HTTPException(status_code=r.status_code, detail=r.text)

    try:
        out = r.json()
        gt = next(o for o in out.get("outputs", []) if o.get("name") == "generated_text")
        text = gt["data"][0] if gt.get("data") else ""
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Invalid Triton response: {e}")

    return GenerateResponse(text=text)

# =========================
# 2) Embedding API 
# =========================
# Triton embedding model.py가 기대하는 텐서:
# - TEXT: STRING, shape [N] - 입력 텍스트 배열
# - MODEL_NAME: STRING, shape [1] - HuggingFace 모델 ID
# 출력:
# - EMBEDDINGS: FLOAT32, shape [N, dim] - 임베딩 벡터들
# - DIM: INT32, shape [1] - 임베딩 차원 수

class EmbeddingRequest(BaseModel):
    # texts: 단일 문자열 또는 문자열 배열 모두 허용
    input: Union[str, List[str]] = Field(..., description="텍스트 또는 텍스트 리스트")
    encoding_format: Literal["float", "base64"] = Field("float", description="인코딩 형식")

class EmbeddingResponse(BaseModel):
    object: str = "list"
    model: str
    data: List[dict]
    usage: dict


@app.post("/embeddings/{model_name:path}", response_model=EmbeddingResponse)
def embeddings(model_name: str = Path(...), req: EmbeddingRequest = ...):
    """
    텍스트 임베딩 생성 API
    OpenAI API 호환 형식으로 응답
    """
    # 입력 텍스트를 리스트로 변환
    if isinstance(req.input, str):
        texts = [req.input]
    else:
        texts = req.input
    
    if not texts:
        raise HTTPException(status_code=400, detail="No input texts provided")
    
    # Triton 입력 텐서 구성
    inputs = [
        _bytes_tensor("TEXT", texts),
        _bytes_tensor("MODEL_NAME", model_name)
    ]
    
    payload = {
        "inputs": inputs,
        "outputs": [
            {"name": "EMBEDDINGS"},
            {"name": "DIM"}
        ]
    }
    
    # Triton 임베딩 모델 호출
    triton_emb_url = f"{TRITON_BASE_EMB}/{TRITON_MODEL_EMB}/infer"
    try:
        r = requests.post(triton_emb_url, json=payload, timeout=TRITON_TIMEOUT)
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Triton connection error: {e}")
    
    if r.status_code != 200:
        raise HTTPException(status_code=r.status_code, detail=f"Triton error: {r.text}")
    
    try:
        result = r.json()
        outputs = result.get("outputs", [])
        
        # EMBEDDINGS 텐서 추출
        emb_output = next((o for o in outputs if o.get("name") == "EMBEDDINGS"), None)
        dim_output = next((o for o in outputs if o.get("name") == "DIM"), None)
        
        if emb_output is None or dim_output is None:
            raise ValueError("Missing required output tensors")
        
        embeddings_data = emb_output.get("data", [])
        dim = dim_output.get("data", [0])[0]
        
        # 임베딩 데이터를 [N, dim] 형태로 재구성
        num_texts = len(texts)
        embeddings = []
        for i in range(num_texts):
            start_idx = i * dim
            end_idx = start_idx + dim
            embedding = embeddings_data[start_idx:end_idx]
            embeddings.append(embedding)
        
        # OpenAI API 호환 형식으로 응답 구성
        data = []
        for i, embedding in enumerate(embeddings):
            data.append({
                "object": "embedding",
                "index": i,
                "embedding": embedding
            })
        
        return EmbeddingResponse(
            model=model_name,
            data=data,
            usage={
                "prompt_tokens": sum(len(text.split()) for text in texts),
                "total_tokens": sum(len(text.split()) for text in texts)
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Invalid Triton response: {e}")

# =========================
# Entry
# =========================
if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", "8080")),
        reload=False,
    )
