import os
import threading
from collections import OrderedDict

import numpy as np
import torch
import triton_python_backend_utils as pb_utils
from sentence_transformers import SentenceTransformer, models

# =========================
# 환경 변수
# =========================s
HF_HOME = os.getenv("HF_HOME", "/data2/huggingface/")
os.makedirs(HF_HOME, exist_ok=True)
os.environ.setdefault("TRANSFORMERS_CACHE", HF_HOME)

# 캐시 크기 (HF ID별 모델 캐시)
MAX_CACHED = int(os.getenv("ST_CACHE_SIZE", "4"))

# 배치 크기
ST_BATCH = int(os.getenv("ST_BATCH", "64"))

# FP16 사용 (CUDA에서만)
ST_USE_FP16 = os.getenv("ST_USE_FP16", "0") == "1"

# (옵션) 최대 시퀀스 길이 고정(0이면 모델 기본값 사용)
ST_MAX_SEQ = int(os.getenv("ST_MAX_SEQ", "0"))

_lock = threading.Lock()
_model_cache = OrderedDict()   # key: hf_id -> SentenceTransformer (mean pooling, normalize는 encode에서 적용)


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _build_model(hf_id: str) -> SentenceTransformer:
    """
    hf_id로 ST 모델을 로드하되,
    - 백본(Transformer)만 재사용할 수 있도록 꺼내서
    - mean Pooling으로 고정한 view를 구성
    - (normalize는 encode() 옵션으로 처리)
    """
    dev = _device()

    # 1) 임시로 전체 ST를 로드
    try:
        st = SentenceTransformer(hf_id, device=dev, cache_folder=HF_HOME)
    except TypeError:
        st = SentenceTransformer(hf_id, device=dev, cache_dir=HF_HOME)

    # 2) 내부 Transformer 모듈 찾기
    transf = None
    for m in st._modules.values():
        if isinstance(m, models.Transformer):
            transf = m
            break
    if transf is None:
        # 안전장치: 보통 첫 모듈이 Transformer
        if isinstance(st[0], models.Transformer):
            transf = st[0]
        else:
            raise RuntimeError("Failed to locate Transformer module in SentenceTransformer pipeline.")

    # 3) FP16 (옵션)
    if ST_USE_FP16 and torch.cuda.is_available():
        try:
            transf.auto_model.half()
        except Exception:
            pass  

    # 4) mean pooling 고정
    word_dim = transf.get_word_embedding_dimension()
    print(f"word_dim: {word_dim}")  
    pool = models.Pooling(
        word_embedding_dimension=word_dim,
        pooling_mode_mean_tokens=True,
        pooling_mode_cls_token=False,
        pooling_mode_max_tokens=False,
    )
    print(f"pool: {pool}")  

    # 5) mean pooling view 구성
    st_view = SentenceTransformer(modules=[transf, pool], device=dev)
    print(f"st_view: {st_view}")  

    # 6) max_seq_length (옵션)
    if ST_MAX_SEQ and hasattr(st_view, "max_seq_length"):
        st_view.max_seq_length = int(ST_MAX_SEQ)
    print(f"st_view.max_seq_length: {st_view.max_seq_length}")  
    st_view.eval()
    return st_view


def _get_model(hf_id: str) -> SentenceTransformer:
    with _lock:
        if hf_id in _model_cache:
            _model_cache.move_to_end(hf_id)
            return _model_cache[hf_id]
        m = _build_model(hf_id)
        _model_cache[hf_id] = m
        if len(_model_cache) > MAX_CACHED:
            _model_cache.popitem(last=False)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return m


def _to_str(x):
    if isinstance(x, (bytes, bytearray)):
        return x.decode()
    return x


class TritonPythonModel:
    def initialize(self, args):
        pass

    def execute(self, requests):
        responses = []
        for req in requests:
            try:
                # 필수: TEXT [N], MODEL_NAME [1]
                text_tensor = pb_utils.get_input_tensor_by_name(req, "TEXT")
                model_tensor = pb_utils.get_input_tensor_by_name(req, "MODEL_NAME")
                if text_tensor is None or model_tensor is None:
                    raise ValueError("Inputs TEXT and MODEL_NAME are required.")

                texts = [_to_str(t) for t in text_tensor.as_numpy().tolist()]
                hf_id = _to_str(model_tensor.as_numpy().tolist()[0])

                model = _get_model(hf_id)

                # normalize는 항상 True로 고정
                with torch.inference_mode():
                    emb = model.encode(
                        texts,
                        batch_size=ST_BATCH,
                        convert_to_numpy=True,
                        normalize_embeddings=True,  # ★ 고정
                    )
                print(f"emb: {emb}")    
                dim = int(emb.shape[1])

                out = [
                    pb_utils.Tensor("EMBEDDINGS", emb.astype(np.float32, copy=False)),
                    pb_utils.Tensor("DIM", np.array([dim], dtype=np.int32)),
                ]
                responses.append(pb_utils.InferenceResponse(output_tensors=out))
            except Exception as e:
                responses.append(
                    pb_utils.InferenceResponse(
                        output_tensors=[],
                        error=pb_utils.TritonError(f"{type(e).__name__}: {e}")
                    )
                )
        return responses
