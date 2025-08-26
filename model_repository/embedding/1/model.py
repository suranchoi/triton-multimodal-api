import os
import threading
from collections import OrderedDict

import numpy as np
import torch
import triton_python_backend_utils as pb_utils
from sentence_transformers import SentenceTransformer, models

# =========================
# Env
# =========================
HF_HOME = os.getenv("HF_HOME", "/data2/huggingface/")
os.makedirs(HF_HOME, exist_ok=True)
os.environ.setdefault("TRANSFORMERS_CACHE", HF_HOME)

# Cache size (model cache per HF ID)
MAX_CACHED = int(os.getenv("ST_CACHE_SIZE", "4"))

# Batch size
ST_BATCH = int(os.getenv("ST_BATCH", "64"))

# Use FP16 (CUDA only)
ST_USE_FP16 = os.getenv("ST_USE_FP16", "0") == "1"

# (Optional) Fixed maximum sequence length (0 to use model default)
ST_MAX_SEQ = int(os.getenv("ST_MAX_SEQ", "0"))

_lock = threading.Lock()
_model_cache = OrderedDict()   # key: hf_id -> SentenceTransformer (mean pooling, normalization applied in encode)


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _build_model(hf_id: str) -> SentenceTransformer:
    """
    Load ST model with hf_id and configure it:
    - Extract backbone (Transformer) for reuse
    - Create view with fixed mean pooling
    - (Normalization handled via encode() option)
    """
    dev = _device()

    # 1) Temporarily load full ST
    try:
        st = SentenceTransformer(hf_id, device=dev, cache_folder=HF_HOME)
    except TypeError:
        st = SentenceTransformer(hf_id, device=dev, cache_dir=HF_HOME)

    # 2) Find internal Transformer module
    transf = None
    for m in st._modules.values():
        if isinstance(m, models.Transformer):
            transf = m
            break
    if transf is None:
        # Fallback: Usually first module is Transformer
        if isinstance(st[0], models.Transformer):
            transf = st[0]
        else:
            raise RuntimeError("Failed to locate Transformer module in SentenceTransformer pipeline.")

    # 3) FP16 (optional)
    if ST_USE_FP16 and torch.cuda.is_available():
        try:
            transf.auto_model.half()
        except Exception:
            pass  

    # 4) Configure mean pooling
    word_dim = transf.get_word_embedding_dimension()
    print(f"word_dim: {word_dim}")  
    pool = models.Pooling(
        word_embedding_dimension=word_dim,
        pooling_mode_mean_tokens=True,
        pooling_mode_cls_token=False,
        pooling_mode_max_tokens=False,
    )
    print(f"pool: {pool}")  

    # 5) Create mean pooling view
    st_view = SentenceTransformer(modules=[transf, pool], device=dev)
    print(f"st_view: {st_view}")  

    # 6) Set max_seq_length (optional)
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
                # Required: TEXT [N], MODEL_NAME [1]
                text_tensor = pb_utils.get_input_tensor_by_name(req, "TEXT")
                model_tensor = pb_utils.get_input_tensor_by_name(req, "MODEL_NAME")
                if text_tensor is None or model_tensor is None:
                    raise ValueError("Inputs TEXT and MODEL_NAME are required.")

                texts = [_to_str(t) for t in text_tensor.as_numpy().tolist()]
                hf_id = _to_str(model_tensor.as_numpy().tolist()[0])

                model = _get_model(hf_id)

                # Normalization always set to True
                with torch.inference_mode():
                    emb = model.encode(
                        texts,
                        batch_size=ST_BATCH,
                        convert_to_numpy=True,
                        normalize_embeddings=True,  
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
