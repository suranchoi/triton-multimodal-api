import os, io, glob, tempfile, subprocess, requests, librosa, re, shutil  # ★ cv2 제거
import numpy as np
from PIL import Image
import triton_python_backend_utils as pb_utils
from vllm import LLM, SamplingParams

# ---------------------------
# Helpers for Triton inputs
# ---------------------------
def _get_str_scalar(req, name, default=None):
    t = pb_utils.get_input_tensor_by_name(req, name)
    if t is None:
        return default
    arr = t.as_numpy().reshape(-1)
    if arr.size == 0:
        return default
    val = arr[0]
    if isinstance(val, bytes):
        return val.decode("utf-8", errors="ignore").strip()
    elif isinstance(val, str):
        return val.strip()
    elif isinstance(val, list):
        return str(val[0]).strip() if val else None
    return str(val).strip()


# ---------------------------
# Media loaders
# ---------------------------
def resize_image_for_model(img: Image.Image, target: int = 512) -> Image.Image:
    """Resize image to square target size with aspect ratio preserved and padding if needed."""
    if target not in (256, 512, 768):
        target = 512

    # 스케일링 (긴 변 기준)
    img = img.convert("RGB")
    w, h = img.size
    scale = target / max(w, h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    img = img.resize((new_w, new_h), Image.BICUBIC)

    # 패딩해서 정사각형으로 맞추기
    out = Image.new("RGB", (target, target), (0, 0, 0))
    paste_x = (target - new_w) // 2
    paste_y = (target - new_h) // 2
    out.paste(img, (paste_x, paste_y))
    return out

def load_image_unified(source: str, target_size: int = None) -> Image.Image:
    """Load image from URL or local path -> RGB PIL, optional resize."""
    if isinstance(source, str) and (source.startswith("http://") or source.startswith("https://")):
        resp = requests.get(source, timeout=30)
        resp.raise_for_status()
        img = Image.open(io.BytesIO(resp.content))
    elif isinstance(source, str):
        img = Image.open(source)
    else:
        raise ValueError(f"Unsupported image source type: {type(source)}")

    img = img.convert("RGB")
    if target_size:
        img = resize_image_for_model(img, target=target_size)
    return img

def load_audio_unified(source: str, limit_sec: float = 30.0):
    """Load audio from URL or local path -> (waveform (np.float32), 16000)."""
    if isinstance(source, str) and (source.startswith("http://") or source.startswith("https://")):
        resp = requests.get(source, timeout=60)
        resp.raise_for_status()
        y, sr = librosa.load(io.BytesIO(resp.content), sr=16000, mono=True)
    elif isinstance(source, str):
        y, sr = librosa.load(source, sr=16000, mono=True)
    else:
        raise ValueError(f"Unsupported audio source type: {type(source)}")
    if limit_sec is not None:
        y = y[: int(16000 * limit_sec)]
    return y.astype(np.float32, copy=False), 16000  

PRAGMA_RE = re.compile(r"#\s*(fps|frames)\s*=\s*(\d+)", re.IGNORECASE)

def parse_pragmas(text: str):
    """input_text 안의 #fps=2 #frames=8 같은 매직 코멘트를 파싱."""
    cfg = {}
    for k, v in PRAGMA_RE.findall(text or ""):
        k = k.lower()
        if k == "fps":
            cfg["fps"] = int(v)
        elif k == "frames":
            cfg["frames"] = int(v)
    return cfg

def strip_pragmas(text: str) -> str:
    """모델 입력에서 매직 코멘트를 제거 (가독/잡음 최소화)."""
    return PRAGMA_RE.sub("", text or "").strip()

# ---------------------------
# Video -> frames(using ffmpeg)
# ---------------------------
def ffmpeg_extract_frames(video_src: str, fps: int = 1, max_h: int = None) -> str:
    """
    Extract frames at given FPS into a temp directory as JPGs.
    Optional: scale height to max_h keeping aspect ratio.
    Return: frames_dir path.
    """
    if not os.path.exists(video_src):
        raise FileNotFoundError(video_src)
    work = tempfile.mkdtemp(prefix="vid_frames_")
    out_dir = os.path.join(work, "frames")
    os.makedirs(out_dir, exist_ok=True)

    vf = [f"fps={fps}"]
    if max_h and max_h > 0:
        # 가로는 자동(-2) 유지, 세로를 max_h로 제한
        vf.append(f"scale=-2:{max_h}")

    cmd = [
        "ffmpeg", "-y", "-i", video_src,
        "-vf", ",".join(vf),
        os.path.join(out_dir, "%04d.jpg")
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg frame extract failed: {proc.stderr.decode('utf-8', errors='ignore')}")
    return out_dir

def load_frames_as_pil(frames_dir: str, max_frames: int = 12, target_size: int = None):
    paths = sorted(glob.glob(os.path.join(frames_dir, "*.jpg")))[:max_frames]
    imgs = []
    for p in paths:
        with Image.open(p) as im:
            im = im.convert("RGB")
            if target_size:
                im = resize_image_for_model(im, target=target_size)
            imgs.append(im)
    if not imgs:
        raise RuntimeError("No frames loaded from video.")
    return imgs

# ---------------------------
# LLM Load 
# ---------------------------
class TritonPythonModel:
    def initialize(self, args):
        # if you want to use different model, you can change the model path
        self.model_path = os.environ.get("GEMMA3N_MODEL", "/data2/huggingface/hub/gemma-3n-E4B-it")
        self.llm = LLM(
            model=self.model_path,
            gpu_memory_utilization=0.8,
            max_model_len=4096,
            tensor_parallel_size=1,
            dtype="bfloat16",
            trust_remote_code=True,
            enforce_eager=True,
        )
        self.tokenizer = self.llm.get_tokenizer()
        self.params = SamplingParams(temperature=0.8, max_tokens=1024)

        self.default_video_fps = int(os.environ.get("GEMMA_VIDEO_FPS", "1"))
        self.default_max_video_frames = int(os.environ.get("GEMMA_MAX_VIDEO_FRAMES", "8"))
        self.default_video_maxh = int(os.environ.get("GEMMA_VIDEO_MAXH", "720")) 
        self.default_image_size = int(os.environ.get("GEMMA_IMAGE_SIZE", "512"))

    def execute(self, requests):
        responses = []
        for req in requests:
            frames_tmp_root = None  
            try:
                user_text = _get_str_scalar(req, "input_text")
                if not user_text:
                    raise ValueError("input_text is required.")

                image_src = _get_str_scalar(req, "image_data")
                audio_src = _get_str_scalar(req, "audio_data")
                video_src = _get_str_scalar(req, "video_data")  

                pragmas = parse_pragmas(user_text or "")
                video_fps = pragmas.get("fps", self.default_video_fps)
                max_video_frames = pragmas.get("frames", self.default_max_video_frames)
                video_maxh = self.default_video_maxh

                clean_text = strip_pragmas(user_text)

                images = []
                if image_src:
                    images.append(load_image_unified(image_src, target_size=self.default_image_size))

                if video_src:
                    frames_dir = ffmpeg_extract_frames(video_src, fps=video_fps, max_h=video_maxh)
                    frames_tmp_root = os.path.dirname(frames_dir)
                    images.extend(load_frames_as_pil(frames_dir,
                                                    max_frames=max_video_frames,
                                                    target_size=self.default_image_size))

                audio = load_audio_unified(audio_src, limit_sec=30.0) if audio_src else None

                # chat template
                contents = [{"type": "text", "text": clean_text}]
                for _ in images:
                    contents.append({"type": "image"})
                if audio is not None:
                    contents.append({"type": "audio"})

                messages = [{"role": "user", "content": contents}]
                prompt_text = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )

                mm = {}
                if images:
                    # vLLM은 다중 이미지 리스트를 지원
                    mm["image"] = images if len(images) > 1 else images[0]
                if audio is not None:
                    mm["audio"] = audio  # (np.ndarray(float32), 16000)

                payload = [{"prompt": prompt_text, "multi_modal_data": mm}]

                # debug
                print(f"[MM] images={len(images)} audio={audio is not None} fps={video_fps} frames_cap={max_video_frames} maxh={video_maxh}")

                result = self.llm.generate(payload, sampling_params=self.params)
                output_text = result[0].outputs[0].text.strip()

                out = pb_utils.Tensor("generated_text", np.array([output_text], dtype=object))
                responses.append(pb_utils.InferenceResponse(output_tensors=[out]))

            except Exception as e:
                responses.append(pb_utils.InferenceResponse(
                    output_tensors=[], error=pb_utils.TritonError(f"{type(e).__name__}: {e}")
                ))
            finally:
                if frames_tmp_root and os.path.exists(frames_tmp_root):
                    try:
                        shutil.rmtree(frames_tmp_root)
                    except Exception:
                        pass
        return responses
