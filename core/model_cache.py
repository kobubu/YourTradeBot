"""Auto-cleanup on startup can be controlled via env:
- PURGE_MODEL_CACHE_ON_START=1     -> remove all cached models on import
- PURGE_EXPIRED_MODEL_CACHE_ON_START=1 and MODEL_CACHE_TTL_SECONDS=86400 (default)
  -> remove entries older than TTL
"""
from pathlib import Path
import os
import json
import hashlib
import time
import joblib
from typing import Optional, Tuple, Dict, Any

MODEL_ROOT = Path(__file__).resolve().parent / "artifacts" / "models"
MODEL_ROOT.mkdir(parents=True, exist_ok=True)

def _hash_obj(obj) -> str:
    b = json.dumps(obj, sort_keys=True, default=str).encode()
    return hashlib.sha1(b).hexdigest()

def make_cache_key(ticker: str, model_type: str, params: Dict, data_sig: str) -> str:
    params_hash = _hash_obj(params)
    safe = f"{ticker}__{model_type}__{params_hash}__{data_sig}"
    return _hash_obj(safe)

def _model_dir_for_key(key: str) -> Path:
    d = MODEL_ROOT / key
    d.mkdir(parents=True, exist_ok=True)
    return d

def save_sklearn_model(key: str, model_obj, meta: Dict[str, Any]):
    d = _model_dir_for_key(key)
    joblib.dump(model_obj, d / "model.pkl")
    with open(d / "meta.json", "w", encoding="utf8") as f:
        json.dump(meta, f, ensure_ascii=False)

def load_sklearn_model(key: str) -> Tuple[Optional[object], Optional[Dict[str, Any]]]:
    d = MODEL_ROOT / key
    mfile = d / "model.pkl"
    mmeta = d / "meta.json"
    if not mfile.exists() or not mmeta.exists():
        return None, None
    model = joblib.load(mfile)
    meta = json.load(open(mmeta, encoding="utf8"))
    return model, meta

def save_statsmodels_result(key: str, res_obj, meta: Dict[str, Any]):
    d = _model_dir_for_key(key)
    joblib.dump(res_obj, d / "sm_res.pkl")
    with open(d / "meta.json", "w", encoding="utf8") as f:
        json.dump(meta, f, ensure_ascii=False)

def load_statsmodels_result(key: str) -> Tuple[Optional[object], Optional[Dict[str, Any]]]:
    d = MODEL_ROOT / key
    mfile = d / "sm_res.pkl"
    mmeta = d / "meta.json"
    if not mfile.exists() or not mmeta.exists():
        return None, None
    res = joblib.load(mfile)
    meta = json.load(open(mmeta, encoding="utf8"))
    return res, meta

def save_tf_model(key: str, keras_model, meta: Dict[str, Any]):
    d = _model_dir_for_key(key)
    model_dir = d / "tf_model"
    if model_dir.exists():
        try:
            import shutil
            shutil.rmtree(model_dir)
        except Exception:
            pass
    keras_model.save(str(model_dir))
    with open(d / "meta.json", "w", encoding="utf8") as f:
        json.dump(meta, f, ensure_ascii=False)

def load_tf_model(key: str) -> Tuple[Optional[object], Optional[Dict[str, Any]]]:
    d = MODEL_ROOT / key
    model_dir = d / "tf_model"
    mmeta = d / "meta.json"
    if not model_dir.exists() or not mmeta.exists():
        return None, None
    from tensorflow import keras
    model = keras.models.load_model(str(model_dir))
    meta = json.load(open(mmeta, encoding="utf8"))
    return model, meta

def remove_key(key: str):
    d = MODEL_ROOT / key
    if d.exists():
        try:
            import shutil
            shutil.rmtree(d)
        except Exception:
            pass

def purge_all(reason: str = "") -> int:
    """Remove ALL cached models. Returns number of removed entries."""
    if not MODEL_ROOT.exists():
        return 0
    removed = 0
    for child in MODEL_ROOT.iterdir():
        if child.is_dir():
            try:
                import shutil
                shutil.rmtree(child)
                removed += 1
            except Exception:
                pass
    try:
        print(f"[model_cache] purge_all removed={removed} reason={reason}")
    except Exception:
        pass
    return removed

def purge_expired(ttl_seconds: int = None) -> int:
    """Remove items whose meta['trained_at'] older than now-ttl_seconds."""
    if ttl_seconds is None:
        try:
            ttl_seconds = int(os.getenv("MODEL_CACHE_TTL_SECONDS", "86400"))
        except Exception:
            ttl_seconds = 86400

    now = int(time.time())
    removed = 0
    if not MODEL_ROOT.exists():
        return 0

    for d in MODEL_ROOT.iterdir():
        if not d.is_dir():
            continue
        meta_path = d / "meta.json"
        try:
            if not meta_path.exists():
                import shutil
                shutil.rmtree(d)
                removed += 1
                continue
            meta = json.load(open(meta_path, encoding="utf8"))
            trained_at = int(meta.get("trained_at", 0))
            if trained_at <= 0 or (now - trained_at) > ttl_seconds:
                import shutil
                shutil.rmtree(d)
                removed += 1
        except Exception:
            try:
                import shutil
                shutil.rmtree(d)
                removed += 1
            except Exception:
                pass

    try:
        print(f"[model_cache] purge_expired ttl={ttl_seconds}s removed={removed}")
    except Exception:
        pass
    return removed

def get_cache_info() -> Dict[str, Any]:
    """Return brief diagnostic info about cache size and entries."""
    info = {"root": str(MODEL_ROOT), "entries": []}
    if not MODEL_ROOT.exists():
        return info
    for d in MODEL_ROOT.iterdir():
        if not d.is_dir():
            continue
        meta = {}
        try:
            meta_path = d / "meta.json"
            if meta_path.exists():
                meta = json.load(open(meta_path, encoding="utf8"))
        except Exception:
            pass
        info["entries"].append({"dir": d.name, "meta": meta})
    return info

def _startup_cleanup():
    """
    Controls:
      PURGE_MODEL_CACHE_ON_START=1          -> purge_all()
      PURGE_EXPIRED_MODEL_CACHE_ON_START=1  -> purge_expired(TTL)
      MODEL_CACHE_TTL_SECONDS               -> TTL value (default 86400)
    """
    try:
        if os.getenv("PURGE_MODEL_CACHE_ON_START", "0") == "1":
            purge_all("env PURGE_MODEL_CACHE_ON_START=1")
        elif os.getenv("PURGE_EXPIRED_MODEL_CACHE_ON_START", "0") == "1":
            ttl = int(os.getenv("MODEL_CACHE_TTL_SECONDS", "86400"))
            purge_expired(ttl)
    except Exception:
        pass

_startup_cleanup()