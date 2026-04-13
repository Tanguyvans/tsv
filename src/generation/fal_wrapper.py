"""Wrapper minimaliste autour du SDK fal-client.

Charge FAL_KEY depuis .env, expose `submit_and_wait` avec :
- cache disque par hash (model, prompt, image_path, params)
- retry exponentiel sur erreurs transitoires
- log JSONL des appels (coût/duration/status)
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()


@dataclass
class FalConfig:
    cache_dir: Path = Path("outputs/fal_cache")
    log_path: Path = Path("outputs/logs/fal_calls.jsonl")
    max_retries: int = 4
    base_backoff: float = 2.0


def _key(model: str, args: dict, image_paths: list[str] | None) -> str:
    h = hashlib.sha256()
    h.update(model.encode())
    h.update(json.dumps(args, sort_keys=True, default=str).encode())
    for p in image_paths or []:
        h.update(p.encode())
        try:
            h.update(str(Path(p).stat().st_mtime).encode())
        except OSError:
            pass
    return h.hexdigest()[:16]


def _file_to_data_url(path: str) -> str:
    p = Path(path)
    mime = "image/png" if p.suffix.lower() == ".png" else "image/jpeg"
    data = base64.b64encode(p.read_bytes()).decode()
    return f"data:{mime};base64,{data}"


def _log(cfg: FalConfig, entry: dict) -> None:
    cfg.log_path.parent.mkdir(parents=True, exist_ok=True)
    with cfg.log_path.open("a") as f:
        f.write(json.dumps(entry) + "\n")


def submit_and_wait(
    model: str,
    arguments: dict[str, Any],
    image_paths: list[str] | None = None,
    cfg: FalConfig | None = None,
) -> dict[str, Any]:
    """Appelle un endpoint fal-ai et retourne le résultat (avec cache)."""
    cfg = cfg or FalConfig()
    cfg.cache_dir.mkdir(parents=True, exist_ok=True)

    cache_key = _key(model, arguments, image_paths)
    cache_file = cfg.cache_dir / f"{cache_key}.json"
    if cache_file.exists():
        return json.loads(cache_file.read_text())

    if not os.environ.get("FAL_KEY"):
        raise RuntimeError("FAL_KEY is not set. Copy .env.example → .env and add your key.")

    import fal_client  # imported lazily to keep CLI tools light

    last_err: Exception | None = None
    for attempt in range(cfg.max_retries):
        t0 = time.time()
        try:
            handler = fal_client.submit(model, arguments=arguments)
            result = handler.get()
            duration = time.time() - t0
            _log(cfg, {
                "ts": time.time(),
                "model": model,
                "duration": duration,
                "status": "ok",
                "cache_key": cache_key,
            })
            cache_file.write_text(json.dumps(result))
            return result
        except Exception as e:  # noqa: BLE001
            last_err = e
            wait = cfg.base_backoff ** attempt
            _log(cfg, {
                "ts": time.time(),
                "model": model,
                "status": "retry",
                "attempt": attempt,
                "error": str(e)[:300],
            })
            time.sleep(wait)

    _log(cfg, {"ts": time.time(), "model": model, "status": "failed", "error": str(last_err)})
    raise RuntimeError(f"fal call failed after {cfg.max_retries} attempts: {last_err}")


def upload_file(path: str) -> str:
    """Upload local file to fal CDN, return URL. Falls back to data URL on failure."""
    try:
        import fal_client
        return fal_client.upload_file(path)
    except Exception:
        return _file_to_data_url(path)
