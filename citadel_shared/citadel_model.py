from __future__ import annotations

import abc
import base64
import json
import os
import re
from typing import Any, AsyncGenerator, Dict, List, Optional

from google.protobuf.json_format import MessageToDict
from google.protobuf.struct_pb2 import Struct

from citadel_shared.logging_setup import get_logger


class CitadelModel(abc.ABC):
    def __init__(self, model_dir: str, config: Dict):
        self.model_dir = model_dir
        self.config = config
        self.logger = get_logger(f"{self.__class__.__module__}.{self.__class__.__qualname__}")
        self.gpu_id: Optional[int] = self._resolve_gpu_id(config)
        if self.gpu_id is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)
        self.gpu_id = 0

    # ─────────────────────────── properties ───────────────────────────
    @property
    def device(self) -> str:
        return f"cuda:{self.gpu_id}" if self.gpu_id is not None else "cpu"

    @device.setter
    def device(self, value: str):
        if isinstance(value, str) and value.startswith("cuda:"):
            try:
                self.gpu_id = int(value.split(":", 1)[1])
            except ValueError as exc:
                raise ValueError(f"Invalid CUDA device format: {value!r}") from exc
        elif value == "cpu":
            self.gpu_id = None
        else:
            raise ValueError("device must be 'cpu' or 'cuda:<int>'")

    # ─────────────────── helpers / utilities ──────────────────────────
    @staticmethod
    def _resolve_gpu_id(cfg: Dict) -> Optional[int]:
        gpu = cfg.get("gpu_id")
        if gpu not in (None, "", -1):
            try:
                return int(gpu)
            except (ValueError, TypeError):
                return None

        ids = cfg.get("gpu_ids")
        if isinstance(ids, (list, tuple)) and ids:
            try:
                return int(ids[0])
            except (ValueError, TypeError):
                return None
        return None

    @staticmethod
    def coerce_to_dict(inp: Any) -> Dict:
        """Normalise Struct / bytes / JSON str / dict → plain dict."""
        # 1) dict with embedded protobuf bytes
        if isinstance(inp, dict) and "__bytes__" in inp:
            raw = base64.b64decode(inp["__bytes__"])
            st  = Struct(); st.ParseFromString(raw)
            return MessageToDict(st, preserving_proto_field_name=True)

        # 2) already a dict
        if isinstance(inp, dict):
            return inp

        # 3) raw bytes → protobuf Struct → dict, else JSON
        if isinstance(inp, (bytes, bytearray)):
            try:
                st = Struct(); st.ParseFromString(inp)
                return MessageToDict(st, preserving_proto_field_name=True)
            except Exception:
                try:
                    return json.loads(inp)
                except Exception:
                    return {"__raw__": inp}

        # 4) JSON string
        if isinstance(inp, str):
            try:
                return json.loads(inp)
            except Exception:
                return {"__raw__": inp}

        # fallback
        return {"__raw__": inp}


    def clear_cache(self) -> None:
        """
        Free framework/device caches (best-effort).
        Default: PyTorch CUDA empty_cache(); safe no-op if unavailable.
        """
        try:
            import torch  # type: ignore
            if getattr(torch, "cuda", None) and torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
        except Exception:
            pass

    # ──────────────────── model lifecycle API ─────────────────────────
    @abc.abstractmethod
    def init(self) -> None: ...

    @abc.abstractmethod
    def finalize(self) -> None: ...

    @abc.abstractmethod
    def execute(self, inp: Dict) -> Any: ...

    # -------------------------- batch ---------------------------------
    def execute_batch(self, items: List[Any]) -> List[Any]:
        decoded = [self.coerce_to_dict(x) for x in items]
        return [self.execute(x) for x in decoded]

    # ------------------------- stream ---------------------------------
    async def stream_execute(self, inp: Any) -> AsyncGenerator[Any, None]:
        raise NotImplementedError(f"{self.__class__.__name__} does not implement streaming")
