from __future__ import annotations
from typing import Any, Dict, List, Callable
from functools import lru_cache

_PRIMITIVES = {"str": str, "int": int, "float": float, "bool": bool}

def _check_primitive(val: Any, expected: str, path: str):
    py_type = _PRIMITIVES[expected]
    if not isinstance(val, py_type):
        raise TypeError(f"{path}: expected {expected}, got {type(val).__name__}")

def _check_list(val: Any, inner: str, path: str):
    if not isinstance(val, list):
        raise TypeError(f"{path}: expected list, got {type(val).__name__}")
    for i, item in enumerate(val):
        _check_primitive(item, inner, f"{path}[{i}]")

def _check_dict(val: Any, schema: Dict[str, Any], path: str):
    if not isinstance(val, dict):
        raise TypeError(f"{path}: expected dict, got {type(val).__name__}")
    for key, sub_schema in schema.items():
        if key not in val:
            raise KeyError(f"{path}: missing key '{key}'")
        _validate(val[key], sub_schema, f"{path}.{key}")

def _validate(val: Any, schema: Any, path: str):
    if isinstance(schema, str):
        if schema.startswith("list[") and schema.endswith("]"):
            inner = schema[5:-1]
            _check_list(val, inner, path)
        elif schema in _PRIMITIVES:
            _check_primitive(val, schema, path)
        else:
            raise ValueError(f"{path}: unknown type spec '{schema}'")
    elif isinstance(schema, dict):
        _check_dict(val, schema, path)
    else:
        raise ValueError(f"{path}: invalid schema definition {schema!r}")

def validate(payload: Any, inputs_schema: List[Dict[str, Any]]):
    """Fallback - parse & walk the schema every call (kept for compatibility)."""
    if not isinstance(payload, dict):
        raise TypeError("root: model input must be a dict")
    for field in inputs_schema:
        name, typ = field["name"], field["type"]
        if name not in payload:
            raise KeyError(f"root: missing field '{name}'")
        _validate(payload[name], typ, name)

Checker = Callable[[Any], None]

def _compile_type(spec: Any, path: str = "") -> Checker:
    if isinstance(spec, str):
        if spec in _PRIMITIVES:
            pt = _PRIMITIVES[spec]
            def _chk(val, p=path, t=pt, s=spec):
                if not isinstance(val, t):
                    raise TypeError(f"{p}: expected {s}, got {type(val).__name__}")
            return _chk
        if spec.startswith("list[") and spec.endswith("]"):
            inner = spec[5:-1]
            inner_chk = _compile_type(inner, f"{path}[]")
            def _chk(val, p=path):
                if not isinstance(val, list):
                    raise TypeError(f"{p}: expected list, got {type(val).__name__}")
                for i, item in enumerate(val):
                    inner_chk(item)
            return _chk
        raise ValueError(f"{path}: unknown type spec '{spec}'")

    if isinstance(spec, dict):
        subs = {k: _compile_type(s, f"{path}.{k}" if path else k)
                for k, s in spec.items()}
        def _chk(val, p=path):
            if not isinstance(val, dict):
                raise TypeError(f"{p}: expected dict, got {type(val).__name__}")
            for k, ch in subs.items():
                if k not in val:
                    raise KeyError(f"{p}: missing key '{k}'")
                ch(val[k])
        return _chk

    raise ValueError(f"{path}: invalid schema definition {spec!r}")

def _freeze(o):
    if isinstance(o, dict):
        return tuple((k, _freeze(v)) for k, v in sorted(o.items()))
    if isinstance(o, list):
        return tuple(_freeze(x) for x in o)
    return o

@lru_cache(maxsize=None)
def _compile_schema_cached(frozen) -> Checker:            # noqa: D401
    checks = [(name, _compile_type(typ, name)) for name, typ in frozen]
    def _validator(payload):
        if not isinstance(payload, dict):
            raise TypeError("root: model input must be a dict")
        for fname, chk in checks:
            if fname not in payload:
                raise KeyError(f"root: missing field '{fname}'")
            chk(payload[fname])
    return _validator

def compile_schema(inputs_schema: List[Dict[str, Any]]) -> Checker:
    """Return *callable* validator compiled once for a given schema."""
    frozen = tuple((f["name"], _freeze(f["type"])) for f in inputs_schema)
    return _compile_schema_cached(frozen)