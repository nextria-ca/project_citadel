"""
Tiny wrapper around google.protobuf.Struct that gives us:
• zero-copy binary encoding on the wire (smaller & faster than JSON)
• no arbitrary code-execution risk like pickle
"""

from google.protobuf.struct_pb2 import Struct
from google.protobuf.json_format import MessageToDict, ParseDict
from typing import Any, Dict

def dumps(obj: Dict[str, Any]) -> bytes:
    """dict → protobuf binary"""
    s = Struct()
    ParseDict(obj, s, ignore_unknown_fields=False)
    return s.SerializeToString()

def loads(raw: bytes) -> Dict[str, Any]:
    """protobuf binary → dict (keys preserved verbatim)"""
    s = Struct()
    s.ParseFromString(raw)
    # `preserving_proto_field_name=True` keeps snake-case keys intact
    return MessageToDict(s, preserving_proto_field_name=True)
