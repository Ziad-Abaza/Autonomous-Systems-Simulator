"""
Binary and JSON communication protocol between the Simulation Platform and External AI/RL models.
Protocol Version 2.0 adds contract discovery, rich schema negotiation, and detailed termination reasons.
"""

from __future__ import annotations
import json
from typing import Dict, Any, Tuple, Optional, Union
import numpy as np


class MessageType:
    HANDSHAKE = "HANDSHAKE"
    HANDSHAKE_ACK = "HANDSHAKE_ACK"
    DISCOVER_CONTRACT = "DISCOVER_CONTRACT"
    CONTRACT_ACK = "CONTRACT_ACK"
    RESET = "RESET"
    RESET_ACK = "RESET_ACK"
    STEP = "STEP"
    STEP_ACK = "STEP_ACK"
    GET_STATE = "GET_STATE"
    STATE_ACK = "STATE_ACK"
    ERROR = "ERROR"


PROTOCOL_VERSION = "2.0"
SUPPORTED_PROTOCOL_VERSIONS = ("2.0",)


class ProtocolEncoder:
    """
    Serializes simulation messages into UTF-8 JSON lines (newline delimited JSON / NDJSON)
    for zero-dependency, robust high-throughput framing over TCP or WebSocket.
    """
    @staticmethod
    def encode(msg_type: str, payload: Dict[str, Any]) -> bytes:
        data = {
            'type': msg_type,
            'payload': payload
        }
        # Helper to convert numpy arrays to lists
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.floating, np.integer)):
                return obj.item()
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

        json_str = json.dumps(data, default=convert_numpy)
        return (json_str + "\n").encode('utf-8')

    @staticmethod
    def decode(line: Union[str, bytes]) -> Tuple[str, Dict[str, Any]]:
        if isinstance(line, bytes):
            line = line.decode('utf-8')
        line = line.strip()
        if not line:
            return "", {}
        data = json.loads(line)
        return data.get('type', ''), data.get('payload', {})
