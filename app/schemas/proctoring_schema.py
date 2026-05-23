"""
Esquemas Pydantic para solicitudes de proctoring enviadas al servicio de IA.
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class FrameRequest(BaseModel):
    entrevista_id: str
    participante_id: str
    frame: str
    timestamp: str
    session_id: Optional[str] = None


class WindowChangeRequest(BaseModel):
    entrevista_id: str
    participante_id: str
    evento: str
    timestamp: str
    session_id: Optional[str] = None


class JitsiEventRequest(BaseModel):
    entrevista_id: str
    participante_id: str
    tipo_evento: str
    valor: str
    timestamp: str
    session_id: Optional[str] = None
