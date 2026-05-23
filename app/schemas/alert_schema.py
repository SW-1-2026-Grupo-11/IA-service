"""
Esquemas Pydantic para alertas enviadas al backend Django.
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class EvidenciaJSON(BaseModel):
    modo: str
    confianza: float
    modelo: str
    session_id: Optional[str] = None


class AlertaDjango(BaseModel):
    id_entrevista: str
    id_participante: str
    tipo_alerta: str
    severidad: str
    origen: str = "fastapi"
    descripcion: str
    evidencia_json: EvidenciaJSON
    timestamp_alerta: str
