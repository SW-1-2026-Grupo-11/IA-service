"""
Router de proctoring para el servicio de IA.

Procesa frames de cámara, cambios de ventana y eventos de Jitsi,
genera alertas y las envía al backend Django. Incluye el session_id
dentro del campo evidencia_json para permitir el aislamiento por sesión.
"""
from __future__ import annotations

from fastapi import APIRouter

from app.schemas.alert_schema import AlertaDjango, EvidenciaJSON
from app.schemas.proctoring_schema import (
    FrameRequest,
    JitsiEventRequest,
    WindowChangeRequest,
)
from app.services.debounce import reset as reset_debounce, should_emit
from app.services.django_client import enviar_alerta_django
from app.services.vision_service import analyze_frame

router = APIRouter()


@router.post("/ia/proctoring/frame")
async def process_frame(request: FrameRequest):
    """Analiza un frame de video y genera alertas si es necesario."""
    alerta = analyze_frame(request)

    # Clave del episodio por sesión (o entrevista+participante si no hay session_id).
    key = request.session_id or f"{request.entrevista_id}:{request.participante_id}"

    if not alerta:
        # Frame normal → se cierra el episodio en curso (si lo había).
        reset_debounce(key)
        return {"success": True, "alerta": False, "mensaje": "Frame analizado sin alerta"}

    # Propagate session_id if present
    if request.session_id:
        alerta.evidencia_json.session_id = request.session_id

    # Debounce: una condición continua (p. ej. "sin rostro") cuenta como UN episodio,
    # no una alerta por frame. Evita inflar el riesgo con cientos de alertas idénticas.
    if not should_emit(key, alerta.tipo_alerta):
        return {
            "success": True,
            "alerta": False,
            "mensaje": f"Alerta {alerta.tipo_alerta} agrupada (episodio en curso)",
        }

    enviado = await enviar_alerta_django(alerta)
    if enviado:
        return {
            "success": True,
            "alerta": True,
            "tipo_alerta": alerta.tipo_alerta,
            "severidad": alerta.severidad,
            "mensaje": f"Alerta {alerta.tipo_alerta} detectada y enviada",
        }
    return {"success": False, "message": "Alerta generada, pero no se pudo enviar a Django"}


@router.post("/ia/proctoring/window-change")
async def process_window_change(request: WindowChangeRequest):
    """Registra un cambio de ventana del participante."""
    alerta = AlertaDjango(
        id_entrevista=request.entrevista_id,
        id_participante=request.participante_id,
        tipo_alerta="cambio_ventana",
        severidad="alta",
        descripcion="El participante cambió a otra ventana o pestaña durante la prueba.",
        evidencia_json=EvidenciaJSON(
            modo="real",
            confianza=100.0,
            modelo="browser_event",
            session_id=request.session_id,
        ),
        timestamp_alerta=request.timestamp,
    )

    enviado = await enviar_alerta_django(alerta)
    if enviado:
        return {
            "success": True,
            "alerta": True,
            "tipo_alerta": alerta.tipo_alerta,
            "severidad": alerta.severidad,
            "mensaje": "Alerta de cambio de ventana registrada",
        }
    return {"success": False, "message": "Alerta de cambio de ventana generada, pero no se pudo enviar a Django"}


@router.post("/ia/proctoring/jitsi-event")
async def process_jitsi_event(request: JitsiEventRequest):
    """Procesa eventos de Jitsi Meet y genera alertas para los eventos críticos."""
    eventos_alerta: dict[str, str] = {
        "camara_apagada": "media",
        "camara_no_disponible": "alta",
        "pantalla_compartida": "alta",
        "participante_salio": "media",
    }

    if request.tipo_evento not in eventos_alerta:
        return {
            "success": True,
            "alerta": False,
            "mensaje": f"Evento {request.tipo_evento} registrado sin alerta",
        }

    alerta = AlertaDjango(
        id_entrevista=request.entrevista_id,
        id_participante=request.participante_id,
        tipo_alerta=request.tipo_evento,
        severidad=eventos_alerta[request.tipo_evento],
        descripcion=f"Se detectó un evento en Jitsi: {request.tipo_evento}",
        evidencia_json=EvidenciaJSON(
            modo="real",
            confianza=100.0,
            modelo="jitsi_event",
            session_id=request.session_id,
        ),
        timestamp_alerta=request.timestamp,
    )

    enviado = await enviar_alerta_django(alerta)
    if enviado:
        return {
            "success": True,
            "alerta": True,
            "tipo_alerta": alerta.tipo_alerta,
            "severidad": alerta.severidad,
            "mensaje": f"Alerta {request.tipo_evento} registrada",
        }
    return {
        "success": False,
        "message": f"Alerta {request.tipo_evento} generada, pero no se pudo enviar a Django",
    }
