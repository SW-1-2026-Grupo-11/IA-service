"""
Análisis visual de frames para proctoring.

Detecta:
- uso_de_celular (ObjectDetector MediaPipe)
- sin_rostro
- multiples_rostros
- mirada_fuera_pantalla (perfil, rostro descentrado o ojos desalineados)
- posible_celular_o_lectura (rostro visible sin ojos — cabeza agachada)
"""

from __future__ import annotations

import base64
import logging
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.schemas.alert_schema import AlertaDjango, EvidenciaJSON
from app.schemas.proctoring_schema import FrameRequest

logger = logging.getLogger("vision_service")

# ─── Constantes ────────────────────────────────────────────────────────────────

_MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "efficientdet_lite0.tflite"
_PHONE_SCORE_MIN = 0.35
_FACE_MIN_SIZE = (50, 50)
_CENTER_MIN = 0.25
_CENTER_MAX = 0.75
_EYE_OFFSET_RATIO = 0.20

# ─── Cascadas Haar (siempre disponibles con OpenCV) ────────────────────────────

_haarcascades = cv2.data.haarcascades  # type: ignore
face_cascade = cv2.CascadeClassifier(
    os.path.join(_haarcascades, "haarcascade_frontalface_default.xml")
)
profile_cascade = cv2.CascadeClassifier(
    os.path.join(_haarcascades, "haarcascade_profileface.xml")
)
eye_cascade = cv2.CascadeClassifier(
    os.path.join(_haarcascades, "haarcascade_eye.xml")
)

# ─── MediaPipe Detector (lazy, opcional) ───────────────────────────────────────

_detector: Any = None
_detector_loaded = False


def _get_detector() -> Any:
    """Carga el detector MediaPipe en el primer uso. Si falla, retorna None."""
    global _detector, _detector_loaded
    if _detector_loaded:
        return _detector
    _detector_loaded = True
    try:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision

        if not _MODEL_PATH.exists():
            logger.warning(
                "Modelo TFLite no encontrado en %s — detección de celular desactivada.",
                _MODEL_PATH,
            )
            return None

        base_options = mp_python.BaseOptions(model_asset_path=str(_MODEL_PATH))
        detector_options = mp_vision.ObjectDetectorOptions(
            base_options=base_options,
            score_threshold=_PHONE_SCORE_MIN,
            max_results=5,
        )
        _detector = mp_vision.ObjectDetector.create_from_options(detector_options)
        logger.info("MediaPipe ObjectDetector cargado correctamente desde %s", _MODEL_PATH)
    except Exception as exc:  # noqa: BLE001
        logger.warning("No se pudo cargar MediaPipe ObjectDetector: %s", exc)
        _detector = None
    return _detector


# ─── Helpers ───────────────────────────────────────────────────────────────────


def _decode_frame(encoded: str) -> np.ndarray | None:
    """Decodifica un frame base64 (con o sin prefijo data:...) a ndarray BGR."""
    try:
        data = encoded.split(",", 1)[-1] if "," in encoded else encoded
        raw = base64.b64decode(data)
        arr = np.frombuffer(raw, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            logger.warning("cv2.imdecode devolvió None — frame inválido o vacío")
        return img
    except Exception as exc:
        logger.warning("Error decodificando frame: %s", exc)
        return None


def _ids(request: FrameRequest) -> tuple[str, str]:
    return request.entrevista_id, request.participante_id


def _alert(
    request: FrameRequest,
    tipo: str,
    severidad: str,
    descripcion: str,
    confianza: float = 85.0,
) -> AlertaDjango:
    entrevista_id, participante_id = _ids(request)
    return AlertaDjango(
        id_entrevista=entrevista_id,
        id_participante=participante_id,
        tipo_alerta=tipo,
        severidad=severidad,
        descripcion=descripcion,
        evidencia_json=EvidenciaJSON(
            modo="real",
            confianza=confianza,
            modelo="opencv_haarcascade",
            session_id=getattr(request, "session_id", None),
        ),
        timestamp_alerta=request.timestamp,
    )


# ─── Detección de teléfono ─────────────────────────────────────────────────────


def _detect_phone(img: np.ndarray, request: FrameRequest) -> AlertaDjango | None:
    detector = _get_detector()
    if detector is None:
        return None

    try:
        import mediapipe as mp

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = detector.detect(mp_image)

        for det in result.detections:
            for cat in det.categories:
                name = (cat.category_name or "").lower()
                if "cell phone" in name or "phone" in name:
                    score = cat.score or 0.0
                    if score < _PHONE_SCORE_MIN:
                        continue
                    logger.info(
                        "Teléfono detectado con confianza %.1f%%", score * 100
                    )
                    entrevista_id, participante_id = _ids(request)
                    return AlertaDjango(
                        id_entrevista=entrevista_id,
                        id_participante=participante_id,
                        tipo_alerta="uso_de_celular",
                        severidad="alta",
                        descripcion="El participante tiene un teléfono celular visible en la cámara.",
                        evidencia_json=EvidenciaJSON(
                            modo="real",
                            confianza=round(float(score) * 100, 1),
                            modelo="mediapipe_efficientdet",
                            session_id=getattr(request, "session_id", None),
                        ),
                        timestamp_alerta=request.timestamp,
                    )
    except Exception as exc:
        logger.warning("Error en detección de teléfono: %s", exc)

    return None


# ─── Análisis de rostros ───────────────────────────────────────────────────────


def _largest_face(faces: Any) -> tuple[int, int, int, int] | None:
    if len(faces) == 0:
        return None
    areas = [w * h for (_, _, w, h) in faces]
    idx = int(np.argmax(areas))
    x, y, w, h = faces[idx]
    return int(x), int(y), int(w), int(h)


def _analyze_faces(gray: np.ndarray, img_w: int, request: FrameRequest) -> AlertaDjango | None:
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=_FACE_MIN_SIZE
    )
    profiles = profile_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=_FACE_MIN_SIZE
    )
    num_faces = len(faces)
    logger.debug(
        "Rostros frontales: %d | Perfiles: %d | Ancho imagen: %dpx",
        num_faces, len(profiles), img_w,
    )

    if num_faces == 0:
        if len(profiles) > 0:
            logger.info("Alerta: mirada_fuera_pantalla (perfil detectado)")
            return _alert(
                request,
                "mirada_fuera_pantalla",
                "media",
                "El participante desvió la mirada de la pantalla (perfil visible, sin rostro frontal).",
            )
        logger.info("Alerta: sin_rostro")
        return _alert(
            request,
            "sin_rostro",
            "alta",
            "El participante no se encuentra en el encuadre de la cámara.",
        )

    if num_faces > 1:
        logger.info("Alerta: multiples_rostros (%d)", num_faces)
        return _alert(
            request,
            "multiples_rostros",
            "alta",
            f"Se detectaron {num_faces} personas en la cámara.",
        )

    face = _largest_face(faces)
    if face is None:
        return None

    x, y, w, h = face
    face_cx = x + w / 2
    if face_cx < img_w * _CENTER_MIN or face_cx > img_w * _CENTER_MAX:
        logger.info("Alerta: mirada_fuera_pantalla (rostro descentrado)")
        return _alert(
            request,
            "mirada_fuera_pantalla",
            "media",
            "El participante desvió la mirada de la pantalla (rostro fuera del centro).",
        )

    roi = gray[y : y + h, x : x + w]
    eyes = eye_cascade.detectMultiScale(roi, scaleFactor=1.1, minNeighbors=6, minSize=(20, 20))

    if len(eyes) == 0:
        logger.info("Alerta: posible_celular_o_lectura (sin ojos detectados)")
        return _alert(
            request,
            "posible_celular_o_lectura",
            "alta",
            "El participante tiene el rostro agachado, posiblemente leyendo o usando el celular fuera de cámara.",
        )

    if len(eyes) >= 2:
        eyes_sorted = sorted(eyes, key=lambda e: e[0])
        ex1 = eyes_sorted[0][0] + eyes_sorted[0][2] / 2
        ex2 = eyes_sorted[-1][0] + eyes_sorted[-1][2] / 2
        eye_mid = (ex1 + ex2) / 2
        face_mid = w / 2
        if abs(eye_mid - face_mid) > w * _EYE_OFFSET_RATIO:
            logger.info("Alerta: mirada_fuera_pantalla (ojos descentrados)")
            return _alert(
                request,
                "mirada_fuera_pantalla",
                "media",
                "El participante desvió la mirada de la pantalla (mirando a un costado).",
            )

    logger.debug("Frame OK — participante visible y atento")
    return None


# ─── Punto de entrada ──────────────────────────────────────────────────────────


def analyze_frame(request: FrameRequest) -> AlertaDjango | None:
    """Analiza un frame y retorna una AlertaDjango si se detecta una irregularidad."""
    try:
        img = _decode_frame(request.frame)
        if img is None:
            logger.warning(
                "Frame inválido para entrevista=%s participante=%s",
                request.entrevista_id, request.participante_id,
            )
            return None

        h, w = img.shape[:2]
        logger.debug(
            "Analizando frame %dx%d — entrevista=%s participante=%s session=%s",
            w, h, request.entrevista_id, request.participante_id, request.session_id,
        )

        phone = _detect_phone(img, request)
        if phone:
            return phone

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return _analyze_faces(gray, w, request)

    except Exception as exc:
        logger.error("Error inesperado procesando el frame: %s", exc, exc_info=True)
        return None
