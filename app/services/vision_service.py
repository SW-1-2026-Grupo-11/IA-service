"""
Análisis visual de frames para proctoring.

Caras  → MediaPipe **FaceLandmarker** (presencia, conteo y POSE de cabeza).
Objetos → MediaPipe **ObjectDetector** (celular).

Reemplaza las cascadas Haar (imprecisas, ~2001) por FaceLandmarker, que da:
- `sin_rostro` / `multiples_rostros` confiables,
- `mirada_fuera_pantalla` por **pose de cabeza real** (yaw/pitch), NO por
  "rostro descentrado" ni por "no veo ojos" (esa regla generaba el celular-fantasma,
  por eso se elimina).

NOTA: los umbrales de pose (_YAW_LIMIT_DEG / _PITCH_LIMIT_DEG) y de celular
(_PHONE_SCORE_MIN) conviene **afinarlos con grabaciones reales**.
"""
from __future__ import annotations

import base64
import logging
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.schemas.alert_schema import AlertaDjango, EvidenciaJSON
from app.schemas.proctoring_schema import FrameRequest

logger = logging.getLogger("vision_service")

# ─── Modelos ─────────────────────────────────────────────────────────────────
_MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
_OBJECT_MODEL_PATH = _MODELS_DIR / "efficientdet_lite0.tflite"
_FACE_MODEL_PATH = _MODELS_DIR / "face_landmarker.task"

# ─── Umbrales (AFINAR con pruebas reales) ────────────────────────────────────
_PHONE_SCORE_MIN = 0.5      # antes 0.35 → subido para menos falsos celulares
_YAW_LIMIT_DEG = 25.0       # giro horizontal de cabeza tolerado (mirar a un lado)
_PITCH_LIMIT_DEG = 20.0     # inclinación vertical tolerada (mirar arriba/abajo)

# ─── Detectores (carga lazy) ─────────────────────────────────────────────────
_object_detector: Any = None
_object_loaded = False
_face_landmarker: Any = None
_face_loaded = False


def _get_object_detector() -> Any:
    global _object_detector, _object_loaded
    if _object_loaded:
        return _object_detector
    _object_loaded = True
    try:
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision

        if not _OBJECT_MODEL_PATH.exists():
            logger.warning("Modelo de objetos no encontrado: %s", _OBJECT_MODEL_PATH)
            return None
        base = mp_python.BaseOptions(model_asset_path=str(_OBJECT_MODEL_PATH))
        opts = mp_vision.ObjectDetectorOptions(
            base_options=base, score_threshold=_PHONE_SCORE_MIN, max_results=5
        )
        _object_detector = mp_vision.ObjectDetector.create_from_options(opts)
        logger.info("ObjectDetector cargado: %s", _OBJECT_MODEL_PATH)
    except Exception as exc:  # noqa: BLE001
        logger.warning("No se pudo cargar ObjectDetector: %s", exc)
        _object_detector = None
    return _object_detector


def _get_face_landmarker() -> Any:
    global _face_landmarker, _face_loaded
    if _face_loaded:
        return _face_landmarker
    _face_loaded = True
    try:
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision

        if not _FACE_MODEL_PATH.exists():
            logger.warning("Modelo FaceLandmarker no encontrado: %s", _FACE_MODEL_PATH)
            return None
        base = mp_python.BaseOptions(model_asset_path=str(_FACE_MODEL_PATH))
        opts = mp_vision.FaceLandmarkerOptions(
            base_options=base,
            num_faces=3,
            output_facial_transformation_matrixes=True,
            output_face_blendshapes=False,
        )
        _face_landmarker = mp_vision.FaceLandmarker.create_from_options(opts)
        logger.info("FaceLandmarker cargado: %s", _FACE_MODEL_PATH)
    except Exception as exc:  # noqa: BLE001
        logger.warning("No se pudo cargar FaceLandmarker: %s", exc)
        _face_landmarker = None
    return _face_landmarker


# ─── Helpers ─────────────────────────────────────────────────────────────────
def _decode_frame(encoded: str) -> np.ndarray | None:
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
    modelo: str = "mediapipe_facelandmarker",
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
            modelo=modelo,
            session_id=getattr(request, "session_id", None),
        ),
        timestamp_alerta=request.timestamp,
    )


def _yaw_pitch_deg(matrix: Any) -> tuple[float, float]:
    """
    yaw (giro horizontal) y pitch (inclinación vertical) en grados, a partir de la
    matriz 4x4 de transformación facial de MediaPipe. Se usan en valor ABSOLUTO,
    así que el signo/convención exacto no afecta (mirar a izq o der, arriba o abajo,
    cuenta igual como "fuera de pantalla").
    """
    R = np.asarray(matrix, dtype=float)[:3, :3]
    sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    if sy > 1e-6:
        pitch = math.degrees(math.atan2(R[2, 1], R[2, 2]))
        yaw = math.degrees(math.atan2(-R[2, 0], sy))
    else:
        pitch = math.degrees(math.atan2(-R[1, 2], R[1, 1]))
        yaw = math.degrees(math.atan2(-R[2, 0], sy))
    return yaw, pitch


# ─── Detección de teléfono (MediaPipe ObjectDetector) ────────────────────────
def _detect_phone(img: np.ndarray, request: FrameRequest) -> AlertaDjango | None:
    detector = _get_object_detector()
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
                    logger.info("Teléfono detectado (%.0f%%)", score * 100)
                    return _alert(
                        request,
                        "uso_de_celular",
                        "alta",
                        "El participante tiene un teléfono celular visible en la cámara.",
                        confianza=round(float(score) * 100, 1),
                        modelo="mediapipe_efficientdet",
                    )
    except Exception as exc:
        logger.warning("Error en detección de teléfono: %s", exc)
    return None


# ─── Análisis de rostro (MediaPipe FaceLandmarker) ───────────────────────────
def _analyze_faces(img: np.ndarray, request: FrameRequest) -> AlertaDjango | None:
    detector = _get_face_landmarker()
    if detector is None:
        # Sin modelo de caras NO analizamos (no falseamos un "sin rostro").
        return None
    try:
        import mediapipe as mp

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = detector.detect(mp_image)
    except Exception as exc:
        logger.warning("Error en análisis de rostro: %s", exc)
        return None

    n = len(result.face_landmarks)
    if n == 0:
        return _alert(
            request,
            "sin_rostro",
            "alta",
            "El participante no se encuentra en el encuadre de la cámara.",
        )
    if n > 1:
        return _alert(
            request,
            "multiples_rostros",
            "alta",
            f"Se detectaron {n} personas en la cámara.",
        )

    # Un solo rostro: ¿está mirando la pantalla? (pose de cabeza real)
    mats = getattr(result, "facial_transformation_matrixes", None)
    if mats:
        yaw, pitch = _yaw_pitch_deg(mats[0])
        if abs(yaw) > _YAW_LIMIT_DEG or abs(pitch) > _PITCH_LIMIT_DEG:
            logger.info("mirada_fuera (yaw=%.0f pitch=%.0f)", yaw, pitch)
            return _alert(
                request,
                "mirada_fuera_pantalla",
                "media",
                "El participante desvió la mirada de la pantalla.",
            )
    return None


# ─── Punto de entrada ────────────────────────────────────────────────────────
def analyze_frame(request: FrameRequest) -> AlertaDjango | None:
    """Analiza un frame y retorna una AlertaDjango si detecta una irregularidad."""
    try:
        img = _decode_frame(request.frame)
        if img is None:
            logger.warning(
                "Frame inválido para entrevista=%s participante=%s",
                request.entrevista_id,
                request.participante_id,
            )
            return None

        phone = _detect_phone(img, request)
        if phone:
            return phone
        return _analyze_faces(img, request)
    except Exception as exc:
        logger.error("Error inesperado procesando el frame: %s", exc, exc_info=True)
        return None
