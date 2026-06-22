"""
Agrupación temporal (debounce) de alertas de proctoring.

La IA analiza ~1 frame cada 2 segundos. Sin esto, una condición continua (p. ej.
"sin rostro" durante 10 minutos) generaría cientos de alertas idénticas —una por
frame— inflando el riesgo del informe (el caso real: 410 alertas "sin rostro").

Con esto, una misma condición continua cuenta como UN episodio:
  - se emite al EMPEZAR el episodio (transición desde "ok" u otro tipo),
  - si persiste, se RE-EMITE a lo sumo cada REALERT_SECONDS (para que la duración
    quede reflejada con varios timestamps, sin spamear por frame),
  - al volver a la normalidad (frame sin alerta) el episodio se CIERRA, y la
    próxima ocurrencia vuelve a ser un episodio nuevo.

El estado vive en memoria por sesión. Es seguro porque el ai-service corre con un
solo worker (uvicorn sin --workers); si algún día se escala a varios workers, esto
debería moverse a un store compartido (Redis) o al backend.
"""
from __future__ import annotations

import time

# Cada cuánto se re-emite una alerta del MISMO tipo que sigue activa (segundos).
REALERT_SECONDS = 30.0

# Poda defensiva: si quedan muchas sesiones colgadas en memoria, se limpian las viejas.
_MAX_ENTRIES = 2000
_STALE_SECONDS = 3600.0

# clave de sesión -> {"tipo": <tipo_alerta>, "last_post": <epoch segundos>}
_state: dict[str, dict] = {}


def _prune(now: float) -> None:
    if len(_state) <= _MAX_ENTRIES:
        return
    for k in [k for k, v in _state.items() if now - v["last_post"] > _STALE_SECONDS]:
        _state.pop(k, None)


def should_emit(key: str, tipo: str, now: float | None = None) -> bool:
    """
    True si esta alerta debe ENVIARSE a Django:
      - episodio nuevo (no había nada, o cambió el tipo de alerta), o
      - el mismo tipo sigue activo pero ya pasó REALERT_SECONDS (re-alerta).
    False si es la misma condición continua dentro de la ventana → se agrupa.
    """
    now = time.time() if now is None else now
    st = _state.get(key)
    if st is None or st["tipo"] != tipo:
        _state[key] = {"tipo": tipo, "last_post": now}
        _prune(now)
        return True
    if now - st["last_post"] >= REALERT_SECONDS:
        st["last_post"] = now
        return True
    return False


def reset(key: str) -> None:
    """Cierra el episodio de una sesión (frame sin alerta = volvió a la normalidad)."""
    _state.pop(key, None)
