"""
Persistencia + agrupación (debounce) de alertas de proctoring.

La IA analiza ~1 frame cada 2 segundos. Sin control, una condición momentánea
(mirar 1 seg a pensar) o continua (10 min sin rostro) generaría falsos o cientos
de alertas. Acá se aplican DOS reglas, calibradas para RECLUTAMIENTO (no acusar
falsamente al candidato):

1) PERSISTENCIA: una condición debe MANTENERSE >= PERSIST_SECONDS antes de emitir
   la PRIMERA alerta. Así, mirar un segundo al costado NO dispara nada (era ruido).
2) RE-ALERTA: si la condición sigue, se re-emite a lo sumo cada REALERT_SECONDS,
   para reflejar la duración sin spamear por frame.

Al volver a la normalidad (frame sin alerta) el episodio se CIERRA (reset) y la
próxima ocurrencia arranca de cero.

Estado en memoria por sesión. Seguro porque el ai-service corre con un solo worker.
Solo aplica a las alertas de VISIÓN (frames); los eventos (cambio de pestaña,
cámara apagada, etc.) son discretos y se emiten al instante.
"""
from __future__ import annotations

import time

# Cuánto debe DURAR una condición antes de la PRIMERA alerta (segundos).
PERSIST_SECONDS = 4.0
# Cada cuánto se RE-EMITE una alerta del mismo tipo que sigue activa (segundos).
REALERT_SECONDS = 30.0

# Poda defensiva de sesiones colgadas en memoria.
_MAX_ENTRIES = 2000
_STALE_SECONDS = 3600.0

# clave de sesión -> {"tipo", "started_at", "last_post", "emitted"}
_state: dict[str, dict] = {}


def _prune(now: float) -> None:
    if len(_state) <= _MAX_ENTRIES:
        return
    ref = lambda v: max(v["started_at"], v["last_post"])  # noqa: E731
    for k in [k for k, v in _state.items() if now - ref(v) > _STALE_SECONDS]:
        _state.pop(k, None)


def should_emit(key: str, tipo: str, now: float | None = None) -> bool:
    """
    True si esta alerta debe ENVIARSE a Django:
      - la condición persistió >= PERSIST_SECONDS (primera alerta del episodio), o
      - el mismo tipo sigue activo y ya pasó REALERT_SECONDS (re-alerta).
    False mientras la condición es nueva/momentánea o está dentro de la ventana.
    """
    now = time.time() if now is None else now
    st = _state.get(key)

    # Condición nueva (o cambió el tipo): arranca el cronómetro, todavía NO emite.
    if st is None or st["tipo"] != tipo:
        _state[key] = {"tipo": tipo, "started_at": now, "last_post": 0.0, "emitted": False}
        _prune(now)
        return False

    # Misma condición en curso.
    if not st["emitted"]:
        if now - st["started_at"] >= PERSIST_SECONDS:
            st["emitted"] = True
            st["last_post"] = now
            return True  # primera alerta tras persistir lo suficiente
        return False  # todavía no duró lo necesario (probablemente momentánea)

    # Ya emitió: re-alerta periódica si la condición persiste.
    if now - st["last_post"] >= REALERT_SECONDS:
        st["last_post"] = now
        return True
    return False


def reset(key: str) -> None:
    """Cierra el episodio (frame sin alerta = volvió a la normalidad)."""
    _state.pop(key, None)
