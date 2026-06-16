import httpx
import logging

from app.core.config import settings
from app.schemas.alert_schema import AlertaDjango

logger = logging.getLogger("django_client")


async def enviar_alerta_django(alerta: AlertaDjango) -> bool:
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.post(
                settings.DJANGO_ALERTS_URL,
                json=alerta.model_dump(),
                headers={"X-API-Key": settings.DJANGO_API_KEY},
            )
        if response.status_code in (200, 201):
            logger.info("Alerta enviada a Django: %s", alerta.tipo_alerta)
            return True
        logger.warning("Django respondió %d: %s", response.status_code, response.text)
        return False
    except Exception as exc:
        logger.error("Error enviando alerta a Django: %s", exc)
        return False
