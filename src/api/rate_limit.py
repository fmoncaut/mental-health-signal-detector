"""Instance partagée du rate limiter — importée par main.py et les routers."""

import ipaddress

from fastapi import Request
from slowapi import Limiter
from slowapi.util import get_remote_address

from src.common.config import get_settings


_MAX_XFF_LEN = 512
_MAX_XFF_PARTS = 10


def _extract_client_ip_from_xff(forwarded_for: str) -> str | None:
    """Extract the first valid client IP from X-Forwarded-For.

    In common reverse-proxy setups, the left-most address is the original
    client and subsequent values are proxy hops. We only accept valid IP
    literals and ignore malformed entries.
    """
    raw = forwarded_for.strip()
    if not raw or len(raw) > _MAX_XFF_LEN:
        return None

    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts:
        return None
    if len(parts) > _MAX_XFF_PARTS:
        parts = parts[:_MAX_XFF_PARTS]

    for part in parts:
        try:
            return str(ipaddress.ip_address(part))
        except ValueError:
            continue
    return None


def _get_client_ip(request: Request) -> str:
    """Retourne l'IP réelle du client.

    Lit X-Forwarded-For uniquement si l'app est derrière un reverse proxy de
    confiance (Render load balancer). Prend la PREMIÈRE IP valide de la chaîne
    (IP client d'origine dans le format XFF standard), sinon fallback socket.

    Fallback sur l'IP socket réelle si le header est absent.
    """
    settings = get_settings()
    forwarded_for = request.headers.get("X-Forwarded-For", "").strip()
    if settings.trust_proxy_headers and forwarded_for:
        parsed = _extract_client_ip_from_xff(forwarded_for)
        if parsed:
            return parsed
    return get_remote_address(request)


limiter = Limiter(key_func=_get_client_ip)
