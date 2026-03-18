"""Instance partagée du rate limiter — importée par main.py et les routers."""

from fastapi import Request
from slowapi import Limiter
from slowapi.util import get_remote_address


def _get_client_ip(request: Request) -> str:
    """Retourne l'IP réelle du client.

    Lit X-Forwarded-For uniquement si l'app est derrière un reverse proxy de
    confiance configuré avec `proxy_set_header X-Forwarded-For $remote_addr`.
    Prend la première IP de la chaîne (la plus proche du client).
    Fallback sur l'IP socket réelle si l'header est absent.
    """
    forwarded_for = request.headers.get("X-Forwarded-For", "").strip()
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    return get_remote_address(request)


limiter = Limiter(key_func=_get_client_ip)
