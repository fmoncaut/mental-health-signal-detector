"""Instance partagée du rate limiter — importée par main.py et les routers."""

from fastapi import Request
from slowapi import Limiter
from slowapi.util import get_remote_address


def _get_client_ip(request: Request) -> str:
    """Retourne l'IP réelle du client.

    Lit X-Forwarded-For uniquement si l'app est derrière un reverse proxy de
    confiance (Render load balancer).  Prend la DERNIÈRE IP de la chaîne :
    c'est celle ajoutée par le proxy de confiance et qui reflète l'IP réelle.
    La première IP est contrôlée par le client et peut être forgée.

    Fallback sur l'IP socket réelle si le header est absent.
    """
    forwarded_for = request.headers.get("X-Forwarded-For", "").strip()
    if forwarded_for:
        return forwarded_for.split(",")[-1].strip()
    return get_remote_address(request)


limiter = Limiter(key_func=_get_client_ip)
