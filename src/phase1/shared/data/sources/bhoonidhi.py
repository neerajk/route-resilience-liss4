"""Bhoonidhi (ISRO/NRSC) data-source adapter — Resourcesat LISS-IV (5.8 m).

Bhoonidhi exposes a STAC-compliant search API with JWT authentication and a
product-download endpoint. Endpoint paths, payloads and rate limits below are
taken from the official API specification:
  https://bhoonidhi.nrsc.gov.in/bhoonidhi-api/   (NRSC, accessed 2026-06)

AUTH FLOW (verified from the spec):
  POST {AUTH_URL} {userId, password, grant_type:"password"}
    -> {access_token (JWT, ~1200s), refresh_token, expires_in}
  Use header:  Authorization: Bearer <access_token>
RATE LIMITS (spec): auth 20/hour/IP; search 3/s/IP; download 3 concurrent/IP.

>>> ============================  USER INPUT REQUIRED  ============================
>>> 1. Register for Bhoonidhi API access (email bhoonidhi@nrsc.gov.in) and put
>>>    credentials in a .env file at repo root (gitignored), NOT in code:
>>>        BHOONIDHI_USER=your_user_id
>>>        BHOONIDHI_PASS=your_password
>>> 2. Confirm the exact LISS-IV collection id(s) from the catalog. Run
>>>    list_collections(token) once and copy the id into
>>>    config.yaml -> sources.bhoonidhi.collections. (The id string is NOT
>>>    hardcoded here because it must match the live catalog — no assumptions.)
>>> =============================================================================
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional

AUTH_URL = "https://bhoonidhi-api.nrsc.gov.in/auth/token"
SEARCH_URL = "https://bhoonidhi-api.nrsc.gov.in/data/search"
DOWNLOAD_URL = "https://bhoonidhi-api.nrsc.gov.in/download"
COLLECTIONS_URL = "https://bhoonidhi-api.nrsc.gov.in/data/collections"  # confirm in spec


def _requests():
    import requests  # lazy
    return requests


def get_token(user: Optional[str] = None, password: Optional[str] = None) -> Dict:
    """Obtain a JWT access/refresh token pair.

    Credentials resolve from args, else from env BHOONIDHI_USER / BHOONIDHI_PASS
    (load a .env with python-dotenv in your entrypoint). Returns the full token
    JSON (access_token, refresh_token, expires_in).
    """
    user = user or os.environ.get("BHOONIDHI_USER")
    password = password or os.environ.get("BHOONIDHI_PASS")
    if not user or not password:
        raise RuntimeError(
            "Bhoonidhi credentials missing. Set BHOONIDHI_USER / BHOONIDHI_PASS "
            "in a .env file (see USER INPUT note at top of bhoonidhi.py)."
        )
    requests = _requests()
    r = requests.post(AUTH_URL, json={"userId": user, "password": password,
                                      "grant_type": "password"}, timeout=60)
    r.raise_for_status()
    return r.json()


def list_collections(access_token: str) -> List[Dict]:
    """List available STAC collections (use once to find the LISS-IV id)."""
    requests = _requests()
    r = requests.get(COLLECTIONS_URL,
                     headers={"Authorization": f"Bearer {access_token}"}, timeout=60)
    r.raise_for_status()
    return r.json()


def search(access_token: str, collections: List[str], bbox: List[float],
           datetime_range: str, limit: int = 100) -> Dict:
    """STAC search. `datetime_range` is RFC3339, e.g.
    '2023-11-01T00:00:00Z/2024-03-31T23:59:59Z'. Returns a GeoJSON
    FeatureCollection of STAC items.
    """
    requests = _requests()
    body = {"collections": collections, "bbox": bbox,
            "datetime": datetime_range, "limit": min(limit, 500)}
    r = requests.post(SEARCH_URL, json=body,
                      headers={"Authorization": f"Bearer {access_token}"}, timeout=120)
    r.raise_for_status()
    return r.json()


def download_item(access_token: str, item_id: str, collection: str,
                  out_dir: str) -> Path:
    """Download one product. Only items with property Online == 'Y' are served
    immediately (spec); others must be ordered. Honour the 3-concurrent limit.
    """
    requests = _requests()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    params = {"id": item_id, "collection": collection}
    with requests.get(DOWNLOAD_URL, params=params,
                      headers={"Authorization": f"Bearer {access_token}"},
                      stream=True, timeout=600) as r:
        r.raise_for_status()
        out = out_dir / f"{item_id}.zip"
        with open(out, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
    return out


def fetch_liss4(cfg: dict) -> List[Path]:
    """High-level: authenticate -> search AOI/date -> download online items.

    Reads cfg['sources']['bhoonidhi'] (collections) and cfg['preprocess']
    (aoi_bbox, date_range, paths). Returns paths to downloaded products.
    """
    b = cfg["sources"]["bhoonidhi"]
    pp = cfg["preprocess"]
    tok = get_token()["access_token"]
    fc = search(tok, collections=b["collections"], bbox=pp["aoi_bbox"],
                datetime_range=pp["liss4_date_range"], limit=b.get("limit", 100))
    out_dir = Path(pp["paths"]["raw"]) / "liss4"
    paths: List[Path] = []
    for feat in fc.get("features", []):
        if feat.get("properties", {}).get("Online", "N") == "Y":
            paths.append(download_item(tok, feat["id"], b["collections"][0], str(out_dir)))
    return paths
