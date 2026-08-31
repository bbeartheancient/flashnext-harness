"""Stock quotes (stooq, keyless) + Firefly III read-only net worth."""

from __future__ import annotations

import os
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET  # noqa: F401  (reserved for rss parity)

from . import secrets

_UA = "woodfire-fabric"


def _get(url: str, timeout: float = 8.0,
         headers: dict | None = None) -> tuple[int, str]:
    req = urllib.request.Request(url, headers={"User-Agent": _UA,
                                               **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read().decode(errors="replace")
    except Exception as e:  # noqa: BLE001
        return 0, str(e)


# ------------------------------------------------------- yahoo quotes
def quotes(symbols: str) -> dict:
    """Keyless quotes via Yahoo finance chart API. symbols: 'AAPL,MSFT'."""
    syms = [s.strip().upper() for s in (symbols or "").split(",")
            if s.strip()][:10]
    if not syms:
        return {"error": "no symbols given"}
    out = {}
    missing = []
    for sym in syms:
        url = (f"https://query1.finance.yahoo.com/v8/finance/chart/"
               f"{urllib.parse.quote(sym)}?range=5d&interval=1d")
        code, text = _get(url, timeout=8)
        if code != 200:
            missing.append(sym)
            continue
        try:
            import json as _json

            data = _json.loads(text)
            meta = data["chart"]["result"][0]["meta"]
            price = float(meta["regularMarketPrice"])
            prev = float(meta.get("chartPreviousClose") or
                         meta.get("previousClose") or price)
            out[sym] = {
                "price": round(price, 2),
                "change_pct": round((price - prev) / prev * 100, 2)
                if prev else 0.0,
                "currency": meta.get("currency", "USD"),
                "name": meta.get("shortName") or sym,
            }
        except Exception:  # noqa: BLE001
            missing.append(sym)
    if not out and missing:
        return {"error": "quote source unavailable", "missing": missing}
    return {"quotes": out, "missing": sorted(missing), "source": "yahoo"}


# ------------------------------------------------------------ firefly iii
def _base() -> str:
    return os.environ.get(
        "FABRIC_FIREFLY_URL",
        "https://finance.woodfireindustries.com").rstrip("/")


def _token() -> str:
    env = os.environ.get("FIREFLY_TOKEN", "")
    if env.strip():
        return env.strip()
    return secrets.load().get("FIREFLY_TOKEN", "")


def net_worth(page_limit: int = 5) -> dict:
    """Read-only asset accounts -> total balance. Needs FIREFLY_TOKEN."""
    tok = _token()
    if not tok:
        return {"error": "no FIREFLY_TOKEN in media-keys.env — create a "
                         "personal access token in Firefly (Options > "
                         "Profile > OAuth) and store it as FIREFLY_TOKEN"}
    headers = {"Authorization": f"Bearer {tok}", "Accept": "application/json"}
    accounts = []
    page = 1
    for _ in range(max(1, page_limit)):
        code, text = _get(f"{_base()}/api/v1/accounts?page={page}"
                          "&type=asset", headers=headers, timeout=12)
        if code != 200:
            return {"error": f"firefly HTTP {code} (token valid? "
                             "server up?)"}
        import json as _json

        data = _json.loads(text)
        for row in data.get("data") or []:
            attrs = row.get("attributes") or {}
            try:
                bal = float(attrs.get("current_balance") or 0)
            except (TypeError, ValueError):
                continue
            accounts.append({"name": attrs.get("name"),
                             "balance": round(bal, 2),
                             "currency": attrs.get("currency_code")})
        meta = data.get("meta") or {}
        if page >= int(meta.get("total_pages") or 1):
            break
        page += 1
    total = round(sum(a["balance"] for a in accounts), 2)
    return {"ok": True, "net_worth": total,
            "currency": (accounts[0]["currency"] if accounts else "USD"),
            "accounts": sorted(accounts,
                               key=lambda a: -abs(a["balance"]))[:20]}
