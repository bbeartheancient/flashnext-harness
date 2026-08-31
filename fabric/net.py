"""Netgear Orbi SOAP client — attached-device map for Security (D-lane).

Verified against RBR50 fw V2.7.5.6NA: JWT login via DeviceConfig:1#SOAPLogin
with a <Password> field (NOT NewPassword), then DeviceInfo:1#GetAttachDevice2
for per-device XML. Password lives in fabric/data/media-keys.env
(NETGEAR_PASS); never logged or returned.
"""

from __future__ import annotations

import os
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET

from . import secrets

_UA = "woodfire-fabric"
_SOAP = "/soap/server_sa/"
_ENVELOPE = ('<?xml version="1.0"?><SOAP-ENV:Envelope '
             'xmlns:SOAP-ENV="http://schemas.xmlsoap.org/soap/envelope/">'
             "<SOAP-ENV:Body><{method}></{method}></SOAP-ENV:Body>"
             "</SOAP-ENV:Envelope>")
_LOGIN = ('<?xml version="1.0"?><SOAP-ENV:Envelope '
          'xmlns:SOAP-ENV="http://schemas.xmlsoap.org/soap/envelope/">'
          "<SOAP-ENV:Body><SOAPLogin><Username>{user}</Username>"
          "<Password>{pw}</Password></SOAPLogin></SOAP-ENV:Body>"
          "</SOAP-ENV:Envelope>")

_COOKIE: str | None = None


def _host() -> str:
    return os.environ.get("FABRIC_ROUTER_URL", "http://127.0.0.1").rstrip("/")


def _pass() -> str:
    env = os.environ.get("NETGEAR_PASS", "")
    if env.strip():
        return env.strip()
    return secrets.load().get("NETGEAR_PASS", "")


def _post(action: str, body: str,
          cookie: str | None = None) -> tuple[int, str, str | None]:
    headers = {"Content-Type": "text/xml",
               "SOAPAction": f'"urn:NETGEAR-ROUTER:service:{action}"',
               "User-Agent": _UA}
    if cookie:
        headers["Cookie"] = cookie
    req = urllib.request.Request(_host() + _SOAP,
                                 data=body.encode(), headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            set_cookie = resp.headers.get("Set-Cookie")
            return resp.status, resp.read().decode(errors="replace"), set_cookie
    except urllib.error.HTTPError as e:
        return e.code, "", None
    except Exception as e:  # noqa: BLE001
        return 0, str(e), None


def _login() -> str | None:
    """JWT login -> cookie value (jwt_local=...) or None."""
    pw = _pass()
    if not pw:
        return None
    body = _LOGIN.format(user="admin", pw=pw)
    _, _, set_cookie = _post("DeviceConfig:1#SOAPLogin", body)
    global _COOKIE
    _COOKIE = set_cookie
    return _COOKIE


def _call(method: str, service: str = "DeviceInfo:1") -> str | None:
    """Call a SOAP method with cached session; one silent re-login."""
    global _COOKIE
    for attempt in range(2):
        status, text, _ = _post(f"{service}#{method}",
                                _ENVELOPE.format(method=method),
                                cookie=_COOKIE)
        if status == 200 and "ResponseCode>000" in text:
            return text
        if attempt == 0:
            _login()
        else:
            return None
    return None


def router_status() -> dict:
    """Router identity + firmware (GetInfo)."""
    xml_text = _call("GetInfo")
    if not xml_text:
        return {"error": "orbi unreachable or auth failed"}
    root = ET.fromstring(xml_text)
    out = {}
    for el in root.iter():
        tag = el.tag.split("}")[-1]
        if tag in ("ModelName", "Firmwareversion", "DeviceName",
                   "SerialNumber", "InternetConnectionStatus") \
                and el.text and el.text.strip():
            out[tag] = el.text.strip()
    return out


def net_devices() -> dict:
    """Attached-device list from the Orbi (name/ip/mac/type/signal)."""
    xml_text = _call("GetAttachDevice2")
    if not xml_text:
        return {"error": "orbi unreachable or auth failed"}
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as e:
        return {"error": f"bad router XML: {e}"}
    devices = []
    for d in root.iter("Device"):
        get = lambda k: ((d.find(k).text or "").strip()
                         if d.find(k) is not None else "")
        ip = get("IP")
        if not ip:
            continue
        devices.append({
            "ip": ip,
            "name": get("Name")[:40],
            "mac": get("MAC"),
            "conn": get("ConnectionType")[:12],
            "signal": int(get("SignalStrength") or 0),
            "link": get("Linkspeed")[:16],
            "model": get("DeviceModel")[:24],
            "ssid": get("SSID")[:20],
            "blocked": get("AllowOrBlock") == "Block",
        })
    wifi = [d for d in devices if d["conn"].endswith("GHz")]
    return {"host": _host(), "total": len(devices),
            "wired": len(devices) - len(wifi),
            "wireless": len(wifi),
            "devices": sorted(devices, key=lambda d: d["ip"])}
