"""net.py: Orbi SOAP client vs fixture responses (no live router)."""

from unittest.mock import patch

from fabric import net

DEVICE_XML = """<?xml version="1.0"?>
<GetAttachDevice2Response xmlns:m="urn:x">
<NewAttachDevice>
<Device><IP>192.0.2.9</IP><Name>GENERATOR</Name><MAC>AA:BB</MAC>
<ConnectionType>wired</ConnectionType><SignalStrength>100</SignalStrength>
<Linkspeed></Linkspeed><DeviceModel>XPU</DeviceModel><SSID></SSID>
<AllowOrBlock>Allow</AllowOrBlock></Device>
<Device><IP>192.0.2.16</IP><Name>IPHONE</Name><MAC>CC:DD</MAC>
<ConnectionType>5GHz</ConnectionType><SignalStrength>80</SignalStrength>
<Linkspeed>866</Linkspeed><DeviceModel></DeviceModel><SSID>Orbi</SSID>
<AllowOrBlock>Allow</AllowOrBlock></Device>
</NewAttachDevice>
<ResponseCode>000</ResponseCode></GetAttachDevice2Response>"""

INFO_XML = ('<GetInfoResponse><ModelName>RBR50</ModelName>'
            '<Firmwareversion>V2.7.5.6NA</Firmwareversion><ResponseCode>000</ResponseCode></GetInfoResponse>')


class _FakeFlow:
    """Serves login then data; records cookie usage."""

    def __init__(self):
        self.calls = []

    def __call__(self, action, body, cookie=None):
        self.calls.append((action, cookie))
        if "SOAPLogin" in action:
            return 200, "", "jwt_local=tok123"
        if "GetAttachDevice2" in action:
            if cookie != "jwt_local=tok123":
                return 200, '<ResponseCode>401</ResponseCode>', None
            return 200, DEVICE_XML, None
        if "GetInfo" in action:
            return 200, INFO_XML, None
        return 200, '<ResponseCode>404</ResponseCode>', None


def test_net_devices_parses_with_session():
    flow = _FakeFlow()
    with patch.object(net, "_pass", return_value="pw"), \
            patch.object(net, "_post", flow), \
            patch.object(net, "_COOKIE", None):
        out = net.net_devices()
    assert out["total"] == 2 and out["wired"] == 1 and out["wireless"] == 1
    iph = next(d for d in out["devices"] if d["name"] == "IPHONE")
    assert iph["signal"] == 80 and iph["ssid"] == "Orbi"
    # first attempt unauthed -> silent re-login -> retry carries cookie
    actions = [c[0] for c in flow.calls]
    assert "DeviceConfig:1#SOAPLogin" in actions
    assert ("DeviceInfo:1#GetAttachDevice2", "jwt_local=tok123") in flow.calls


def test_router_status_fields():
    with patch.object(net, "_pass", return_value="pw"), \
            patch.object(net, "_post", _FakeFlow()), \
            patch.object(net, "_COOKIE", None):
        out = net.router_status()
    assert out["ModelName"] == "RBR50"
    assert out["Firmwareversion"] == "V2.7.5.6NA"


def test_no_password_is_honest(monkeypatch):
    monkeypatch.delenv("NETGEAR_PASS", raising=False)
    with patch.object(net.secrets, "load", return_value={}):
        out = net.net_devices()
    assert "error" in out
