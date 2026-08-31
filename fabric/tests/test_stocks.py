"""stocks.py + firefly net_worth vs fakes."""

import json
from unittest.mock import patch

from fabric import stocks

CSV = ("Symbol,Date,Time,Open,High,Low,Close\n"
       "aapl.us,2026-08-25,17:00,200.1,205.0,199.5,204.25\n"
       "msft.us,2026-08-25,17:00,410.0,415.2,409.0,413.7\n")


def test_net_worth_needs_token(monkeypatch):
    monkeypatch.delenv("FIREFLY_TOKEN", raising=False)
    with patch.object(stocks.secrets, "load", return_value={}):
        out = stocks.net_worth()
    assert "FIREFLY_TOKEN" in out["error"]


def test_net_worth_sums_asset_accounts():
    page = {"data": [
        {"attributes": {"name": "Checking", "current_balance": "1250.40",
                        "currency_code": "USD"}},
        {"attributes": {"name": "Savings", "current_balance": "-50",
                        "currency_code": "USD"}}],
        "meta": {"total_pages": 1}}

    class R:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return json.dumps(page).encode()

    def fake_open(req, timeout=0):
        return R()

    with patch.object(stocks.urllib.request, "urlopen", fake_open), \
            patch.object(stocks, "_token", return_value="t"):
        out = stocks.net_worth()
    assert out["ok"] is True
    assert out["net_worth"] == 1200.4
