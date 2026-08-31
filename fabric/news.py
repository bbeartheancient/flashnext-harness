"""News feed — RSS fetch + geocode + sqlite event store (D16 clean-room).

Feeds are configurable via FABRIC_NEWS_FEEDS (comma-separated RSS URLs;
a small default set ships). Items are stored in the fabric sqlite
(`news` table) with lat/lon resolved through geo.geocode when a place is
detectable in the title; unplaced items keep coords None.
"""

from __future__ import annotations

import html as _html
import os
import re
import sqlite3
import threading
import time
import urllib.request
import xml.etree.ElementTree as ET

_SCHEMA = """
CREATE TABLE IF NOT EXISTS news (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    first_seen REAL NOT NULL,
    published TEXT,
    source TEXT,
    title TEXT NOT NULL,
    link TEXT,
    place TEXT,
    lat REAL, lon REAL
);
CREATE INDEX IF NOT EXISTS idx_news_seen ON news(first_seen DESC);
"""

_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "data", "news.db")
_conn_holder: dict = {"c": None}
_lock = threading.Lock()
_DEFAULT_FEEDS = [
    "https://feeds.bbci.co.uk/news/world/rss.xml",
    "https://rss.nytimes.com/services/xml/rss/nyt/World.xml",
]
_PLACE_RE = re.compile(
    r"\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){0,2})\b")


def _dbconn():
    """Own sqlite file — never contend with the live service's fabric.db."""
    c = _conn_holder["c"]
    if c is None:
        os.makedirs(os.path.dirname(_DB_PATH), exist_ok=True)
        c = sqlite3.connect(_DB_PATH, timeout=30, check_same_thread=False)
        _conn_holder["c"] = c
    c.executescript(_SCHEMA)
    return c


def _feeds() -> list[str]:
    raw = os.environ.get("FABRIC_NEWS_FEEDS", "")
    feeds = [f.strip() for f in raw.split(",") if f.strip()]
    return feeds or list(_DEFAULT_FEEDS)


def _fetch(url: str, timeout: float = 8.0) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "woodfire-fabric"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode(errors="replace")


def _parse_feed(xml_text: str) -> list[dict]:
    out = []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return out
    for item in root.iter("item"):
        title = (item.findtext("title") or "").strip()
        if not title:
            continue
        out.append({
            "title": _html.unescape(title)[:220],
            "link": (item.findtext("link") or "").strip(),
            "published": (item.findtext("pubDate") or "").strip()[:40],
            "source": "",
        })
    return out


_GEO_CACHE: dict[str, tuple] = {}
_NEG_TTL_S = 900.0


def _geocode_place(title: str) -> tuple[str, float | None, float | None]:
    from .geo import geocode_fast as _geo_lookup

    # skip common news stopwords that look like proper nouns
    stop = {"The", "This", "That", "These", "Those", "A", "An", "It",
            "Its", "As", "At", "On", "In", "He", "She", "They", "We",
            "What", "How", "Who", "Why", "When", "Now", "Then", "Here",
            "There", "President", "Minister", "Court", "University",
            "Report", "Live", "Update", "Analysis", "Opinion", "Video",
            "Watch", "Photos", "Exclusive", "Review"}
    good_kinds = {"city", "town", "administrative", "country", "state",
                  "village", "municipality", "province", "region"}
    stop.update({"A", "An", "It", "Its", "As", "At", "On", "In", "He",
                 "She", "They", "We", "What", "How", "Who", "Why",
                 "When", "Now", "Then", "Here", "There", "Live",
                 "Update", "Analysis", "Opinion", "Video", "Watch",
                 "Photos", "Exclusive", "Review", "Rips", "Through",
                 "Bans", "Sentenced", "Killed", "Buys", "Requests"})
    # right-to-left: places skew to the end of a headline
    cands = [m.group(1) for m in _PLACE_RE.finditer(title)][::-1]
    for cand in cands:
        parts = cand.split()
        if len(parts) > 3:
            continue
        if len(parts) == 1 and cand in stop:
            continue
        if len(cand) > 40:
            continue
        cached = _GEO_CACHE.get(cand)
        if cached is not None:
            latlon, at = cached
            if latlon is None and time.time() - at < _NEG_TTL_S:
                continue  # recent miss; don't re-hit the geocoder
        else:
            latlon = None
            try:
                hits = _geo_lookup(cand, limit=2)
            except Exception:
                hits = []
            for h in hits if isinstance(hits, list) else []:
                if not isinstance(h, dict):
                    continue
                kind = str(h.get("kind") or "")
                if kind and kind not in good_kinds:
                    continue
                lat = h.get("lat")
                lon = h.get("lon") or h.get("lng")
                if lat is not None and lon is not None:
                    latlon = (float(lat), float(lon))
                    break
            _GEO_CACHE[cand] = (latlon, time.time())
        if latlon:
            return cand, latlon[0], latlon[1]
    return "", None, None


def refresh(max_per_feed: int = 25, geocode_budget_s: float = 25.0) -> dict:
    """Fetch all feeds, insert unseen titles, geocode within a budget.

    Geocoding is capped per run (budget + count) so a big news batch can
    never stall the caller; unplaced items are picked up on later runs.
    """
    inserted = 0
    placed = 0
    geo_count = 0
    t_start = time.time()
    errors: list[str] = []
    seen_titles: set[str] = set()
    with _lock:
        conn = _dbconn()
        known = {r[0] for r in
                 conn.execute("SELECT title FROM news ORDER BY id DESC "
                              "LIMIT 2000").fetchall()}
        # backfill queue: previously unplaced items, oldest first
        try:
            pending = [r[0] for r in conn.execute(
                "SELECT id FROM news WHERE lat IS NULL AND attempts < 3"
                " ORDER BY id LIMIT 12").fetchall()]
        except sqlite3.OperationalError:
            conn.execute("ALTER TABLE news ADD COLUMN attempts INTEGER DEFAULT 0")
            conn.commit()
            pending = []

        def _record(title: str, published: str, source: str,
                    link: str) -> None:
            nonlocal inserted
            if title in known:
                return
            known.add(title)
            conn.execute(
                "INSERT INTO news (first_seen, published, source,"
                " title, link) VALUES (?,?,?,?,?)",
                (time.time(), published, source, title, link))
            inserted += 1

        for feed_url in _feeds():
            try:
                items = _parse_feed(_fetch(feed_url))[:max_per_feed]
            except Exception as e:  # noqa: BLE001
                errors.append(f"{feed_url}: {e}")
                continue
            host = re.sub(r"^https?://(www\.)?", "", feed_url).split("/")[0]
            for it in items:
                it["source"] = host
                if it["title"] in known or it["title"] in seen_titles:
                    continue
                seen_titles.add(it["title"])
                if geo_count < 10 and time.time() - t_start < geocode_budget_s:
                    place, lat, lon = _geocode_place(it["title"])
                    geo_count += 1
                else:
                    place, lat, lon = "", None, None
                conn.execute(
                    "INSERT INTO news (first_seen, published, source,"
                    " title, link, place, lat, lon) VALUES (?,?,?,?,?,?,?,?)",
                    (time.time(), it["published"], it["source"], it["title"],
                     it["link"], place or None, lat, lon))
        # backfill: geocode a few previously unplaced items per run
        for row_id in pending:
            if geo_count >= 10 or time.time() - t_start > geocode_budget_s:
                break
            row = conn.execute(
                "SELECT title FROM news WHERE id=?", (row_id,)).fetchone()
            if not row:
                continue
            place, lat, lon = _geocode_place(row[0])
            geo_count += 1
            attempts = conn.execute(
                "SELECT attempts FROM news WHERE id=?", (row_id,)).fetchone()
            tries = ((attempts[0] if attempts else 0) or 0) + 1
            conn.execute(
                "UPDATE news SET place=?, lat=?, lon=?, attempts=?"
                " WHERE id=?",
                (place or None, lat, lon,
                 tries if lat is None else 99, row_id))
            if lat is not None:
                placed += 1
                inserted += 1
                if lat is not None:
                    placed += 1
        conn.commit()
    return {"inserted": inserted, "geolocated": placed, "errors": errors}


def latest(limit: int = 20) -> dict:
    with _lock:
        conn = _dbconn()
        rows = conn.execute(
            "SELECT first_seen, source, title, link, place, lat, lon"
            " FROM news ORDER BY id DESC LIMIT ?",
            (int(max(1, min(limit, 100))),)).fetchall()
        total_placed = conn.execute(
            "SELECT COUNT(*) FROM news WHERE lat IS NOT NULL").fetchone()[0]
        total = conn.execute("SELECT COUNT(*) FROM news").fetchone()[0]
    return {
        "items": [{"ts": r[0], "source": r[1], "title": r[2],
                   "place": r[4], "lat": r[5], "lon": r[6]}
                  for r in rows],
        "stored": total, "geolocated": total_placed,
    }
