import glob
import json
import os
import shutil
import socket
import time
import urllib.request

_HWMON_GLOB = "/sys/class/drm/card*/device/hwmon/hwmon*"
_ENERGY_STATE = {}

SLOTS = [
    ("duty-vllm", 8000),
    ("backlot-api", 8001),
    ("hoa64-webapp", 8770, "127.0.0.1"),
    ("comfy-main", 8188),
    ("comfy-h3", 8189),
    ("comfy-lab", 8190),
    ("pi-bridge", 8642),
    ("desktop-bridge", 19421),
]


def _read(path):
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return None


def _port_up(port, hosts=("127.0.0.1",), timeout=0.35):
    for h in hosts:
        try:
            with socket.socket() as s:
                s.settimeout(timeout)
                if s.connect_ex((h, int(port))) == 0:
                    return True
        except OSError:
            continue
    return False


def gpu_hwmon():
    out = []
    for hw in sorted(glob.glob(_HWMON_GLOB)):
        card = hw.split("/")[4]
        entry = {"card": card, "chip": (_read(os.path.join(hw, "name")) or "").strip()}
        temp = _read(os.path.join(hw, "temp1_input"))
        if temp:
            entry["temp_c"] = round(int(temp) / 1000, 1)
        cap = _read(os.path.join(hw, "power1_cap"))
        if cap:
            entry["power_cap_w"] = round(int(cap) / 1_000_000, 1)
        rating = _read(os.path.join(hw, "power1_max"))
        if rating:
            entry["power_rating_w"] = round(int(rating) / 1_000_000, 1)
        energy = _read(os.path.join(hw, "energy1_input"))
        if energy:
            now = time.monotonic()
            val = int(energy)
            prev = _ENERGY_STATE.get(hw)
            if prev and now > prev[1]:
                entry["power_w"] = round(
                    max(0.0, (val - prev[0]) / ((now - prev[1]) * 1_000_000)), 1
                )
            _ENERGY_STATE[hw] = (val, now)
        out.append(entry)
    return out


def slot_health():
    out = []
    for entry in SLOTS:
        name, port = entry[0], entry[1]
        hosts = ("127.0.0.1",) + tuple(entry[2:])
        out.append({"name": name, "port": port, "host": list(hosts), "up": _port_up(port, hosts)})
    return out


def comfy_queue(url="http://127.0.0.1:8188/queue", timeout=1.0):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            data = json.loads(r.read())
        return {
            "running": len(data.get("queue_running") or []),
            "pending": len(data.get("queue_pending") or []),
        }
    except Exception:
        return None


def disk_free(path="/"):
    u = shutil.disk_usage(path)
    return {
        "path": path,
        "total_gb": round(u.total / 1e9, 1),
        "free_gb": round(u.free / 1e9, 1),
    }


def memory():
    info = {}
    with open("/proc/meminfo") as f:
        for line in f:
            key, _, val = line.partition(":")
            fields = val.split()
            info[key.strip()] = int(fields[0]) if fields else 0
    zram_kb = 0
    try:
        with open("/proc/swaps") as f:
            for line in f.readlines()[1:]:
                if "zram" in line:
                    zram_kb += int(line.split()[2])
    except OSError:
        pass
    return {
        "mem_available_gb": round(info.get("MemAvailable", 0) / 1e6, 1),
        "mem_total_gb": round(info.get("MemTotal", 0) / 1e6, 1),
        "zram_total_gb": round(zram_kb / 1e6, 1),
    }


def uptime_s():
    with open("/proc/uptime") as f:
        return int(float(f.read().split()[0]))


def ship_status():
    return {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "gpus": gpu_hwmon(),
        "slots": slot_health(),
        "comfy_queue": comfy_queue(),
        "disk": disk_free(),
        "memory": memory(),
        "uptime_s": uptime_s(),
        "load": [round(x, 2) for x in os.getloadavg()],
    }
