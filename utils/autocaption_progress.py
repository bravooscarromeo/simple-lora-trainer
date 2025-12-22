from threading import Lock

_progress = {
    "running": False,
    "current": 0,
    "total": 0,
}

_lock = Lock()

def start(total: int):
    with _lock:
        _progress["running"] = True
        _progress["current"] = 0
        _progress["total"] = total

def step():
    with _lock:
        _progress["current"] += 1

def finish():
    with _lock:
        _progress["running"] = False

def get():
    with _lock:
        return dict(_progress)
