import json, time, threading
import numpy as np
import torch
from flask import Flask, jsonify, request
from flask_sock import Sock
from model.net import WinProbNet, FEATURE_COLS
from server.poller import LivePoller

app = Flask(__name__)
sock = Sock(app)

# ── Load model once at startup ──────────────────────────────────────────────
device = torch.device("cpu")
model = WinProbNet()
model.load_state_dict(torch.load("checkpoints/model.pt", map_location=device))
model.eval()

def predict(features: dict) -> float:
    x = torch.tensor(
        [[features.get(f, 0.0) for f in FEATURE_COLS]],
        dtype=torch.float32
    )
    with torch.no_grad():
        return float(model(x).item())

# ── REST endpoint ────────────────────────────────────────────────────────────
@app.route("/predict", methods=["POST"])
def predict_endpoint():
    data = request.get_json(force=True)
    prob = predict(data)
    return jsonify({"home_win_prob": prob, "away_win_prob": 1 - prob})

# ── WebSocket endpoint ───────────────────────────────────────────────────────
connected_ws = []

@sock.route("/live")
def live(ws):
    connected_ws.append(ws)
    try:
        while True:
            msg = ws.receive(timeout=30)
            if msg is None:
                break
    finally:
        connected_ws.remove(ws)

def broadcast(payload: dict):
    dead = []
    for ws in connected_ws:
        try:
            ws.send(json.dumps(payload))
        except Exception:
            dead.append(ws)
    for ws in dead:
        connected_ws.remove(ws)

# ── Background polling thread ────────────────────────────────────────────────
def polling_loop(game_id: str, interval: float = 5.0):
    poller = LivePoller(game_id)
    while True:
        state = poller.get_current_state()
        if state:
            prob = predict(state["features"])
            state["home_win_prob"] = prob
            state["away_win_prob"] = 1 - prob
            broadcast(state)
        time.sleep(interval)

def start_polling(game_id: str):
    t = threading.Thread(target=polling_loop, args=(game_id,), daemon=True)
    t.start()

@app.route("/start/<game_id>", methods=["POST"])
def start_game(game_id):
    start_polling(game_id)
    return jsonify({"status": "polling started", "game_id": game_id})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)