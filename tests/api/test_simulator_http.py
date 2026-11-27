"""Test d'un rejeu historique via un vrai serveur HTTP local."""

import json
import socket
import time
from threading import Thread

import uvicorn

from novaria.api.main import create_app
from novaria.simulator.replay import Replay, read_measurements
from novaria.simulator.transport import HttpTransport
from tests.factories import raw_csv_frame


def test_simulator_sends_reference_and_spike_over_http(tmp_path, monkeypatch, capsys):
    """Envoie deux mesures HTTP et corrèle le résultat avec l'étiquette locale."""
    monkeypatch.setenv("NOVARIA_MODEL_PROVIDER", "mock")
    monkeypatch.delenv("NOVARIA_HOURLY_LIMITS", raising=False)
    path = tmp_path / "station.csv"
    raw_csv_frame("Changping", n=2, start="2016-03-01").to_csv(path, index=False)
    server = uvicorn.Server(uvicorn.Config(create_app(), log_level="error"))
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
        thread = Thread(target=server.run, kwargs={"sockets": [listener]}, daemon=True)
        thread.start()
        try:
            deadline = time.monotonic() + 5
            while (
                not server.started and thread.is_alive() and time.monotonic() < deadline
            ):
                time.sleep(0.01)
            assert server.started

            events = list(read_measurements(path, "Changping", "IoT-003"))
            replay = Replay(events, HttpTransport(f"http://127.0.0.1:{port}/predict"))
            replay.inject("pic")
            replay.run(0)
            spike, reference = [
                json.loads(line) for line in capsys.readouterr().out.splitlines()
            ]
            assert spike["event_id"] == events[0].event_id
            assert spike["is_anomaly"] and spike["anomaly_type"] == "pollution_atypique"
            assert reference["event_id"] == events[1].event_id
            assert not reference["is_anomaly"]
        finally:
            server.should_exit = True
            thread.join(timeout=5)
        assert not thread.is_alive()
