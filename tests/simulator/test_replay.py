"""Tests du rejeu chronologique, des scénarios et de l'interface de transport."""

from datetime import datetime

import pytest

from novaria.common.schemas import RawMeasurement
from novaria.simulator.replay import SCENARIOS, Replay, read_measurements
from novaria.simulator.transport import ConsoleTransport


@pytest.fixture
def sample_csv(tmp_path):
    """Crée un CSV mêlant une ligne de test et trois lignes de simulation."""
    path = tmp_path / "station.csv"
    path.write_text(
        "year,month,day,hour,PM2.5,PM10,NO2,CO,SO2,O3,TEMP,wd,station\n"
        "2016,3,1,3,12,48,22,850,10,50,-1,N,Changping\n"
        "2016,2,29,23,99,100,21,801,9,52,-2,,Changping\n"
        "2016,3,1,1,11,33,20,800,8,51,-2,,Changping\n"
        "2016,3,1,2,13,35,21,801,9,53,-2,W,Changping\n"
        "2016,3,1,0,42,56,21,801,9,52,-2,,Other\n"
        "2017,3,1,0,42,56,21,801,9,52,-2,,Changping\n",
        encoding="utf-8",
    )
    return path


def test_reference_uses_only_demo_period(sample_csv, capsys):
    """Vérifie les bornes temporelles, le filtre station et le JSON de référence."""
    replay = Replay(
        read_measurements(sample_csv, "Changping", "IoT-003"), ConsoleTransport()
    )
    replay.run(0)
    events = [
        RawMeasurement.model_validate_json(line)
        for line in capsys.readouterr().out.splitlines()
    ]
    assert [event.timestamp for event in events] == [
        datetime(2016, 3, 1, h) for h in (1, 2, 3)
    ]
    assert len({event.event_id for event in events}) == 3
    assert events[0].sensor_id == "IoT-003"
    assert events[0].pm25 == 11
    assert events[0].wd is None
    assert events[0].injection is None


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_one_injection_leaves_following_reference_intact(sample_csv, capsys, scenario):
    """Vérifie les valeurs, l'étiquette et la portée de chaque scénario."""
    replay = Replay(
        read_measurements(sample_csv, "Changping", "IoT-003"), ConsoleTransport()
    )
    replay.inject(scenario)
    replay.run(0, limit=2)
    first, second = [
        RawMeasurement.model_validate_json(line)
        for line in capsys.readouterr().out.splitlines()
    ]
    assert first.injection == scenario
    assert {name: getattr(first, name) for name in SCENARIOS[scenario]} == SCENARIOS[
        scenario
    ]
    assert second.injection is None
    assert second.pm25 == 13


def test_stop_prevents_further_emission(sample_csv):
    """Arrête le rejeu à la première émission sans attendre le délai configuré."""

    class StopAfterOne:
        def send(self, event):
            sent.append(event)
            replay.stop()

    sent = []
    replay = Replay(
        read_measurements(sample_csv, "Changping", "IoT-003"), StopAfterOne()
    )
    replay.run(interval_seconds=60)
    assert len(sent) == 1


def test_unknown_station_has_no_demo_data(sample_csv):
    """Signale une station absente du fichier de rejeu."""
    with pytest.raises(ValueError, match="Aucune mesure"):
        list(read_measurements(sample_csv, "Absent", "IoT-003"))
