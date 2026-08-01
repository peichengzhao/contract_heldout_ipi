from pathlib import Path

from contract_heldout_ipi.contract.loader import load_episode
from contract_heldout_ipi.referee import Referee

ROOT = Path(__file__).resolve().parents[1]


def test_train_seed_episode_accepted():
    path = ROOT / "episodes" / "train" / "email_meeting_summary_exfil_001.json"
    episode = load_episode(path)
    report = Referee().validate(episode, raw=episode.model_dump(mode="json"))
    assert report.ok, report.summary()


def test_heldout_seed_episode_accepted():
    path = ROOT / "episodes" / "heldout" / "email_invoice_forward_exfil_001.json"
    episode = load_episode(path)
    report = Referee().validate(episode, raw=episode.model_dump(mode="json"))
    assert report.ok, report.summary()
