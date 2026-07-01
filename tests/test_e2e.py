"""Functional end-to-end tests driving the real CLI and API on crafted documents.

The full path - argument parsing, ``_read``, segmentation, encode, ``_build_diff`` / ``_build_transport_map``,
JSON output - runs for real; only the model layer (SAT segmenter + mmBERT encoder) is faked so the suite
stays offline in the uv ``.venv``. The fake encoder maps each statement text to a deterministic unit vector
(identical text -> identical vector, different text -> near-orthogonal vector), so a reorder is content-
preserving (SMD ~ 0, positive order-gap) and a reword shows up as a per-statement ``semantic_gap`` spike.
Real-model e2e lives in ``notebooks/09-kj-docdistance-api-e2e.ipynb``.
"""

import hashlib
import json

import numpy as np
import pytest
from typer.testing import CliRunner

from docdistance import pipeline, settings
from docdistance.cli import app
from docdistance.pipeline import DIFF_CHANGED_COST, DocDistance, document_distance

runner = CliRunner()
_WIDE = {"COLUMNS": "200", "NO_COLOR": "1", "TERM": "dumb"}

# crafted documents - four statements each; the fake segmenter splits on the full stop
DOC = "The cat sat still. The dog ran fast. Birds fly high. Fish swim deep."
DOC_SWAP = "The cat sat still. The dog ran fast. Fish swim deep. Birds fly high."  # statements 2<->3 swapped
DOC_REWORD = "The cat sat still. The dog ran fast. Birds fly high. Volcanoes erupt violently."  # stmt 3 reworded


def _vec(text: str, dim: int = 32) -> np.ndarray:
    seed = int(hashlib.sha1(text.encode()).hexdigest(), 16) % (2**32)
    v = np.random.default_rng(seed).standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


class _FakeSegmenter:
    def __init__(self, offline: bool = True):
        pass

    def split(self, text: str) -> list[str]:
        return [s.strip() for s in text.replace("\n", ". ").split(".") if s.strip()]


class _FakeEncoder:
    def encode(self, statements: list[str]) -> np.ndarray:
        return np.stack([_vec(s) for s in statements])


def _fake_load_encoder(backend, offline=True, device=None):
    return _FakeEncoder()


@pytest.fixture(autouse=True)
def _wired(monkeypatch):
    """Mark the wmd mode ready and swap in the fake segmenter + encoder for every test here."""
    settings.reset()
    settings.mark_ready("wmd")
    monkeypatch.setattr(pipeline, "Segmenter", _FakeSegmenter)
    monkeypatch.setattr(pipeline, "load_encoder", _fake_load_encoder)
    yield
    settings.reset()


def _by_index(diff: dict) -> dict[int, dict]:
    return {s["index"]: s for s in diff["statements"]}


# --------------------------------------------------------------------------- CLI


def test_cli_distance_plain_reports_a_verdict():
    res = runner.invoke(app, ["distance", DOC, DOC_SWAP], env=_WIDE)
    assert res.exit_code == 0, res.output
    assert "SMD" in res.output
    assert "closeness" in res.output


def test_cli_distance_json_emits_machine_readable():
    res = runner.invoke(app, ["distance", DOC, DOC_REWORD, "--json"], env=_WIDE)
    assert res.exit_code == 0, res.output
    assert '"smd"' in res.output and '"verdict"' in res.output


def test_cli_diff_json_localizes_a_reorder(tmp_path):
    """A pure statement swap: content preserved (SMD ~ 0), positive order-gap, only the swapped pair moves."""
    out = tmp_path / "diff.json"
    res = runner.invoke(app, ["distance", DOC, DOC_SWAP, "--diff-json", str(out)], env=_WIDE)
    assert res.exit_code == 0, res.output
    diff = json.loads(out.read_text())

    assert diff["smd"] == pytest.approx(0.0, abs=2e-3)  # same statements, reordered (float floor ~3e-4)
    assert diff["order_gap"] > 0.0  # arrangement changed
    assert diff["structure_closeness"] < 1.0
    st = _by_index(diff)
    assert st[0]["displacement"] == 0 and st[0]["moved"] is False  # statements 0,1 stayed put
    assert st[1]["displacement"] == 0 and st[1]["moved"] is False
    assert st[2]["moved"] is True and st[3]["moved"] is True  # the swapped pair moved
    for s in diff["statements"]:
        assert s["semantic_gap"] < 0.01  # nothing changed in meaning (floors at ~3e-4, not exact 0)
        assert s["changed"] is False


def test_cli_diff_json_localizes_a_reword(tmp_path):
    """One reworded statement: its semantic_gap spikes and it flags changed, order is untouched."""
    out = tmp_path / "diff.json"
    res = runner.invoke(app, ["distance", DOC, DOC_REWORD, "--diff-json", str(out)], env=_WIDE)
    assert res.exit_code == 0, res.output
    diff = json.loads(out.read_text())

    st = _by_index(diff)
    assert st[3]["semantic_gap"] > DIFF_CHANGED_COST  # the reworded statement
    assert st[3]["changed"] is True
    assert st[3]["displacement"] == 0  # it did not move, only its wording changed
    for i in (0, 1, 2):
        assert st[i]["semantic_gap"] < 0.01  # the untouched statements (floor ~3e-4)
        assert st[i]["changed"] is False


def test_cli_transport_map_json_writes_flows(tmp_path):
    out = tmp_path / "map.json"
    res = runner.invoke(app, ["distance", DOC, DOC_SWAP, "--transport-map-json", str(out)], env=_WIDE)
    assert res.exit_code == 0, res.output
    m = json.loads(out.read_text())
    assert m["n_statements"] == {"a": 4, "b": 4}
    assert len(m["flows"]) == 4
    assert all(f["matches"] for f in m["flows"])


def test_cli_result_only_prints_a_bare_scalar():
    res = runner.invoke(app, ["distance", DOC, DOC_SWAP, "--result-only"], env=_WIDE)
    assert res.exit_code == 0, res.output
    floats = [float(tok) for line in res.output.splitlines() for tok in line.split() if _isfloat(tok)]
    assert floats and floats[-1] == pytest.approx(0.0, abs=1e-4)  # reorder -> SMD ~ 0


def _isfloat(tok: str) -> bool:
    try:
        float(tok)
        return True
    except ValueError:
        return False


# --------------------------------------------------------------------------- API


def test_api_document_distance_on_text():
    r = document_distance(DOC, DOC_REWORD)
    assert 0.0 <= r.closeness <= 1.0
    assert r.verdict in {"similar", "not similar"}
    assert r.n_statements_a == 4 and r.n_statements_b == 4
    assert r.wcd <= r.smd + 1e-6 and r.rwmd <= r.smd + 1e-6  # both are lower bounds of SMD


def test_api_distance_with_diff_localizes_reorder_and_reword():
    dd = DocDistance()

    _, reorder = dd.distance_with_diff(DOC, DOC_SWAP)
    assert reorder["smd"] == pytest.approx(0.0, abs=2e-3)
    assert reorder["order_gap"] > 0.0
    moved = {s["index"] for s in reorder["statements"] if s["moved"]}
    assert moved == {2, 3}

    _, reword = dd.distance_with_diff(DOC, DOC_REWORD)
    changed = {s["index"] for s in reword["statements"] if s["changed"]}
    assert changed == {3}
    assert all(s["displacement"] == 0 for s in reword["statements"])  # nothing moved, only reworded


def test_api_distance_with_map_shares_one_encode_pass():
    dd = DocDistance()
    result, m = dd.distance_with_map(DOC, DOC_SWAP)
    assert m["smd"] == pytest.approx(result.smd, abs=1e-6)  # map and result agree
    assert len(m["flows"]) == 4


def test_api_and_cli_agree_on_smd(tmp_path):
    """The same pair scored through the API and through the CLI diff path gives the same SMD."""
    api_smd = document_distance(DOC, DOC_REWORD).smd
    out = tmp_path / "diff.json"
    res = runner.invoke(app, ["distance", DOC, DOC_REWORD, "--diff-json", str(out)], env=_WIDE)
    assert res.exit_code == 0, res.output
    cli_smd = json.loads(out.read_text())["smd"]
    assert cli_smd == pytest.approx(api_smd, abs=1e-6)
