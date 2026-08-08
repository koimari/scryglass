from __future__ import annotations

from pathlib import Path

import pytest

from tools.lol_mechanics_mcp.server import (
    LeagueMechanicsServer,
    StartupError,
    _tool_specs,
    dispatch,
)


INDEX = Path("data/lol/knowledge/patch-packets/cdragon/2026/26.15/mechanics-index.json")


class FakeEngine:
    def __init__(self, pack: object) -> None:
        self.pack = pack
        self.calls: list[str] = []

    def answer(self, question: str) -> dict[str, object]:
        self.calls.append(question)
        return {
            "status": "available",
            "display": f"ready: {question}",
            "assumptions": ["patch 26.15"],
        }


def test_mcp_initialize_and_answer_call() -> None:
    engine = FakeEngine({"patch": "26.15"})
    server = LeagueMechanicsServer(engine=engine, pack=engine.pack)
    initialized = dispatch(
        server,
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {"protocolVersion": "2024-11-05"},
        },
    )
    assert initialized is not None
    assert initialized["result"]["serverInfo"]["name"] == "scryglass-lol-mechanics"
    assert "league_oracle_answer" in initialized["result"]["instructions"]

    response = dispatch(
        server,
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "league_mechanics_answer",
                "arguments": {"question": "Malphite MP5 at level 13"},
            },
        },
    )
    assert response is not None
    assert response["result"]["structuredContent"]["status"] == "available"
    assert engine.calls == ["Malphite MP5 at level 13"]


def test_tool_list_exposes_one_canonical_oracle_entrypoint() -> None:
    names = [item["name"] for item in _tool_specs()]
    assert names == ["league_oracle_answer", "league_oracle_status"]


def test_real_resident_server_routes_exact_and_semantic_requests() -> None:
    server = LeagueMechanicsServer(index_path=INDEX)
    try:
        status = server.status()
        assert status["oracle_router_loaded"] is True
        assert status["semantic_engine_loaded"] is True
        assert status["wiki_fallback_loaded"] is True

        exact = server.answer({"question": "What is Gnar's attack damage at level 14?"})
        assert exact["status"] == "available"
        assert exact["value"] == 98.69
        assert exact["route"] == "exact_packet"
        assert exact["router"] == "oracle-router-v1.1.0"

        composite = server.answer(
            {
                "question":
                "On patch 26.15, Malphite is level 6 with rank-3 Seismic Shard, "
                "one Sapphire Crystal equipped, and Manaflow Band at 10 stacks. "
                "Starting at full mana with no regeneration, how many Q casts "
                "can he make before running out?"
            }
        )
        assert composite["status"] == "available"
        assert composite["value"] == 13
        assert composite["remainder"] == 27
        assert composite["route"] == "exact_packet"

        unspecified = server.answer(
            {
                "question":
                "At level 6, how many rank-3 casts of Malphite's Q can be made "
                "with Sapphire Crystal and Manaflow Band from full mana?"
            }
        )
        assert unspecified["status"] == "unsupported"
        assert unspecified["value"] is None
        assert "stack state is required" in unspecified["reason"]

        semantic = server.answer({"question": "How much damage does Aatrox deal with the current build?"})
        assert semantic["status"] == "needs_input"
        assert semantic["route"] == "semantic_contract"
        assert semantic["value"] is None

        wiki = server.answer(
            {
                "question": "closed mode rule",
                "context": {
                    "intent": "mode_rule",
                    "patch": "26.15",
                    "mode": "arena",
                    "rule": "anvil augment multiplier",
                },
            }
        )
        assert wiki["status"] == "unsupported"
        assert wiki["route"] == "wiki_evidence"
        assert wiki["wiki_evidence"]["response_mode"] == "fast"
        assert wiki["wiki_evidence"]["evidence"]
    finally:
        server.close()


def test_status_reports_resident_engine_without_reloading() -> None:
    engine = FakeEngine({"patch": "26.15"})
    server = LeagueMechanicsServer(engine=engine, pack=engine.pack)
    first = server.status()
    server.answer({"question": "first"})
    second = server.status()
    assert first["resident"] is True
    assert first["pack_loaded"] is True
    assert first["patch"] == "26.15"
    assert first["answers_served"] == 0
    assert second["answers_served"] == 1
    assert second["network_calls"] == 0


def test_malformed_input_and_invalid_answer_are_fail_closed() -> None:
    server = LeagueMechanicsServer(engine=FakeEngine({}), pack={})
    response = dispatch(server, {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "league_mechanics_answer", "arguments": {}}})
    assert response is not None
    assert response["result"]["isError"] is True

    invalid = dispatch(server, {"jsonrpc": "2.0", "id": 4, "method": "not-a-method"})
    assert invalid is not None
    assert invalid["error"]["code"] == -32601


def test_default_compile_and_load_are_called_once_for_warm_calls(tmp_path: Path) -> None:
    index = tmp_path / "26.15" / "mechanics-index.json"
    index.parent.mkdir(parents=True)
    index.write_text('{"patch":"26.15"}\n', encoding="utf-8")
    fastpack = tmp_path / "fastpack.json"
    fastpack.write_text('{"patch":"26.15"}\n', encoding="utf-8")
    events: list[tuple[str, Path]] = []

    def compile(index_path: Path) -> Path:
        events.append(("compile", index_path))
        return fastpack

    def load(path: Path) -> dict[str, str]:
        events.append(("load", path))
        return {"patch": "26.15"}

    server = LeagueMechanicsServer(
        engine_class=FakeEngine,
        compiler=compile,
        loader=load,
        index_path=index,
    )
    server.answer({"question": "one"})
    server.answer({"question": "two"})
    assert events == [("compile", index.resolve()), ("load", fastpack.resolve())]


def test_explicit_fastpack_path_is_loaded_without_compiling(tmp_path: Path) -> None:
    fastpack = tmp_path / "explicit.fastpack.json"
    fastpack.write_text("{}\n", encoding="utf-8")
    events: list[str] = []

    def compile(_index_path: Path) -> object:
        events.append("compile")
        return {"patch": "wrong"}

    def load(path: Path) -> dict[str, str]:
        events.append(f"load:{path.name}")
        return {"patch": "26.15"}

    server = LeagueMechanicsServer(
        engine_class=FakeEngine,
        compiler=compile,
        loader=load,
        fastpack_path=fastpack,
    )
    assert server.status()["patch"] == "26.15"
    assert events == ["load:explicit.fastpack.json"]


def test_explicit_environment_failure_does_not_fallback_to_local_index(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("SCRYGLASS_LOL_MECHANICS_FASTPACK", str(tmp_path / "missing.json"))
    events: list[str] = []

    def compile(_index_path: Path) -> object:
        events.append("compile")
        return {"patch": "26.14"}

    def load(path: Path) -> object:
        events.append(f"load:{path.name}")
        raise FileNotFoundError(path)

    with pytest.raises(StartupError):
        LeagueMechanicsServer(
            engine_class=FakeEngine,
            compiler=compile,
            loader=load,
            packet_root=tmp_path,
        )
    assert events == ["load:missing.json"]
