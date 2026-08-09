from __future__ import annotations

from lol_kills.knowledge.oracle_router import LeagueOracleRouter


class FakeExact:
    patch = "26.15"

    def __init__(self) -> None:
        self.questions: list[str] = []

    def answer(self, question: str) -> dict[str, object]:
        self.questions.append(question)
        if "current build" in question:
            return {"status": "unsupported", "value": None, "reason": "missing state"}
        return {"status": "available", "value": 42, "patch": "26.15"}


class FakeSemantic:
    def __init__(self) -> None:
        self.questions: list[str] = []

    def answer(self, question: str, context: object = None) -> dict[str, object]:
        self.questions.append(question)
        return {
            "status": "needs_input",
            "intent": "build_damage",
            "value": None,
            "required_inputs": [{"path": "patch"}],
        }


def test_exact_packet_is_always_first_for_natural_language() -> None:
    exact = FakeExact()
    semantic = FakeSemantic()
    router = LeagueOracleRouter(exact, semantic)  # type: ignore[arg-type]

    result = router.answer("What is Malphite's attack damage at level 14?")

    assert result["status"] == "available"
    assert result["route"] == "exact_packet"
    assert semantic.questions == []


def test_unsupported_exact_result_enters_semantic_contract_path() -> None:
    exact = FakeExact()
    semantic = FakeSemantic()
    router = LeagueOracleRouter(exact, semantic)  # type: ignore[arg-type]

    result = router.answer("How much damage does Aatrox deal with the current build?")

    assert result["status"] == "needs_input"
    assert result["route"] == "semantic_contract"
    assert semantic.questions == ["How much damage does Aatrox deal with the current build?"]


def test_explicit_context_enters_semantic_path_without_exact_recalculation() -> None:
    exact = FakeExact()
    semantic = FakeSemantic()
    router = LeagueOracleRouter(exact, semantic)  # type: ignore[arg-type]

    result = router.answer("calculate this", {"intent": "build_damage"})

    assert result["route"] == "semantic_contract"
    assert exact.questions == []
