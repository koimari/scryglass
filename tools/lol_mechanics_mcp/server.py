#!/usr/bin/env python3
"""Dependency-free MCP server for the resident League oracle.

The server keeps the exact patch packet, exact oracle, semantic state router,
and normalized read-only state warehouse resident.  The model exposes one
canonical ``league_oracle_answer`` tool; the router first uses exact packet
mechanics, then semantic slot-filling/state execution, and never guesses a
missing rule.  The same tool accepts bounded warehouse operations for direct
SQL/schema/state work.  Legacy ``league_mechanics_*`` names remain callable as
compatibility aliases but are not advertised in the tool list.

The normal process mode reads newline-delimited JSON-RPC messages from stdin
and writes one response per request to stdout.  ``--question`` is a convenient
one-shot mode for local callers that have not installed an MCP entry yet.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Callable, Mapping


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SUPPORTED_PROTOCOL_VERSIONS = ("2024-11-05", "2025-03-26", "2025-06-18")
SUPPORTED_PROTOCOL_VERSION = SUPPORTED_PROTOCOL_VERSIONS[0]
SERVER_NAME = "scryglass-lol-mechanics"
SERVER_VERSION = "0.3.0"
FASTPACK_ENV = "SCRYGLASS_LOL_MECHANICS_FASTPACK"
DEFAULT_PACKET_ROOT = ROOT / "data/lol/knowledge/patch-packets/cdragon"
DEFAULT_WAREHOUSE_MAX_ROWS = 200


class ToolError(ValueError):
    """An invalid tool argument or unavailable mechanics result."""


class StartupError(RuntimeError):
    """The resident engine/fastpack could not be initialized."""


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)


def _jsonable(value: Any) -> Any:
    """Convert common result containers without changing engine semantics.

    The fast engine normally returns a plain mapping.  The small adapter is
    useful for tests and for a future frozen answer dataclass, while avoiding
    a second formatting/calculation layer in this transport.
    """

    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return dataclasses.asdict(value)
    for name in ("to_mapping", "to_dict", "as_dict"):
        method = getattr(value, name, None)
        if callable(method):
            converted = method()
            if converted is not value:
                return _jsonable(converted)
    if isinstance(value, Mapping):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, set):
        return sorted(_jsonable(item) for item in value)
    return value


def _patch_key(value: Any) -> tuple[int, int, int, str]:
    """Sort public patch labels semantically and deterministically."""

    text = str(value or "").strip().strip("/")
    match = re.fullmatch(r"(\d+)\.(\d+)", text)
    if match:
        return (int(match.group(1)), int(match.group(2)), 0, text)
    match = re.fullmatch(r"(\d+)\.S(\d+)\.(\d+)", text, flags=re.IGNORECASE)
    if match:
        return (int(match.group(1)), int(match.group(2)), int(match.group(3)), text)
    # Unknown labels sort before a known semantic patch and are then stable by
    # path.  They are never treated as a silent nearest-patch fallback.
    return (-1, -1, -1, text)


def _index_patch(path: Path) -> str:
    """Read a local index's declared patch, falling back to its directory."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, Mapping) and isinstance(payload.get("patch"), str):
            return str(payload["patch"]).strip("/")
    except (OSError, UnicodeError, json.JSONDecodeError):
        # A malformed index is rejected while its legacy path fallback remains available.
        pass
    # A malformed index must not become a candidate just because its parent is
    # named like a patch; the caller will still fail when compiling it.  The
    # path fallback is only for old/local fixtures without a patch field.
    for parent in path.parents:
        candidate = parent.name
        if re.fullmatch(r"\d+\.(?:\d+|S\d+\.\d+)", candidate, flags=re.IGNORECASE):
            return candidate
    return ""


def _discover_index(packet_root: Path = DEFAULT_PACKET_ROOT) -> Path:
    """Find the newest exact local packet index without network access."""

    candidates: list[tuple[tuple[int, int, int, str], str, Path]] = []
    if packet_root.exists():
        for path in packet_root.rglob("mechanics-index.json"):
            if not path.is_file():
                continue
            patch = _index_patch(path)
            if not patch:
                continue
            candidates.append((_patch_key(patch), str(path.resolve()), path.resolve()))
    if not candidates:
        raise StartupError(f"no local exact mechanics-index.json found under {packet_root}")
    # Sort by semantic patch first and absolute path as a stable tiebreaker.
    candidates.sort(key=lambda item: (item[0], item[1]))
    return candidates[-1][2]


def _component(name: str) -> Any:
    """Resolve one engine component through local, dependency-injected modules.

    Keeping these imports inside startup makes this transport easy to test with
    fakes and lets the mechanics worker adjust its module split without making
    MCP import-time startup fragile.
    """

    modules = (
        "lol_kills.knowledge.quick_mechanics",
        "lol_kills.knowledge.quick_mechanics_fastpack",
        "lol_kills.knowledge.mechanics_fastpack",
        "lol_kills.knowledge.mechanics_engine",
        "lol_kills.knowledge.cdragon_patch_packet",
    )
    errors: list[str] = []
    for module_name in modules:
        try:
            module = importlib.import_module(module_name)
        except (ImportError, ModuleNotFoundError) as exc:
            errors.append(f"{module_name}: {exc}")
            continue
        candidate = getattr(module, name, None)
        if candidate is not None:
            return candidate
    raise StartupError(
        f"required mechanics component {name} is unavailable; checked "
        + ", ".join(modules)
    )


def _load_components(
    *,
    engine_class: type[Any] | Callable[[Any], Any] | None = None,
    compiler: Callable[[Path], Any] | None = None,
    loader: Callable[[Path], Any] | None = None,
) -> tuple[Any, Callable[[Path], Any], Callable[[Path], Any]]:
    return (
        engine_class or _component("QuickMechanicsEngine"),
        compiler or _component("compile_fastpack"),
        loader or _component("load_fastpack"),
    )


def _path_from_compiled(compiled: Any) -> Path | None:
    """Extract a fastpack path from common compiler return shapes."""

    if isinstance(compiled, (str, os.PathLike)):
        return Path(compiled).expanduser().resolve()
    if isinstance(compiled, Mapping):
        for key in ("path", "fastpack_path", "output", "output_path"):
            candidate = compiled.get(key)
            if isinstance(candidate, (str, os.PathLike)):
                return Path(candidate).expanduser().resolve()
    for attr in ("path", "fastpack_path", "output_path"):
        candidate = getattr(compiled, attr, None)
        if isinstance(candidate, (str, os.PathLike)):
            return Path(candidate).expanduser().resolve()
    return None


def _compile_or_load(
    index_path: Path,
    *,
    compiler: Callable[[Path], Any],
    loader: Callable[[Path], Any],
    configured_path: Path | None,
) -> tuple[Any, Path | None, Any]:
    """Build/load one resident pack, preserving explicit path failures.

    ``compile_fastpack`` has intentionally stayed a small worker API.  During
    its evolution it may return an already-loaded pack or a path to one; both
    shapes are accepted here.  A configured fastpack is always loaded directly
    and never replaced with another local patch after failure.
    """

    if configured_path is not None:
        pack = loader(configured_path)
        return pack, configured_path, None

    compiled = compiler(index_path)
    path = _path_from_compiled(compiled)
    if path is not None:
        return loader(path), path, compiled
    if compiled is not None:
        # A compiler may materialize and return the resident pack itself.
        return compiled, None, compiled
    # A None-returning compiler is allowed when it writes a conventional file.
    possible = (
        index_path.with_name("fastpack.json"),
        index_path.with_name("mechanics-fastpack.json"),
        index_path.with_suffix(".fastpack.json"),
    )
    for path in possible:
        if path.is_file():
            return loader(path), path, None
    raise StartupError(
        f"compile_fastpack returned no pack/path and no conventional fastpack exists for {index_path}"
    )


def _build_oracle_router(
    *,
    pack: Any,
    index_path: Path | None,
    oracle_engine: Any | None = None,
    semantic_engine: Any | None = None,
    router: Any | None = None,
) -> tuple[Any | None, str | None]:
    """Build the resident exact/semantic router without weakening startup.

    Dependency-injected tests may provide only a fake quick engine or a fake
    router; production startup has the exact index path and can construct the
    full League oracle.  A router construction failure is reported in status
    and leaves the legacy exact engine usable rather than selecting a fallback
    patch or silently fabricating semantic answers.
    """

    if router is not None:
        return router, None
    if semantic_engine is not None:
        if oracle_engine is None:
            return None, "semantic engine was injected without an exact oracle"
        try:
            from lol_kills.knowledge.oracle_router import LeagueOracleRouter

            return LeagueOracleRouter(oracle_engine, semantic_engine), None
        except Exception as exc:  # pragma: no cover - defensive dependency boundary
            return None, f"router construction failed: {exc}"
    if index_path is None:
        exact_candidate = oracle_engine or getattr(router, "exact_engine", None)
        raw_root = getattr(exact_candidate, "raw_champion_root", None)
        if raw_root is not None:
            candidate = Path(raw_root).parent.parent / "mechanics-index.json"
            if candidate.is_file():
                index_path = candidate
    if index_path is None or not isinstance(pack, Mapping):
        return None, "semantic router requires a compiled pack and exact index path"
    try:
        from lol_kills.knowledge.lol_oracle import LeagueOracleEngine
        from lol_kills.knowledge.oracle_router import LeagueOracleRouter
        from lol_kills.knowledge.semantic_engine import SemanticOracleEngine

        exact = oracle_engine or LeagueOracleEngine(
            pack,
            raw_champion_root=index_path.parent / "raw" / "champions",
        )
        semantic = SemanticOracleEngine(exact)
        return LeagueOracleRouter(exact, semantic), None
    except Exception as exc:  # pragma: no cover - startup diagnostics are surfaced in status
        return None, f"router construction failed: {exc}"


def _build_state_warehouse(
    *,
    pack: Any,
    index_path: Path | None,
    oracle_engine: Any | None = None,
    router: Any | None = None,
    warehouse: Any | None = None,
) -> tuple[Any | None, str | None]:
    """Build one resident normalized state warehouse.

    The warehouse is an optional serving companion: dependency-injected tests
    and legacy quick engines can run without it, while a production packet
    with an exact index receives a read-only SQLite star schema at startup.
    The exact engine already resident in the router is reused so champion/item
    payloads are never loaded twice.
    """

    if warehouse is not None:
        return warehouse, None
    if index_path is None or not isinstance(pack, Mapping):
        return None, "state warehouse requires a compiled pack and exact index path"
    try:
        from lol_kills.knowledge.state_warehouse import StateWarehouse

        exact = oracle_engine or getattr(router, "exact_engine", None)
        return StateWarehouse(pack, oracle=exact, index_path=index_path), None
    except Exception as exc:  # pragma: no cover - startup diagnostics are surfaced in status
        return None, f"state warehouse construction failed: {exc}"


def _build_wiki_server(wiki_server: Any | None = None) -> tuple[Any | None, str | None]:
    """Load the local Wiki index as a resident evidence fallback."""

    if wiki_server is not None:
        return wiki_server, None
    try:
        from tools.league_wiki_mcp.server import LeagueWikiServer

        return LeagueWikiServer(), None
    except Exception as exc:  # pragma: no cover - local evidence is optional
        return None, f"Wiki fallback unavailable: {exc}"


class LeagueMechanicsServer:
    """MCP dispatcher holding one engine and one compiled pack in memory."""

    def __init__(
        self,
        pack: Any | None = None,
        *,
        engine: Any | None = None,
        fastpack_path: str | os.PathLike[str] | None = None,
        index_path: str | os.PathLike[str] | None = None,
        packet_root: str | os.PathLike[str] | None = None,
        engine_class: type[Any] | Callable[[Any], Any] | None = None,
        engine_factory: type[Any] | Callable[[Any], Any] | None = None,
        compiler: Callable[[Path], Any] | None = None,
        loader: Callable[[Path], Any] | None = None,
        oracle_engine: Any | None = None,
        semantic_engine: Any | None = None,
        router: Any | None = None,
        wiki_server: Any | None = None,
        warehouse: Any | None = None,
    ) -> None:
        self._answers = 0
        self._engine = engine
        self._pack = pack
        self._router = router
        self._router_error: str | None = None
        self._oracle_engine = oracle_engine
        self._semantic_engine = semantic_engine
        self._wiki_server = wiki_server
        self._wiki_error: str | None = None
        self._warehouse = warehouse
        self._warehouse_error: str | None = None
        self._fastpack_path: Path | None = None
        self._index_path: Path | None = (
            Path(index_path).expanduser().resolve() if index_path is not None else None
        )
        self._compile_result: Any = None
        self._configured_fastpack = fastpack_path is not None or bool(os.environ.get(FASTPACK_ENV))

        if self._engine is not None:
            # An injected engine is the fastest and cleanest test path.  Keep
            # the optional injected pack for status/provenance only.
            if fastpack_path is not None:
                self._fastpack_path = Path(fastpack_path).expanduser().resolve()
            if self._router is None and self._semantic_engine is not None:
                self._router, self._router_error = _build_oracle_router(
                    pack=self._pack,
                    index_path=self._index_path,
                    oracle_engine=self._oracle_engine,
                    semantic_engine=self._semantic_engine,
                )
            if self._wiki_server is None and self._router is not None:
                self._wiki_server, self._wiki_error = _build_wiki_server()
            if self._warehouse is None:
                self._warehouse, self._warehouse_error = _build_state_warehouse(
                    pack=self._pack,
                    index_path=self._index_path,
                    oracle_engine=self._oracle_engine,
                    router=self._router,
                )
            return

        # A constructor/CLI path is explicit configuration too; it takes
        # precedence over the environment so a caller can select a pack
        # without mutating process-wide state.
        configured = (
            os.fspath(fastpack_path)
            if fastpack_path is not None
            else os.environ.get(FASTPACK_ENV)
        )
        configured_path = Path(configured).expanduser().resolve() if configured else None

        # Resolve only what this startup path needs.  In particular, loading
        # an explicitly configured pack should not require importing a compiler
        # that a dependency-injected test does not provide.
        engine_type = engine_class or engine_factory or _component("QuickMechanicsEngine")
        compile_fn = compiler
        load_fn = loader
        if self._pack is None:
            if load_fn is None:
                load_fn = _component("load_fastpack")
            if configured_path is None and compile_fn is None:
                compile_fn = _component("compile_fastpack")
            if configured_path is not None:
                # Explicit configuration is fail-closed: if it is missing or
                # malformed, startup fails instead of selecting an older patch.
                source_index = None
            else:
                source_index = (
                    Path(index_path).expanduser().resolve()
                    if index_path is not None
                    else _discover_index(
                        Path(packet_root).expanduser().resolve()
                        if packet_root is not None
                        else DEFAULT_PACKET_ROOT
                    )
                )
            try:
                self._pack, self._fastpack_path, self._compile_result = _compile_or_load(
                    source_index or Path("<configured-fastpack>"),
                    compiler=compile_fn,  # type: ignore[arg-type]
                    loader=load_fn,  # type: ignore[arg-type]
                    configured_path=configured_path,
                )
            except Exception as exc:
                source = str(configured_path or source_index or "<unknown>")
                raise StartupError(f"unable to initialize exact mechanics pack {source}: {exc}") from exc
            self._index_path = source_index
        if self._pack is not None:
            try:
                self._engine = engine_type(self._pack)
            except Exception as exc:
                raise StartupError(f"unable to initialize QuickMechanicsEngine: {exc}") from exc
        if self._engine is None:
            raise StartupError("mechanics engine is unavailable")
        if self._router is None:
            self._router, self._router_error = _build_oracle_router(
                pack=self._pack,
                index_path=self._index_path,
                oracle_engine=self._oracle_engine,
                semantic_engine=self._semantic_engine,
            )
        if self._wiki_server is None:
            self._wiki_server, self._wiki_error = _build_wiki_server()
        if self._warehouse is None:
            self._warehouse, self._warehouse_error = _build_state_warehouse(
                pack=self._pack,
                index_path=self._index_path,
                oracle_engine=self._oracle_engine,
                router=self._router,
            )

    @property
    def engine(self) -> Any:
        return self._engine

    @property
    def pack(self) -> Any:
        return self._pack

    @property
    def warehouse(self) -> Any:
        """Return the resident warehouse, or ``None`` when startup lacked one."""

        return self._warehouse

    def close(self) -> None:
        """Release optional engine resources; resident fakes need no close."""

        close = getattr(self._engine, "close", None)
        if callable(close):
            close()
        wiki_close = getattr(self._wiki_server, "close", None)
        if callable(wiki_close):
            wiki_close()
        warehouse_close = getattr(self._warehouse, "close", None)
        if callable(warehouse_close):
            warehouse_close()

    def _warehouse_request(self, arguments: Mapping[str, Any]) -> Any | None:
        """Handle explicit warehouse operations without a second model path."""

        warehouse = self._warehouse
        if warehouse is None:
            requested = any(
                key in arguments
                for key in ("sql", "warehouse_sql", "state", "warehouse_state", "operation")
            )
            if requested:
                raise ToolError(self._warehouse_error or "resident state warehouse is unavailable")
            return None
        operation = str(arguments.get("operation") or "").casefold()
        sql = arguments.get("sql", arguments.get("warehouse_sql"))
        if sql is None and isinstance(arguments.get("context"), Mapping):
            context = arguments["context"]
            sql = context.get("sql", context.get("warehouse_sql"))
        if sql is None:
            question = arguments.get("question")
            if isinstance(question, str) and re.match(r"^(?:select|with)\b", question.strip(), re.I):
                sql = question
        if sql is not None or operation in {"sql", "query", "warehouse_sql"}:
            if not isinstance(sql, str) or not sql.strip():
                raise ToolError("warehouse SQL operation requires a non-empty sql string")
            max_rows = arguments.get("max_rows", DEFAULT_WAREHOUSE_MAX_ROWS)
            try:
                return warehouse.query_sql(sql, max_rows=max_rows)
            except Exception as exc:
                raise ToolError(str(exc)) from exc
        if operation in {"schema", "describe", "warehouse_schema"}:
            return warehouse.schema()
        if operation in {"status", "warehouse_status"}:
            return warehouse.status()
        state = arguments.get("state", arguments.get("warehouse_state"))
        if state is None and isinstance(arguments.get("context"), Mapping):
            context = arguments["context"]
            state = context.get("state", context.get("warehouse_state"))
        if state is not None:
            try:
                return warehouse.state_query(state)
            except Exception as exc:
                raise ToolError(str(exc)) from exc
        return None

    def answer(self, arguments: Mapping[str, Any] | str) -> Any:
        mapping = arguments if isinstance(arguments, Mapping) else {"question": arguments}
        if not isinstance(mapping, Mapping):
            raise ToolError("arguments must be an object")
        warehouse_answer = self._warehouse_request(mapping)
        if warehouse_answer is not None:
            self._answers += 1
            return _jsonable(warehouse_answer)
        question = mapping.get("question")
        if not isinstance(question, str) or not question.strip():
            raise ToolError("question must be a non-empty string")
        context = None
        if "context" in mapping:
            context = mapping.get("context")
            if not isinstance(context, Mapping):
                raise ToolError("context must be an object when supplied")
        if self._router is not None:
            answer = self._router.answer(question, context)
        else:
            if context is not None:
                raise ToolError(
                    "semantic router is unavailable; context-bearing requests cannot use the legacy exact engine"
                )
            answer = self._engine.answer(question)
        # The exact packet and semantic contract stay first.  A closed
        # champion/item state that neither legacy path recognizes can still be
        # answered by the resident normalized warehouse without starting a
        # second process or asking the caller to translate it into SQL.
        if (
            self._warehouse is not None
            and isinstance(answer, Mapping)
            and answer.get("status") in {"unsupported", "needs_input"}
            and context is None
        ):
            try:
                warehouse_candidate = self._warehouse.natural_query(question)
            except Exception:
                warehouse_candidate = None
            if isinstance(warehouse_candidate, Mapping):
                if warehouse_candidate.get("status") == "available":
                    answer = dict(warehouse_candidate)
                    answer["route"] = "warehouse_state"
                    answer["fallback_from"] = answer.get("fallback_from", "exact_or_semantic")
                elif warehouse_candidate.get("status") == "unsupported":
                    # Keep the compact Wiki fallback route for display, but
                    # expose the typed warehouse blocker so the caller can
                    # see exactly which state effect prevented a number.
                    answer = dict(answer)
                    answer["warehouse_state"] = dict(warehouse_candidate)
        if (
            self._wiki_server is not None
            and isinstance(answer, Mapping)
            and answer.get("status") == "unsupported"
        ):
            # Evidence is a fallback, never a numeric substitute.  Preserve
            # the semantic/exact blocker and attach compact revision-backed
            # Wiki evidence under a distinct key and route.
            try:
                evidence = self._wiki_server.query(
                    {"question": question, "response_mode": "fast"}
                )
                answer = dict(answer)
                answer["wiki_evidence"] = evidence
                answer["route"] = "wiki_evidence"
                from lol_kills.knowledge.oracle_router import ROUTER_VERSION

                answer["router"] = ROUTER_VERSION
                provenance = dict(answer.get("provenance") or {})
                provenance["route"] = "wiki_evidence"
                answer["provenance"] = provenance
            except Exception as exc:
                # Keep the original unavailable result; Wiki errors must not
                # turn a blocked calculation into a guessed answer.
                answer = dict(answer)
                answer["wiki_fallback_error"] = str(exc)
        self._answers += 1
        # Keep the engine's ready-to-display object as the sole answer source.
        return _jsonable(answer)

    def status(self, _arguments: Mapping[str, Any] | None = None) -> dict[str, Any]:
        pack = self._pack
        patch = None
        for candidate in (pack, self._compile_result):
            if isinstance(candidate, Mapping):
                patch = candidate.get("patch") or candidate.get("public_patch")
            else:
                patch = getattr(candidate, "patch", None)
            if patch is not None:
                break
        return {
            "status": "ready" if self._engine is not None else "unavailable",
            "server": SERVER_NAME,
            "server_version": SERVER_VERSION,
            "resident": self._engine is not None,
            "engine_loaded": self._engine is not None,
            "pack_loaded": pack is not None,
            "engine": type(self._engine).__name__ if self._engine is not None else None,
            "pack_type": type(pack).__name__ if pack is not None else None,
            "oracle_router_loaded": self._router is not None,
            "oracle_router": type(self._router).__name__ if self._router is not None else None,
            "oracle_engine_loaded": getattr(self._router, "exact_engine", None) is not None,
            "semantic_engine_loaded": getattr(self._router, "semantic_engine", None) is not None,
            "router_error": self._router_error,
            "wiki_fallback_loaded": self._wiki_server is not None,
            "wiki_fallback": type(self._wiki_server).__name__ if self._wiki_server is not None else None,
            "wiki_error": self._wiki_error,
            "state_warehouse_loaded": self._warehouse is not None,
            "state_warehouse": type(self._warehouse).__name__ if self._warehouse is not None else None,
            "state_warehouse_error": self._warehouse_error,
            "warehouse": self._warehouse.status() if self._warehouse is not None else None,
            "patch": patch,
            "fastpack_path": str(self._fastpack_path) if self._fastpack_path else None,
            "index_path": str(self._index_path) if self._index_path else None,
            "configured_fastpack": self._configured_fastpack,
            "answers_served": self._answers,
            "network_calls": 0,
            "answer_contract": {
                "engine_result_is_ready_to_display": True,
                "do_not_recalculate": True,
                "preserve_assumptions_and_unknowns": True,
                "canonical_answer_tool": "league_oracle_answer",
                "routing_order": ["exact_packet", "semantic_contract_or_execution", "wiki_evidence", "exact_unsupported"],
                "route_metadata_required": True,
                "warehouse_operations": ["sql", "schema", "state", "status"],
                "warehouse_sql": "read-only SELECT/WITH over the resident patch snapshot; bounded to 1000 rows",
            },
        }

    def call_tool(self, name: str, arguments: Mapping[str, Any]) -> Any:
        if name in {"league_oracle_answer", "league_mechanics_answer"}:
            return self.answer(arguments)
        if name in {"league_oracle_sql", "league_warehouse_query", "league_state_query"}:
            if not isinstance(arguments, Mapping):
                raise ToolError("warehouse arguments must be an object")
            if name == "league_oracle_sql":
                payload = dict(arguments)
                payload.setdefault("operation", "sql")
                return self.answer(payload)
            if name == "league_state_query":
                payload = dict(arguments)
                payload.setdefault("operation", "state")
                return self.answer(payload)
            return self.answer(arguments)
        if name in {"league_oracle_status", "league_mechanics_status"}:
            return self.status(arguments)
        raise ToolError(f"unknown tool: {name}")


# Friendly alias for callers that used the shorter name in early prototypes.
MechanicsServer = LeagueMechanicsServer
MechanicsMCPServer = LeagueMechanicsServer


def _tool_specs() -> list[dict[str, Any]]:
    return [
        {
            "name": "league_oracle_answer",
            "title": "Answer League oracle question",
            "description": (
                "The single canonical League answer path. It routes exact questions "
                "through the resident patch-pinned packet, then routes underspecified "
                "or counterfactual questions through the resident semantic state "
                "engine. It also accepts bounded operation=sql/schema/state/status "
                "requests against the resident patch warehouse. Preserve status, "
                "route, patch, assumptions, provenance, and sources; never recalculate "
                "or choose a slower alternate path."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "A League of Legends mechanics or interaction question.",
                    },
                    "context": {
                        "type": "object",
                        "description": "Optional explicit semantic state; do not invent missing fields.",
                    },
                    "operation": {
                        "type": "string",
                        "enum": ["sql", "schema", "state", "status"],
                        "description": "Optional resident-warehouse operation. Omit for a normal natural-language answer.",
                    },
                    "sql": {
                        "type": "string",
                        "description": "One bounded read-only SELECT/WITH statement over the warehouse schema.",
                    },
                    "max_rows": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 1000,
                        "default": DEFAULT_WAREHOUSE_MAX_ROWS,
                    },
                    "state": {
                        "type": "object",
                        "description": "Structured champion state for static stat composition; non-empty unsupported effects are reported, never ignored.",
                    },
                },
                "required": [],
                "anyOf": [
                    {"required": ["question"]},
                    {"required": ["operation"]},
                    {"required": ["sql"]},
                    {"required": ["state"]},
                ],
                "additionalProperties": False,
            },
        },
        {
            "name": "league_oracle_status",
            "title": "League oracle status",
            "description": "Report resident exact/semantic router, patch, and route contract status.",
            "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    ]


def _error_response(request_id: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


def dispatch(server: LeagueMechanicsServer, message: Mapping[str, Any]) -> dict[str, Any] | None:
    """Dispatch one JSON-RPC/MCP message; return ``None`` for notifications."""

    if not isinstance(message, Mapping):
        return _error_response(None, -32600, "JSON-RPC message must be an object")
    method = message.get("method")
    request_id = message.get("id")
    notification = "id" not in message
    if method == "notifications/initialized":
        return None
    if method == "initialize":
        params = message.get("params") or {}
        requested = params.get("protocolVersion") if isinstance(params, Mapping) else None
        version = requested if requested in SUPPORTED_PROTOCOL_VERSIONS else SUPPORTED_PROTOCOL_VERSION
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "protocolVersion": version,
                "capabilities": {"tools": {}},
                "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
                "instructions": (
                    "Use league_oracle_answer for every League question. It is the single "
                    "resident route: exact packet mechanics first, semantic state routing "
                    "second. Its result is ready to display: do not recalculate it. Preserve "
                    "status, route, patch, assumptions, provenance, sources, and unavailable "
                    "state; this server performs no network calls. For direct warehouse work, "
                    "pass operation=sql/schema/state/status (or call the compatibility aliases); "
                    "SQL is read-only SELECT/WITH against the resident patch snapshot."
                ),
            },
        }
    if method == "ping":
        return None if notification else {"jsonrpc": "2.0", "id": request_id, "result": {}}
    if method == "tools/list":
        return None if notification else {"jsonrpc": "2.0", "id": request_id, "result": {"tools": _tool_specs()}}
    if method == "tools/call":
        if notification:
            return None
        params = message.get("params") or {}
        if not isinstance(params, Mapping):
            return _error_response(request_id, -32602, "tools/call params must be an object")
        name = params.get("name")
        arguments = params.get("arguments") or {}
        if not isinstance(name, str) or not isinstance(arguments, Mapping):
            return _error_response(request_id, -32602, "tools/call requires a tool name and object arguments")
        try:
            result = server.call_tool(name, arguments)
            result = _jsonable(result)
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "content": [{"type": "text", "text": _json(result)}],
                    "structuredContent": result,
                },
            }
        except Exception as exc:  # keep a persistent MCP process alive on one bad call
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {"isError": True, "content": [{"type": "text", "text": str(exc)}]},
            }
    if notification:
        return None
    return _error_response(request_id, -32601, f"method not found: {method}")


def _build_server(args: argparse.Namespace) -> LeagueMechanicsServer:
    return LeagueMechanicsServer(
        fastpack_path=args.fastpack,
        index_path=args.index,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--question", help="answer one question and exit")
    parser.add_argument("--fastpack", help=f"explicit fastpack path (or set {FASTPACK_ENV})")
    parser.add_argument("--index", help="explicit local mechanics-index.json for compilation")
    args = parser.parse_args(argv)
    try:
        server = _build_server(args)
    except Exception as exc:  # pragma: no cover - startup diagnostics
        print(f"{SERVER_NAME}: startup failed: {exc}", file=sys.stderr, flush=True)
        return 1
    try:
        if args.question is not None:
            try:
                print(_json(server.answer({"question": args.question})), flush=True)
                return 0
            except Exception as exc:
                print(f"{SERVER_NAME}: {exc}", file=sys.stderr, flush=True)
                return 2
        for line in sys.stdin:
            if not line.strip():
                continue
            try:
                message = json.loads(line)
                if not isinstance(message, dict):
                    raise ValueError("JSON-RPC message must be an object")
                response = dispatch(server, message)
            except (json.JSONDecodeError, ValueError) as exc:
                response = _error_response(None, -32700, str(exc))
            if response is not None:
                print(json.dumps(response, ensure_ascii=False, separators=(",", ":")), flush=True)
    finally:
        server.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
