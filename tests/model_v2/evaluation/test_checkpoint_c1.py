from __future__ import annotations

import ast
import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

import lol_kills.v2.evaluation.checkpoint_c1 as c1
from lol_kills.v2.evaluation.checkpoint_c1 import (
    AUTHORITY_LOCATOR,
    CLAIM_BOUNDARY,
    CONFIG_LOCATOR,
    INPUT_ROLE_LOCATORS,
    REPORT_LOCATOR,
    REQUIRED_GATES,
    build_checkpoint_c1_bundle,
    canonical_json_bytes,
    load_checkpoint_c1,
)
from lol_kills.v2.evaluation.checks import ValidationFailure
from lol_kills.v2.evaluation.generate_checkpoint_c1_artifacts import generate


ROOT = Path(__file__).resolve().parents[3]
OUTPUTS = (CONFIG_LOCATOR, REPORT_LOCATOR, AUTHORITY_LOCATOR)
SOURCES = (
    c1.CHECKPOINT_SOURCE_LOCATOR,
    c1.GENERATOR_SOURCE_LOCATOR,
)


def _copy_root(tmp_path: Path) -> Path:
    target = tmp_path / "repo"
    for locator in (*INPUT_ROLE_LOCATORS, *SOURCES, *OUTPUTS):
        source = ROOT / locator
        destination = target / locator
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    return target


def _objects(root: Path) -> tuple[dict, dict, dict]:
    return tuple(json.loads((root / locator).read_bytes()) for locator in OUTPUTS)  # type: ignore[return-value]


def _write_chain(root: Path, config: dict, report: dict, authority: dict) -> None:
    config_raw = canonical_json_bytes(config)
    report["config_ref"] = {
        "locator": CONFIG_LOCATOR,
        "raw_sha256": hashlib.sha256(config_raw).hexdigest(),
        "object_sha256": c1.canonical_sha256(config),
    }
    report_raw = canonical_json_bytes(report)
    authority["config_ref"] = dict(report["config_ref"])
    authority["report_ref"] = {
        "locator": REPORT_LOCATOR,
        "raw_sha256": hashlib.sha256(report_raw).hexdigest(),
        "object_sha256": c1.canonical_sha256(report),
    }
    (root / CONFIG_LOCATOR).write_bytes(config_raw)
    (root / REPORT_LOCATOR).write_bytes(report_raw)
    (root / AUTHORITY_LOCATOR).write_bytes(canonical_json_bytes(authority))


def test_exact_bundle_loads_with_read_only_projection_and_narrow_semantics() -> None:
    authority = load_checkpoint_c1(ROOT)
    payload = authority.authenticate()
    assert payload["decision_kind"] == "foundation_freeze"
    assert payload["authority_scope"] == "wave_1_foundation_freeze_only"
    assert dict(payload["claim_boundary"]) == dict(CLAIM_BOUNDARY)
    assert set(payload["gate_set"]) == set(REQUIRED_GATES)
    assert all(value is True for value in payload["gate_set"].values())
    assert payload["threat_model"]["hostile_same_process_security"] is False
    assert payload["threat_model"]["singleton_and_content_hashes_authorize_promotion"] is False
    with pytest.raises(TypeError):
        payload["gate_set"]["forged"] = True
    with pytest.raises(AttributeError):
        authority._root = ROOT
    config, _, _ = _objects(ROOT)
    assert len(config["input_roles"]) == 26
    schema_role = next(
        role for role in config["input_roles"] if role["role"] == "l3_schema_implementation"
    )
    assert schema_role["raw_sha256"] == "8e7de9d10b6e9b3ca7945ecc4031b12ffc0538b0eb290d92625822b9028c7e72"


def test_report_accept_is_only_foundation_freeze_and_not_decision_accept() -> None:
    _, report, authority = _objects(ROOT)
    assert report["status"] == "ACCEPT"
    assert report["decision_kind"] == "foundation_freeze"
    assert report["acceptance_scope"].endswith("downstream_wave_2_work_only")
    assert "decision" not in report
    assert "status" not in authority
    assert all(value is False for key, value in report["claim_boundary"].items() if key != "promotion_decision")
    assert report["claim_boundary"]["promotion_decision"] is None


def test_exact_gate_set_and_independent_evidence_hashes() -> None:
    _, report, authority = _objects(ROOT)
    assert set(report["gates"]) == set(REQUIRED_GATES)
    assert set(authority["gate_set"]) == set(REQUIRED_GATES)
    hashes = []
    for gate, entry in report["gates"].items():
        assert entry["passed"] is True
        assert entry["evidence_sha256"] == c1.canonical_sha256(entry["evidence"])
        hashes.append(entry["evidence_sha256"])
    assert len(set(hashes)) == len(REQUIRED_GATES)


@pytest.mark.parametrize("locator", INPUT_ROLE_LOCATORS)
def test_one_byte_mutation_of_every_input_role_fails(tmp_path: Path, locator: str) -> None:
    root = _copy_root(tmp_path)
    path = root / locator
    raw = bytearray(path.read_bytes())
    raw[len(raw) // 2] ^= 1
    path.write_bytes(raw)
    with pytest.raises(ValidationFailure):
        load_checkpoint_c1(root)


@pytest.mark.parametrize("locator", OUTPUTS)
def test_one_byte_mutation_of_every_artifact_layer_fails(tmp_path: Path, locator: str) -> None:
    root = _copy_root(tmp_path)
    path = root / locator
    raw = bytearray(path.read_bytes())
    raw[len(raw) // 2] ^= 1
    path.write_bytes(raw)
    with pytest.raises(ValidationFailure):
        load_checkpoint_c1(root)


@pytest.mark.parametrize("field,value", [
    ("pass_b2", True),
    ("production_authority", True),
    ("real_data_evidence", True),
    ("reliability_authorized", True),
    ("probability_wording_authorized", True),
    ("publication_authorized", True),
    ("sota_authorized", True),
    ("sealed_decision_opened", True),
    ("promotion_decision", "ACCEPT"),
])
def test_claim_elevation_fails_even_when_caller_rehashes_chain(
    tmp_path: Path, field: str, value: object
) -> None:
    root = _copy_root(tmp_path)
    config, report, authority = _objects(root)
    config["claim_boundary"][field] = value
    report["claim_boundary"][field] = value
    authority["claim_boundary"][field] = value
    _write_chain(root, config, report, authority)
    with pytest.raises(ValidationFailure):
        load_checkpoint_c1(root)


@pytest.mark.parametrize("mode", ["missing", "extra", "renamed", "truthy"])
def test_claim_boundary_shape_and_type_are_exact(tmp_path: Path, mode: str) -> None:
    root = _copy_root(tmp_path)
    config, report, authority = _objects(root)
    for obj in (config, report, authority):
        boundary = obj["claim_boundary"]
        if mode == "missing":
            boundary.pop("pass_b2")
        elif mode == "extra":
            boundary["pass_b2_alias"] = False
        elif mode == "renamed":
            boundary["pass-b2"] = boundary.pop("pass_b2")
        else:
            boundary["pass_b2"] = 0
    _write_chain(root, config, report, authority)
    with pytest.raises(ValidationFailure):
        load_checkpoint_c1(root)


@pytest.mark.parametrize("mode", ["false", "missing", "extra", "self_rehashed"])
def test_gate_set_cannot_be_changed(tmp_path: Path, mode: str) -> None:
    root = _copy_root(tmp_path)
    config, report, authority = _objects(root)
    gate = REQUIRED_GATES[0]
    if mode == "false":
        report["gates"][gate]["passed"] = False
    elif mode == "missing":
        report["gates"].pop(gate)
    elif mode == "extra":
        report["gates"]["C1_GATE_ALIAS"] = report["gates"][gate]
    else:
        report["gates"][gate]["evidence"]["contract_tree_sha256"] = "0" * 64
        report["gates"][gate]["evidence_sha256"] = c1.canonical_sha256(report["gates"][gate]["evidence"])
    _write_chain(root, config, report, authority)
    with pytest.raises(ValidationFailure):
        load_checkpoint_c1(root)


@pytest.mark.parametrize("mode", ["raw_swap", "id_swap", "schema_swap", "locator_alias", "role_reorder", "role_missing", "role_extra"])
def test_role_and_artifact_substitution_fails_after_rehash(
    tmp_path: Path, mode: str
) -> None:
    root = _copy_root(tmp_path)
    config, report, authority = _objects(root)
    roles = config["input_roles"]
    if mode == "raw_swap":
        roles[0]["raw_sha256"] = roles[1]["raw_sha256"]
    elif mode == "id_swap":
        config["artifact_id"] = "scryglass:c1:foundation-freeze-report:v1"
    elif mode == "schema_swap":
        config["schema_version"] = "checkpoint-c1-foundation-freeze-report-v1"
    elif mode == "locator_alias":
        roles[0]["locator"] = "data/lol/v2/snapshots/b1/../b1/source-snapshot-passb1.json"
    elif mode == "role_reorder":
        roles[0], roles[1] = roles[1], roles[0]
    elif mode == "role_missing":
        roles.pop()
    else:
        roles.append(dict(roles[-1]))
    _write_chain(root, config, report, authority)
    with pytest.raises(ValidationFailure):
        load_checkpoint_c1(root)


def test_noncanonical_duplicate_key_and_nonfinite_artifacts_fail(tmp_path: Path) -> None:
    for raw in (
        b'{\"artifact_id\":\"a\",\"artifact_id\":\"b\"}',
        b'{\"x\":NaN}',
        b'{ \"x\": 1 }',
    ):
        root = _copy_root(tmp_path / hashlib.sha256(raw).hexdigest())
        (root / CONFIG_LOCATOR).write_bytes(raw)
        with pytest.raises(ValidationFailure):
            load_checkpoint_c1(root)


def test_constructor_and_object_new_forgery_fail() -> None:
    with pytest.raises(TypeError):
        c1.CheckpointC1Authority()
    forged = object.__new__(c1.CheckpointC1Authority)
    with pytest.raises(ValidationFailure):
        forged.authenticate()


def test_no_exposed_issuance_token_registry_or_copyable_state() -> None:
    assert not hasattr(c1, "_ISSUE_TOKEN")
    assert not hasattr(c1, "_ISSUED")
    assert not hasattr(c1, "_AUTHORITY_GUARD_CELL")
    authority = load_checkpoint_c1(ROOT)
    forged = object.__new__(c1.CheckpointC1Authority)
    for name in ("_root", "_token"):
        with pytest.raises(AttributeError):
            object.__setattr__(forged, name, getattr(authority, name, ROOT))
    with pytest.raises(ValidationFailure):
        forged.authenticate()
    with pytest.raises(TypeError):
        class ForgedSubclass(c1.CheckpointC1Authority):
            pass


def test_warm_load_then_payload_or_source_mutation_fails(tmp_path: Path) -> None:
    root = _copy_root(tmp_path)
    authority = load_checkpoint_c1(root)
    config, report, forged_authority = _objects(root)
    forged_authority["gate_set"][REQUIRED_GATES[0]] = False
    (root / AUTHORITY_LOCATOR).write_bytes(canonical_json_bytes(forged_authority))
    with pytest.raises(ValidationFailure):
        authority.authenticate()
    (root / AUTHORITY_LOCATOR).write_bytes(build_checkpoint_c1_bundle(root)[AUTHORITY_LOCATOR])
    source = root / c1.CHECKPOINT_SOURCE_LOCATOR
    source.write_bytes(source.read_bytes() + b"\\n")
    with pytest.raises(ValidationFailure):
        authority.authenticate()


def test_returned_nested_projection_cannot_mutate() -> None:
    payload = load_checkpoint_c1(ROOT).payload
    with pytest.raises(TypeError):
        payload["claim_boundary"]["pass_b2"] = True
    with pytest.raises(TypeError):
        payload["source_code"]["checkpoint_implementation"]["raw_sha256"] = "0" * 64


def test_runtime_helper_substitution_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    authority = load_checkpoint_c1(ROOT)
    monkeypatch.setattr(c1, "_read_verified_inputs", lambda *args, **kwargs: ([], {}, {}))
    with pytest.raises(ValidationFailure, match="helper substitution"):
        authority.authenticate()


@pytest.mark.parametrize("_name", ["_freeze", "_evaluate_gates", "_validate_ref", "_source_refs"])
def test_each_trust_helper_rebinding_fails_closed(
    monkeypatch: pytest.MonkeyPatch, _name: str
) -> None:
    authority = load_checkpoint_c1(ROOT)
    monkeypatch.setattr(c1, _name, lambda *args, **kwargs: {})
    with pytest.raises(ValidationFailure, match="helper substitution"):
        authority.authenticate()


def test_runtime_defaults_and_kwdefaults_mutation_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority = load_checkpoint_c1(ROOT)
    monkeypatch.setattr(c1._materialize_once, "__kwdefaults__", {"observer": lambda value: value})
    with pytest.raises(ValidationFailure, match="helper mutation"):
        authority.authenticate()


def test_authority_method_mutation_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    authority = load_checkpoint_c1(ROOT)
    monkeypatch.setattr(c1.CheckpointC1Authority.authenticate, "__defaults__", ())
    with pytest.raises(ValidationFailure, match="authority method mutation"):
        authority.authenticate()


def test_loader_helper_rebinding_is_detected_by_issued_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority = load_checkpoint_c1(ROOT)
    monkeypatch.setattr(c1, "load_checkpoint_c1", lambda root=ROOT: object())
    with pytest.raises(ValidationFailure, match="helper substitution"):
        authority.authenticate()


def test_single_public_build_performs_measured_internal_replays(tmp_path: Path) -> None:
    root = _copy_root(tmp_path)
    invocations: list[str] = []
    bundle = build_checkpoint_c1_bundle(root, _observer=invocations.append)
    assert len(invocations) == 4
    assert len(set(invocations)) == 1
    for locator, raw in bundle.items():
        (root / locator).write_bytes(raw)
    loaded = load_checkpoint_c1(root).payload
    report = json.loads(bundle[REPORT_LOCATOR])
    proof = report["gates"]["C1_GATE_EXACT_FRESH_REPLAY"]["evidence"]
    assert proof["one_pass_materializations"] == len(invocations)
    assert proof["probe_materializations"] == 2
    assert proof["final_materializations"] == 2
    assert proof["probe_state_sha256"] == invocations[0]
    assert proof["final_state_sha256"] == invocations[-1]
    assert proof["probe_byte_identical"] is True
    assert proof["final_state_byte_identical"] is True
    assert proof["serialized_final_byte_identical"] is True
    assert loaded["decision_kind"] == "foundation_freeze"


def test_one_pass_materializer_monkeypatch_fails_before_false_replay_claim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = c1._materialize_once
    calls = 0

    def differing(*args, **kwargs):
        nonlocal calls
        calls += 1
        result = original(*args, **kwargs)
        if calls % 2:
            object.__setattr__(result, "state_bytes", result.state_bytes + b"x")
        return result

    monkeypatch.setattr(c1, "_materialize_once", differing)
    with pytest.raises(ValidationFailure, match="helper substitution"):
        build_checkpoint_c1_bundle(ROOT)
    assert calls == 0


def test_legacy_pointer_and_inner_synthetic_claims_are_not_inherited() -> None:
    _, report, _ = _objects(ROOT)
    evidence = report["gates"]["C1_GATE_LEGACY_REPORT_NONAUTHORITATIVE"]["evidence"]
    assert evidence["classification"] == "HISTORICAL_NONAUTHORITATIVE"
    assert evidence["stale_pointer_repaired"] is False
    assert evidence["inner_synthetic_reliability_inherited"] is False
    assert evidence["inner_probability_wording_inherited"] is False
    assert evidence["internal_report_sha256"] == "86a0629525d92fd9fd0db3c19c35504685b24bbbcb10902539cfeaa32e635c12"
    assert evidence["stale_frozen_pointer"] == "86a0629525d92fd9fd0db3c19c35504685b24bbbcb10902539cfeaa32e635c12"


@pytest.mark.parametrize(
    "locator,mutator",
    [
        ("data/lol/v2/evaluation/b2/b3-reliability-authority.json", lambda x: x.__setitem__("status", "authorized")),
        ("data/lol/v2/evaluation/b2/b3-reliability-authority.json", lambda x: x["production_authorities"].append("forged")),
        ("data/lol/v2/evaluation/b2/b3-reliability-authority.json", lambda x: x.__setitem__("synthetic_authority_allowed", True)),
        ("data/lol/v2/publication/c4-authority-registry-b2.json", lambda x: x["authorities"].append("forged")),
        ("data/lol/v2/publication/c4-authority-registry-b2.json", lambda x: x["approved_packets"].append("forged")),
        ("data/lol/v2/publication/artifact-allowlist-public-b2.json", lambda x: x["rows"].append({"forged": True})),
        ("data/lol/v2/publication/artifact-allowlist-authenticated-b2.json", lambda x: x["rows"].append({"forged": True})),
        ("data/lol/v2/publication/artifact-allowlist-private-b2.json", lambda x: x["rows"][0].__setitem__("effective_decision", "public")),
    ],
)
def test_negative_boundary_elevations_fail(
    tmp_path: Path, locator: str, mutator
) -> None:
    root = _copy_root(tmp_path)
    path = root / locator
    value = json.loads(path.read_bytes())
    mutator(value)
    path.write_bytes(canonical_json_bytes(value))
    with pytest.raises(ValidationFailure):
        load_checkpoint_c1(root)


def test_root_symlink_rejected(tmp_path: Path) -> None:
    real = _copy_root(tmp_path)
    alias = tmp_path / "root-alias"
    alias.symlink_to(real, target_is_directory=True)
    with pytest.raises(ValidationFailure, match="symlink"):
        load_checkpoint_c1(alias)


def test_parent_symlink_rejected(tmp_path: Path) -> None:
    root = _copy_root(tmp_path)
    parent = root / "data/lol/v2/publication"
    moved = tmp_path / "publication-real"
    parent.rename(moved)
    parent.symlink_to(moved, target_is_directory=True)
    with pytest.raises(ValidationFailure, match="symlink"):
        load_checkpoint_c1(root)


def test_leaf_symlink_rejected(tmp_path: Path) -> None:
    root = _copy_root(tmp_path)
    leaf = root / INPUT_ROLE_LOCATORS[0]
    moved = leaf.with_suffix(".real")
    leaf.rename(moved)
    leaf.symlink_to(moved)
    with pytest.raises(ValidationFailure, match="symlink"):
        load_checkpoint_c1(root)


def test_hardlink_leaf_rejected(tmp_path: Path) -> None:
    root = _copy_root(tmp_path)
    leaf = root / INPUT_ROLE_LOCATORS[0]
    backup = leaf.with_suffix(".hardlink")
    os.link(leaf, backup)
    with pytest.raises(ValidationFailure, match="hardlinked"):
        load_checkpoint_c1(root)


def test_nonregular_leaf_rejected(tmp_path: Path) -> None:
    root = _copy_root(tmp_path)
    leaf = root / INPUT_ROLE_LOCATORS[0]
    leaf.unlink()
    os.mkfifo(leaf)
    with pytest.raises(ValidationFailure, match="nonregular"):
        load_checkpoint_c1(root)


def test_path_escape_is_rejected_directly(tmp_path: Path) -> None:
    with pytest.raises(ValidationFailure, match="unsafe or aliased"):
        c1._safe_read(tmp_path, "../escape.json", seen_inodes={})


def test_generation_and_load_preserve_all_frozen_input_bytes(tmp_path: Path) -> None:
    root = _copy_root(tmp_path)
    before = {locator: (root / locator).read_bytes() for locator in INPUT_ROLE_LOCATORS}
    first = generate(root)
    first_bytes = {locator: (root / locator).read_bytes() for locator in OUTPUTS}
    second = generate(root)
    second_bytes = {locator: (root / locator).read_bytes() for locator in OUTPUTS}
    after = {locator: (root / locator).read_bytes() for locator in INPUT_ROLE_LOCATORS}
    assert before == after
    assert first == second
    assert first_bytes == second_bytes == build_checkpoint_c1_bundle(root)


@pytest.mark.parametrize("locator", OUTPUTS)
@pytest.mark.parametrize("attack", ["symlink", "hardlink", "directory", "fifo"])
def test_output_leaf_attacks_never_change_victim_or_sibling_outputs(
    tmp_path: Path, locator: str, attack: str
) -> None:
    root = _copy_root(tmp_path)
    target = root / locator
    siblings = {
        other: (root / other).read_bytes()
        for other in OUTPUTS
        if other != locator
    }
    original = target.read_bytes()
    target.unlink()
    victim = tmp_path / f"victim-{attack}-{target.name}"
    victim.write_bytes(b"external-victim-bytes")
    victim_before = victim.read_bytes()
    if attack == "symlink":
        target.symlink_to(victim)
    elif attack == "hardlink":
        os.link(victim, target)
    elif attack == "directory":
        target.mkdir()
    else:
        os.mkfifo(target)
    with pytest.raises(ValidationFailure):
        generate(root)
    assert victim.read_bytes() == victim_before
    assert {
        other: (root / other).read_bytes()
        for other in siblings
    } == siblings
    if attack == "symlink":
        assert target.is_symlink()
    elif attack == "hardlink":
        assert target.stat().st_ino == victim.stat().st_ino
    elif attack == "directory":
        assert target.is_dir()
    else:
        assert stat.S_ISFIFO(target.lstat().st_mode)
    assert original != victim_before


def test_generator_root_symlink_rejects_before_any_output_write(tmp_path: Path) -> None:
    real = _copy_root(tmp_path)
    before = {locator: (real / locator).read_bytes() for locator in OUTPUTS}
    alias = tmp_path / "generator-root-alias"
    alias.symlink_to(real, target_is_directory=True)
    with pytest.raises(ValidationFailure):
        generate(alias)
    assert {locator: (real / locator).read_bytes() for locator in OUTPUTS} == before


def test_generator_parent_symlink_rejects_before_any_output_write(tmp_path: Path) -> None:
    root = _copy_root(tmp_path)
    parent = root / "data/lol/v2/evaluation/b2"
    moved = tmp_path / "b2-real"
    parent.rename(moved)
    parent.symlink_to(moved, target_is_directory=True)
    before = {locator: (root / locator).read_bytes() for locator in OUTPUTS}
    with pytest.raises(ValidationFailure):
        generate(root)
    assert {locator: (root / locator).read_bytes() for locator in OUTPUTS} == before


def test_schema_implementation_mutation_rejects_build(tmp_path: Path) -> None:
    root = _copy_root(tmp_path)
    schema = root / "lol_kills/v2/champions/schema.py"
    schema.write_bytes(schema.read_bytes() + b"\\n")
    with pytest.raises(ValidationFailure):
        build_checkpoint_c1_bundle(root)


@pytest.mark.parametrize(
    "injection",
    [
        b"\\nimport promotion\\n",
        b"\\nsealed.open()\\n",
        b"\\nPath('unauthorized.json').write_text('x')\\n",
    ],
)
def test_owned_source_policy_rejects_forbidden_import_call_or_write(
    tmp_path: Path, injection: bytes
) -> None:
    root = _copy_root(tmp_path)
    checkpoint = root / c1.CHECKPOINT_SOURCE_LOCATOR
    checkpoint.write_bytes(checkpoint.read_bytes() + injection)
    with pytest.raises(ValidationFailure):
        build_checkpoint_c1_bundle(root)


def test_generator_source_policy_rejects_injected_authorized_api_at_wrong_surface(
    tmp_path: Path,
) -> None:
    root = _copy_root(tmp_path)
    generator = root / c1.GENERATOR_SOURCE_LOCATOR
    generator.write_bytes(generator.read_bytes() + b"\nos.replace('x', 'y')\n")
    with pytest.raises(ValidationFailure):
        build_checkpoint_c1_bundle(root)


def test_sealed_fixture_shape_and_rehash_forgery_reject(tmp_path: Path) -> None:
    root = _copy_root(tmp_path)
    sealed_locator = "data/lol/v2/evaluation/sealed-ledger-fixture.jsonl"
    (root / sealed_locator).write_bytes(
        b'{"fixture_version":1,"kind":"opened-sealed-ledger"}\\n'
    )
    config, report, authority = _objects(root)
    sealed_role = next(
        role for role in config["input_roles"] if role["role"] == "sealed_ledger_boundary"
    )
    sealed_role["raw_sha256"] = hashlib.sha256((root / sealed_locator).read_bytes()).hexdigest()
    _write_chain(root, config, report, authority)
    with pytest.raises(ValidationFailure):
        load_checkpoint_c1(root)


def test_two_fresh_interpreter_replays_are_byte_identical(tmp_path: Path) -> None:
    root = _copy_root(tmp_path)
    script = (
        "from pathlib import Path;"
        "from lol_kills.v2.evaluation.generate_checkpoint_c1_artifacts import generate;"
        f"generate(Path({str(root)!r}))"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT)
    subprocess.run([sys.executable, "-c", script], check=True, cwd=ROOT, env=env)
    first = {locator: (root / locator).read_bytes() for locator in OUTPUTS}
    subprocess.run([sys.executable, "-c", script], check=True, cwd=ROOT, env=env)
    second = {locator: (root / locator).read_bytes() for locator in OUTPUTS}
    assert first == second


def test_no_import_or_call_path_to_promotion_or_sealed_apis() -> None:
    forbidden_modules = {"promotion", "sealed"}
    forbidden_calls = {"request", "claim", "open", "execute", "finalize", "receipt"}
    for locator in SOURCES:
        tree = ast.parse((ROOT / locator).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert all(not any(part in alias.name.split(".") for part in forbidden_modules) for alias in node.names)
            if isinstance(node, ast.ImportFrom):
                assert node.module is None or not any(part in node.module.split(".") for part in forbidden_modules)
            if isinstance(node, ast.Call):
                name = node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id if isinstance(node.func, ast.Name) else ""
                if name in forbidden_calls:
                    assert (
                        isinstance(node.func, ast.Attribute)
                        and isinstance(node.func.value, ast.Name)
                        and node.func.value.id == "os"
                        and name == "open"
                    )
    checkpoint_text = (ROOT / c1.CHECKPOINT_SOURCE_LOCATOR).read_text(encoding="utf-8")
    assert "PromotionReport" not in checkpoint_text


def test_generator_output_allowlist_is_exact() -> None:
    tree = ast.parse((ROOT / c1.GENERATOR_SOURCE_LOCATOR).read_text(encoding="utf-8"))
    forbidden_fragments = ("__init__.py", "registry", "allowlist", "promotion", "publication")
    string_literals = {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    assert not any(
        fragment in value
        for value in string_literals
        for fragment in forbidden_fragments
    )
    assert set(build_checkpoint_c1_bundle(ROOT)) == set(OUTPUTS)
