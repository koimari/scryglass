from __future__ import annotations

from copy import deepcopy
from collections.abc import Mapping
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from lol_kills.v2.evaluation import b3_coverage as b3
from lol_kills.v2.evaluation.generate_b3_coverage_artifacts import generate


ROOT = Path(__file__).resolve().parents[3]


def _mutable(value):
    if isinstance(value, Mapping):
        return {key: _mutable(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_mutable(item) for item in value]
    if isinstance(value, frozenset):
        return {_mutable(item) for item in value}
    return value


@pytest.fixture(scope="module")
def root_state():
    authority = b3.load_b3_coverage_authority(ROOT)
    return authority, b3.snapshot_b3_coverage_authority(authority)


@pytest.fixture(scope="module")
def root_authority(root_state):
    return root_state[0]


@pytest.fixture(scope="module")
def root_payloads(root_state):
    return root_state[1]["payloads"]


@pytest.fixture(scope="session")
def bundle_template(tmp_path_factory: pytest.TempPathFactory) -> Path:
    template = tmp_path_factory.mktemp("b3-template")
    shutil.copytree(ROOT / "data/lol/v2/evaluation/b2", template / "data/lol/v2/evaluation/b2")
    (template / "lol_kills/v2/evaluation").mkdir(parents=True)
    shutil.copy2(
        ROOT / "lol_kills/v2/evaluation/generate_b3_coverage_artifacts.py",
        template / "lol_kills/v2/evaluation/generate_b3_coverage_artifacts.py",
    )
    generate(template)
    return template


@pytest.fixture()
def bundle(tmp_path: Path, bundle_template: Path) -> Path:
    shutil.copytree(bundle_template, tmp_path, dirs_exist_ok=True)
    return tmp_path


def _write(path: Path, value: dict) -> None:
    path.write_bytes(b3._canonical_bytes(value))


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def _mutate_and_reject(bundle: Path, name: str, mutate) -> None:
    path = bundle / "data/lol/v2/evaluation/b3" / name
    value = _load(path)
    mutate(value)
    _write(path, value)
    with pytest.raises(b3.B3CoverageError):
        b3._authenticate_bundle(bundle)


def test_frozen_bundle_loads_and_claim_ceiling_is_synthetic(
    root_authority, root_payloads
) -> None:
    b3.validate_b3_coverage_authority(root_authority)
    report = root_payloads["report"]
    assert report["mechanics_status"] == "PASS"
    assert report["real_coverage_status"].startswith("unavailable_")
    assert report["public_interval_wording"] == "95% model range"
    assert report["claim_ceiling"] == (
        "synthetic_sbc_coverage_dependence_mechanics_only",
    )
    assert "production_coverage" in report["forbidden_claims"]
    threat = report["authority_threat_model"]
    assert threat["honest_interpreter_required"]
    assert not threat["hostile_same_process_security"]
    assert threat["closure_cells_module_globals_and_class_code_are_mutable"]
    assert threat["content_revalidated_on_every_public_use"]
    assert threat["production_authority_requires"] == (
        "independently_pinned_signature_native_process_or_os_trust_boundary"
    )


def test_exact_regime_universe_and_all_strata(root_payloads) -> None:
    regimes = root_payloads["regimes"]["regimes"]
    assert len(regimes) == 20
    assert {item["output_type"] for item in regimes} == set(b3.OUTPUT_TYPES)
    assert {item["regime_kind"] for item in regimes} == set(b3.REGIME_KINDS)
    strata = root_payloads["report"]["aggregate_coverage"]["strata"]
    assert {(x["output_type"], x["frozen_stratum"]) for x in strata} == {
        (output, stratum)
        for output in b3.OUTPUT_TYPES
        for stratum in ("established", "sparse")
    }


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value["regimes"].pop(),
        lambda value: value["regimes"].__setitem__(0, value["regimes"][1]),
        lambda value: value["regimes"][0].__setitem__("regime_id", "substituted"),
    ],
)
def test_missing_duplicate_or_substituted_regime_rejects(bundle: Path, mutation) -> None:
    _mutate_and_reject(bundle, "regimes.json", mutation)


@pytest.mark.parametrize(
    ("name", "mutation"),
    [
        ("replications.json", lambda v: v["replications"][0]["observation"]["outcomes"].__setitem__(0, 9)),
        ("replications.json", lambda v: v["replications"][0].__setitem__("inference_input_sha256", "0" * 64)),
        ("replications.json", lambda v: v["replications"][0]["posterior_draws"].__setitem__(0, 0.5)),
        ("replications.json", lambda v: v["replications"][0].__setitem__("randomized_rank", 128)),
        ("replications.json", lambda v: v["replications"][0].__setitem__("tie_policy", "centre")),
        ("replications.json", lambda v: v["replications"][0]["posterior_support"].__setitem__("exact_ess", 9999.0)),
    ],
)
def test_simulation_lineage_rank_and_support_attacks_reject(bundle: Path, name, mutation) -> None:
    _mutate_and_reject(bundle, name, mutation)


def test_inference_adapter_rejects_latent_truth() -> None:
    regime = b3._regime_universe()[0]
    with pytest.raises(b3.B3CoverageError, match="forbidden"):
        b3._inference_adapter(
            regime,
            {"outcomes": [0] * regime["observation_count"], "latent_truth": 0.2},
            1,
            8,
        )


def test_controls_pass_and_fail_for_registered_reasons(root_payloads) -> None:
    controls = {x["control"]: x for x in root_payloads["report"]["controls"]}
    assert controls["known_good"]["passed"]
    for name in ("biased", "underdispersed", "overdispersed", "centre_ranked_degenerate"):
        assert not controls[name]["passed"]
        assert controls[name]["intended_control_rejected"]
        assert controls[name]["failure_reasons"]


def test_every_known_good_regime_passes_simultaneously(root_payloads) -> None:
    known_good = root_payloads["report"]["controls"][0]
    assert known_good["simultaneous_rule"] == (
        "pooled_and_all_20_regimes_pass_bonferroni_family_alpha_0.01"
    )
    assert known_good["passed"]
    assert known_good["all_regimes_pass"]
    assert known_good["simultaneous_family"] == {
        "family_alpha": 0.01,
        "family_size": 84,
        "per_test_alpha": 0.01 / 84.0,
        "datasets": 21,
        "diagnostics_per_dataset": 4,
        "diagnostics": (
            "rank_chi_square",
            "rank_ecdf",
            "interval_coverage",
            "rank_location",
        ),
    }
    assert len(known_good["per_regime"]) == 20
    assert all(item["passed"] for item in known_good["per_regime"])
    assert all(sum(item["rank_histogram"]) == 120 for item in known_good["per_regime"])


def test_faulty_controls_replay_posterior_adapters(root_payloads) -> None:
    controls = root_payloads["replications"]["control_replications"]
    assert {record["adapter_id"] for record in controls} == {
        f"frozen:{name}:v1" for name in b3.CONTROL_NAMES
    }
    assert all(len(record["posterior_draws"]) == 128 for record in controls)
    assert all(record["inference_input_sha256"] for record in controls)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda v: v["cells"][0]["row_ids"].pop(),
        lambda v: v["cells"][0]["joint_posterior_predictive"]["column_row_ids"].reverse(),
        lambda v: v["cells"][0]["joint_posterior_predictive"]["column_row_ids"].append(
            v["cells"][0]["joint_posterior_predictive"]["column_row_ids"][0]
        ),
        lambda v: v["cells"][0].__setitem__("aggregate_interval", [0.0, 1.0]),
        lambda v: v["cells"][0].__setitem__("aggregate_interval", [0.5, 0.5]),
    ],
)
def test_predictive_column_split_and_interval_attacks_reject(bundle: Path, mutation) -> None:
    _mutate_and_reject(bundle, "heldout-cells.json", mutation)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda v: v.__setitem__("map_resampling_allowed", True),
        lambda v: v.__setitem__("naive_series_iid_allowed", True),
        lambda v: v["levels"].pop(),
        lambda v: v["top_level_support"].__setitem__("kish_ess", 9999.0),
        lambda v: v["top_level_support"].__setitem__(
            "status", "unavailable_dependence_support"
        ),
        lambda v: v["leave_largest_cluster"].__setitem__(
            "mechanics_availability_stable", False
        ),
        lambda v: v["nested_resampling"].__setitem__(
            "global_series_times_component_product", True
        ),
        lambda v: v["unresolved_sensitivity"].__setitem__(
            "groups", {"fabricated-singleton": ["R060"]}
        ),
    ],
)
def test_dependence_attacks_reject(bundle: Path, mutation) -> None:
    _mutate_and_reject(bundle, "dependence.json", mutation)


@pytest.mark.parametrize(
    ("collapse", "failed_dimension"),
    [
        (
            lambda row: row.__setitem__(
                "participant_ids", ["P_COLLAPSED", row["participant_ids"][1]]
            ),
            "identity_component_id",
        ),
        (
            lambda row: row.__setitem__(
                "team_ids", ["TEAM_COLLAPSED", row["team_ids"][1]]
            ),
            "identity_component_id",
        ),
        (
            lambda row: row.__setitem__(
                "tournament_time_id", "COLLAPSED_TOURNAMENT_TIME"
            ),
            "tournament_time_id",
        ),
        (
            lambda row: row.__setitem__("patch_shock_id", "collapsed"),
            "patch_shock_id",
        ),
    ],
)
def test_collapsed_required_dimension_blocks_despite_large_series_support(
    root_payloads, collapse, failed_dimension
) -> None:
    rows = _mutable(root_payloads["heldout_rows"]["rows"])
    cells = root_payloads["heldout_cells"]["cells"]
    config = root_payloads["config"]
    for row in rows:
        if row["resolved"]:
            collapse(row)
    dependence = b3._dependence(rows, cells, config)
    assert dependence["top_level_support"]["status"] == (
        "unavailable_dependence_support"
    )
    assert failed_dimension in dependence["top_level_support"]["failed_dimensions"]
    series = next(
        level for level in dependence["levels"] if level["field"] == "series_id"
    )
    assert series["raw_cluster_count"] >= 60


def test_multiway_estimator_and_leave_largest_are_executed(root_payloads) -> None:
    dependence = root_payloads["dependence"]
    assert dependence["multiway_inference"]["replicate_count"] == 512
    assert dependence["multiway_inference"]["replicate_distribution"]
    assert not dependence["naive_iid_diagnostic"]["authoritative"]
    assert dependence["naive_iid_diagnostic"]["materiality_rule"] == (
        "descriptive_only_no_authority_gate"
    )
    assert dependence["naive_iid_diagnostic"]["mcse_sd_difference"] >= 0.0
    cases = dependence["leave_largest_cluster"]["cases"]
    assert len(cases) >= 3
    assert all(case["changed_cell_aggregate_count"] > 0 for case in cases)
    assert all(case["remaining_row_ids_sha256"] for case in cases)
    assert dependence["leave_largest_cluster"]["mechanics_availability_stable"]
    assert dependence["leave_largest_cluster"]["coverage_decision"].startswith(
        "unavailable_"
    )
    assert dependence["nested_resampling"] == {
        "higher_level": "identity_component_id",
        "inner_atomic_block": "series_id",
        "crossed_dimensions": ("tournament_time_id", "patch_shock_id"),
        "global_series_times_component_product": False,
    }


def test_bonferroni_adjusted_single_regime_failure_is_executable() -> None:
    config = b3._config()
    records = [
        {
            "randomized_rank": 0,
            "latent_truth": 0.5,
            "interval": [0.0, 0.9],
        }
        for _ in range(120)
    ]
    diagnostic = b3._diagnostics(records, config, "known_good")
    assert diagnostic["simultaneous_adjusted_alpha"] == pytest.approx(0.01 / 84.0)
    assert diagnostic["rank_chi_square_p_value"] < (
        diagnostic["simultaneous_adjusted_alpha"]
    )
    assert "rank_nonuniform_chi_square" in diagnostic["failure_reasons"]
    assert diagnostic["rank_chi_square_critical"] > 21.665994333461917


def test_finite_draw_interval_matches_exchangeable_rank_oracle() -> None:
    draws = [float(index) for index in range(128)]
    interval, rule = b3._finite_draw_interval(draws, 0.95)
    covered_ranks = list(
        range(
            rule["covered_rank_bounds"][0],
            rule["covered_rank_bounds"][1] + 1,
        )
    )
    assert interval == [2.0, 125.0]
    assert len(covered_ranks) == rule["covered_rank_count"] == 123
    assert rule["exact_finite_draw_coverage"] == pytest.approx(123 / 129)
    assert rule["exact_finite_draw_coverage"] >= 0.95


def test_actual_participant_and_team_identities_drive_dependence(root_payloads) -> None:
    rows = [row for row in root_payloads["heldout_rows"]["rows"] if row["resolved"]]
    dependence = root_payloads["dependence"]
    levels = {level["field"]: level for level in dependence["levels"]}
    assert "participant_team_id" not in levels
    assert levels["identity_component_id"]["raw_cluster_count"] == 6
    network = dependence["identity_network"]
    assert network["one_label_per_row_dimension"]
    assert network["component_count"] == 6
    for field, metric_name in (
        ("participant_ids", "participant_identity_support"),
        ("team_ids", "team_identity_support"),
    ):
        expected = {identity for row in rows for identity in row[field]}
        metrics = network[metric_name]
        assert metrics["raw_cluster_count"] == len(expected)
        assert metrics["membership_count"] == 2 * len(rows)
        assert metrics["consumed_identity_sha256"]


def test_bootstrap_recomputes_predictive_and_observed_aggregates(root_payloads) -> None:
    rows = [row for row in root_payloads["heldout_rows"]["rows"] if row["resolved"]]
    cell = root_payloads["heldout_cells"]["cells"][0]
    cell_rows = [row for row in rows if row["row_id"] in cell["row_ids"]]
    removed_series = cell_rows[0]["series_id"]
    weights = {
        row["row_id"]: 0.0 if row["series_id"] == removed_series else 1.0
        for row in rows
    }
    detail = b3._weighted_coverage_statistic(rows, [cell], weights)
    contribution = detail["cell_contributions"][0]
    assert contribution["weighted_predictive_draws_sha256"]
    assert contribution["weighted_predictive_interval"] != cell["aggregate_interval"]
    assert contribution["weighted_observed_aggregate"] != cell["observed_aggregate"]


def test_synthetic_coverage_performance_remains_unavailable(root_payloads) -> None:
    report = root_payloads["report"]
    assert report["synthetic_coverage_performance_status"].startswith("unavailable_")
    assert "coverage_decision_threshold" not in root_payloads["config"]
    assert root_payloads["dependence"]["multiway_inference"][
        "coverage_performance_decision"
    ].startswith("unavailable_")


def test_unresolved_outcome_before_resolution_rejects(bundle: Path) -> None:
    def mutate(value):
        unresolved = next(row for row in value["rows"] if not row["resolved"])
        unresolved["observed_outcome"] = True

    _mutate_and_reject(bundle, "heldout-rows.json", mutate)


def test_row_series_cell_aggregate_reconciliation(root_payloads) -> None:
    rows = root_payloads["heldout_rows"]["rows"]
    cells = root_payloads["heldout_cells"]["cells"]
    resolved = {row["row_id"] for row in rows if row["resolved"]}
    consumed = [row_id for cell in cells for row_id in cell["row_ids"]]
    assert set(consumed) == resolved
    assert len(consumed) == len(set(consumed))
    for cell in cells:
        for series_id in cell["series_ids"]:
            expected = {
                row["row_id"] for row in rows if row["resolved"] and row["series_id"] == series_id
            }
            assert expected <= set(cell["row_ids"])


def test_heldout_label_mutation_preserves_pre_outcome_predictive_bytes(
    root_payloads,
) -> None:
    cell = _mutable(root_payloads["heldout_cells"]["cells"][0])
    before = b3.predictive_bytes_hash(cell)
    old_covered = cell["covered"]
    cell["observed_aggregate"] = 1.0 - cell["observed_aggregate"]
    low, high = cell["aggregate_interval"]
    cell["covered"] = low <= cell["observed_aggregate"] <= high
    assert b3.predictive_bytes_hash(cell) == before
    assert cell["covered"] != old_covered or cell["observed_aggregate"] != 0.5


def test_noncanonical_duplicate_path_symlink_and_hardlink_reject(bundle: Path) -> None:
    config = bundle / "data/lol/v2/evaluation/b3/config.json"
    config.write_text(config.read_text() + "\n")
    with pytest.raises(b3.B3CoverageError):
        b3._authenticate_bundle(bundle)

    generate(bundle)
    authority_path = bundle / b3.AUTHORITY_LOCATOR
    authority = _load(authority_path)
    authority["artifacts"]["config"]["locator"] = "../b3/config.json"
    _write(authority_path, authority)
    with pytest.raises(b3.B3CoverageError):
        b3._authenticate_bundle(bundle)

    generate(bundle)
    config.unlink()
    config.symlink_to(bundle / "data/lol/v2/evaluation/b3/regimes.json")
    with pytest.raises(b3.B3CoverageError):
        b3._authenticate_bundle(bundle)

    config.unlink()
    generate(bundle)
    artifact_dir = bundle / "data/lol/v2/evaluation/b3"
    real_artifact_dir = artifact_dir.with_name("b3-real")
    artifact_dir.rename(real_artifact_dir)
    artifact_dir.symlink_to(real_artifact_dir, target_is_directory=True)
    with pytest.raises(b3.B3CoverageError, match="symlink artifact path"):
        b3._authenticate_bundle(bundle)
    artifact_dir.unlink()
    real_artifact_dir.rename(artifact_dir)

    config = artifact_dir / "config.json"
    os.link(config, bundle / "data/lol/v2/evaluation/b3/config-alias.json")
    with pytest.raises(b3.B3CoverageError):
        b3._authenticate_bundle(bundle)


def test_constructor_object_new_and_same_content_forgery_reject(
    root_authority,
) -> None:
    legitimate = root_authority
    with pytest.raises(TypeError, match="loader-issued"):
        b3.LoadedB3CoverageAuthority()
    forged = object.__new__(b3.LoadedB3CoverageAuthority)
    with pytest.raises(b3.B3CoverageError, match="not issued"):
        b3.validate_b3_coverage_authority(forged)
    assert not hasattr(b3, "_ISSUANCE_TOKEN")
    assert not hasattr(b3, "_ISSUED")
    with pytest.raises(TypeError, match="cannot be subclassed"):
        class ForgedSubclass(b3.LoadedB3CoverageAuthority):
            pass


def test_closure_cells_expose_no_forgeable_registry_or_token(root_authority) -> None:
    seen: set[int] = set()
    stack = [
        b3.load_b3_coverage_authority,
        b3.validate_b3_coverage_authority,
        b3.snapshot_b3_coverage_authority,
    ]
    recovered_capabilities = []
    while stack:
        function = stack.pop()
        if id(function) in seen:
            continue
        seen.add(id(function))
        for cell in function.__closure__ or ():
            value = cell.cell_contents
            if value is root_authority:
                recovered_capabilities.append(value)
            if isinstance(value, dict):
                assert value is b3.__dict__
                assert "_ISSUANCE_TOKEN" not in value
                assert "_ISSUED" not in value
            if callable(value) and hasattr(value, "__closure__"):
                stack.append(value)
    assert recovered_capabilities
    forged = object.__new__(b3.LoadedB3CoverageAuthority)
    with pytest.raises(b3.B3CoverageError, match="not issued"):
        b3.validate_b3_coverage_authority(forged)


def test_threat_model_acknowledges_writable_python_closure_cells() -> None:
    captured = "original"

    def reader():
        return captured

    cell = reader.__closure__[0]
    cell.cell_contents = "mutated"
    assert reader() == "mutated"


def test_snapshot_is_recursively_read_only_and_reauthenticates(root_state) -> None:
    authority, snapshot = root_state
    payloads = snapshot["payloads"]
    with pytest.raises(TypeError):
        payloads["config"]["master_seed"] = 1
    with pytest.raises(AttributeError):
        payloads["regimes"]["regimes"].append({})
    with pytest.raises(TypeError):
        payloads["regimes"]["regimes"][0]["regime_id"] = "forged"
    with pytest.raises(TypeError):
        payloads["report"]["hard_gates"][0]["evidence"]["forged"] = True
    with pytest.raises(TypeError):
        payloads["heldout_cells"]["cells"][0][
            "joint_posterior_predictive"
        ]["joint_draws"][0][0] = 9.0

    fresh = b3.snapshot_b3_coverage_authority(authority)
    assert fresh["payloads"]["config"]["master_seed"] == 730241
    assert fresh["payloads"]["regimes"]["regimes"][0]["regime_id"] != "forged"


def test_callable_rebind_code_defaults_and_registry_mutation_reject(monkeypatch) -> None:
    original = b3._inference_adapter
    monkeypatch.setattr(b3, "_inference_adapter", lambda *args: [])
    with pytest.raises(b3.B3CoverageError, match="rebound"):
        b3.load_b3_coverage_authority(ROOT)
    monkeypatch.setattr(b3, "_inference_adapter", original)

    original_code = original.__code__
    original.__code__ = (lambda *args: []).__code__
    try:
        with pytest.raises(b3.B3CoverageError, match="code/default"):
            b3.load_b3_coverage_authority(ROOT)
    finally:
        original.__code__ = original_code

    monkeypatch.setattr(b3, "OUTPUT_TYPES", b3.OUTPUT_TYPES[:-1])
    with pytest.raises(b3.B3CoverageError, match="registry"):
        b3.load_b3_coverage_authority(ROOT)


def test_comparator_defaults_kwdefaults_and_public_api_mutation_reject(monkeypatch) -> None:
    original_hash = b3._object_hash
    monkeypatch.setattr(b3, "_object_hash", lambda value: "0" * 64)
    with pytest.raises(b3.B3CoverageError, match="rebound"):
        b3.load_b3_coverage_authority(ROOT)
    monkeypatch.setattr(b3, "_object_hash", original_hash)

    original = b3._diagnostics
    old_defaults = original.__defaults__
    old_kwdefaults = original.__kwdefaults__
    original.__defaults__ = (None,)
    try:
        with pytest.raises(b3.B3CoverageError, match="code/default"):
            b3.load_b3_coverage_authority(ROOT)
    finally:
        original.__defaults__ = old_defaults
    original.__kwdefaults__ = {"forged": True}
    try:
        with pytest.raises(b3.B3CoverageError, match="code/default"):
            b3.load_b3_coverage_authority(ROOT)
    finally:
        original.__kwdefaults__ = old_kwdefaults

    original_loader = b3.load_b3_coverage_authority
    legitimate = original_loader(ROOT)
    monkeypatch.setattr(b3, "load_b3_coverage_authority", lambda root: legitimate)
    with pytest.raises(b3.B3CoverageError, match="loader rebound"):
        b3.validate_b3_coverage_authority(legitimate)
    monkeypatch.setattr(b3, "load_b3_coverage_authority", original_loader)
    original_snapshot = b3.snapshot_b3_coverage_authority
    monkeypatch.setattr(b3, "snapshot_b3_coverage_authority", lambda value: {})
    with pytest.raises(b3.B3CoverageError, match="snapshot reader rebound"):
        b3.validate_b3_coverage_authority(legitimate)
    monkeypatch.setattr(b3, "snapshot_b3_coverage_authority", original_snapshot)

    init_function = b3.LoadedB3CoverageAuthority.__dict__["__init__"]
    old_code = init_function.__code__
    init_function.__code__ = (lambda self, *args, **kwargs: None).__code__
    try:
        with pytest.raises(b3.B3CoverageError, match="authority class code"):
            b3.validate_b3_coverage_authority(legitimate)
    finally:
        init_function.__code__ = old_code


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value["hard_gates"].pop(),
        lambda value: value["hard_gates"].append(deepcopy(value["hard_gates"][0])),
        lambda value: value["hard_gates"][0].__setitem__("predicate", False),
        lambda value: (
            value["hard_gates"][0]["evidence"].__setitem__("forged", True),
            value["hard_gates"][0].__setitem__(
                "evidence_sha256",
                b3._object_hash(value["hard_gates"][0]["evidence"]),
            ),
        ),
    ],
)
def test_hard_gate_missing_extra_false_and_self_rehash_reject(
    bundle: Path, mutation
) -> None:
    _mutate_and_reject(bundle, "report.json", mutation)


def test_hard_gate_set_is_literal_and_complete(root_payloads) -> None:
    report = root_payloads["report"]
    assert [gate["gate_id"] for gate in report["hard_gates"]] == [
        "sbc_lineage_complete",
        "sbc_every_regime_uniform",
        "faulty_inference_controls_rejected",
        "multiway_dependence_available",
        "unresolved_rows_excluded",
        "aggregate_reconciliation",
        "wording_and_nonpromotion",
    ]
    assert all(gate["predicate"] is True for gate in report["hard_gates"])
    assert root_payloads["authority"]["hard_gates"][0]["gate_id"] == (
        "source_and_authority_closure"
    )


def test_fresh_process_replay_is_exact() -> None:
    script = (
        "from pathlib import Path;"
        "from lol_kills.v2.evaluation.b3_coverage import load_b3_coverage_authority;"
        f"a=load_b3_coverage_authority(Path({str(ROOT)!r}));"
        "print(a.authority_raw_sha256)"
    )
    first = subprocess.check_output([sys.executable, "-c", script], cwd=ROOT, text=True)
    second = subprocess.check_output([sys.executable, "-c", script], cwd=ROOT, text=True)
    assert first == second
