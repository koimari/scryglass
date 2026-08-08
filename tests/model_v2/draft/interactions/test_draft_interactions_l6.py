from collections import defaultdict
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

import lol_kills.v2.draft.interactions.artifacts as artifacts_module
from lol_kills.v2.draft.interactions.artifacts import (
    ARTIFACT_CONFIG_ID,
    _owned_regular_path,
    _validate_artifact_payload,
    build_authority,
    build_development_report,
    build_fixture_payload,
    build_interactions_config,
    canonical_artifact_bytes,
    load_authorized_l6_model,
    render_draft_interactions_manifest,
)
from lol_kills.v2.draft.interactions.fixtures import load_synthetic_rows
from lol_kills.v2.draft.interactions.model import (
    DraftInteractionModel,
    score_row_pair_swap,
    run_candidate_selection,
)
from lol_kills.v2.draft.interactions.types import DraftCompositionRow, DraftInteractionError
from lol_kills.v2.evaluation.types import canonical_sha256


def _base_rows() -> list[DraftCompositionRow]:
    return load_synthetic_rows()


def test_side_swap_antisymmetry_on_raw_logit() -> None:
    rows = _base_rows()
    model = DraftInteractionModel(draw_count=24, draw_seed=20260728)
    for family in model.FAMILY_REGISTRY:
        for base in rows:
            swapped = score_row_pair_swap(base)
            forward_terms = defaultdict(float)
            reverse_terms = defaultdict(float)
            for term in model._row_terms(base, family):
                forward_terms[term.term_id] += term.value
            for term in model._row_terms(swapped, family):
                reverse_terms[term.term_id] += term.value
            assert set(forward_terms) == set(reverse_terms)
            assert all(
                forward_terms[term_id] == pytest.approx(-reverse_terms[term_id])
                for term_id in forward_terms
            )
        fit = model.fit(rows, family_id=family.family_id)
        forward = model.predict(fit, rows[0])
        reverse = model.predict(fit, score_row_pair_swap(rows[0]))
        assert abs(forward.raw_logit + reverse.raw_logit) < 1e-8


def test_canonical_role_order_invariance_for_payload_input() -> None:
    rows = _base_rows()
    base = rows[0]
    permuted_side = {role: champ for role, champ in reversed(base.side_a)}
    row = DraftCompositionRow.from_payload(
        {
            "row_id": "perm-invariant",
            "patch_id": base.patch_id,
            "league_id": base.league_id,
            "side_a": permuted_side,
            "side_b": {role: champ for role, champ in base.side_b},
            "label": 1,
            "source_id": "perm",
            "source_patch_pool": base.source_patch_pool,
        }
    )
    assert row.side_a == base.side_a
    assert row.side_b == base.side_b


def test_reject_illegal_role_sets_and_duplicates() -> None:
    row = _base_rows()[0].to_payload()
    missing_role = {role: champ for role, champ in row["side_a"].items() if role != "top"}
    with pytest.raises(DraftInteractionError):
        DraftCompositionRow.from_payload(
            {
                **row,
                "row_id": "bad-missing",
                "side_a": missing_role,
            }
        )

    for invalid_patch in ("foo.bar", "26.", " 26.14", "26.14 "):
        with pytest.raises(DraftInteractionError):
            DraftCompositionRow.from_payload(
                {**row, "row_id": f"bad-patch-{invalid_patch}", "patch_id": invalid_patch}
            )
    collision = dict(row["side_a"])
    collision["top"] = "riot:champion:1|role=mid"
    with pytest.raises(DraftInteractionError):
        DraftCompositionRow.from_payload(
            {**row, "row_id": "bad-delimiter", "side_a": collision}
        )
    direct = DraftCompositionRow(
        row_id="direct-normalized",
        patch_id=row["patch_id"],
        league_id=row["league_id"],
        side_a=tuple(reversed(tuple(row["side_a"].items()))),
        side_b=tuple(reversed(tuple(row["side_b"].items()))),
        label=row["label"],
        source_id=row["source_id"],
    )
    assert tuple(role for role, _ in direct.side_a) == tuple(
        role for role, _ in _base_rows()[0].side_a
    )

    cross_side_duplicate = dict(row["side_b"])
    cross_side_duplicate["top"] = row["side_a"]["top"]
    with pytest.raises(DraftInteractionError):
        DraftCompositionRow.from_payload(
            {
                **row,
                "row_id": "bad-cross-side-duplicate",
                "side_b": cross_side_duplicate,
            }
        )

    dup_side = dict(row["side_a"])
    duplicate_champ = next(iter(dup_side.values()))
    role_keys = list(dup_side.keys())
    dup_side[role_keys[1]] = duplicate_champ
    with pytest.raises(DraftInteractionError):
        DraftCompositionRow.from_payload(
            {
                **row,
                "row_id": "bad-duplicate",
                "side_a": dup_side,
            }
        )


def test_ally_pair_coverage_and_cross_pair_coverage() -> None:
    rows = _base_rows()
    model = DraftInteractionModel()
    row = rows[0]
    family = model._family_by_id["residual-full"]

    row_terms = model._row_terms(row, family)
    ally_terms = [term for term in row_terms if term.block.startswith("ally_")]
    ally_by_side: dict[str, set[tuple[str, str, str, str]]] = {"A": set(), "B": set()}
    for term in ally_terms:
        if term.metadata["scope"] != "global":
            continue
        side = str(term.metadata["side"])
        champ_a, champ_b = term.metadata["champions"]
        role_a, role_b = term.metadata["roles"]
        ally_by_side[side].add((role_a, champ_a, role_b, champ_b))
    assert len(ally_by_side["A"]) == 10
    assert len(ally_by_side["B"]) == 10

    side_a = {champ for _, champ in row.side_a}
    side_b = {champ for _, champ in row.side_b}
    for champion in side_a:
        paired = [pair for pair in ally_by_side["A"] if pair[1] == champion or pair[3] == champion]
        assert len(paired) == 4
    for champion in side_b:
        paired = [pair for pair in ally_by_side["B"] if pair[1] == champion or pair[3] == champion]
        assert len(paired) == 4

    enemy_global = [term for term in row_terms if term.block.startswith("enemy") and term.metadata["scope"] == "global"]
    pairs = set()
    by_a_champ = defaultdict(set)
    by_b_champ = defaultdict(set)
    same_role_present = False
    for term in enemy_global:
        champs = term.metadata["champions"]
        roles = term.metadata["roles"]
        if term.metadata["side"] == "A":
            a_role, a_champ = roles[0], champs[0]
            b_role, b_champ = roles[1], champs[1]
        else:
            a_role, a_champ = roles[1], champs[1]
            b_role, b_champ = roles[0], champs[0]
        if a_role == b_role:
            same_role_present = True
        pairs.add((a_role, a_champ, b_role, b_champ))
        by_a_champ[a_champ].add((a_role, b_role, b_champ))
        by_b_champ[b_champ].add((b_role, a_role, a_champ))

    assert len(pairs) == 25
    assert same_role_present
    assert all(len(v) == 5 for v in by_a_champ.values())
    assert all(len(v) == 5 for v in by_b_champ.values())

    swapped_terms = model._row_terms(score_row_pair_swap(row), family)
    forward = defaultdict(float)
    reverse = defaultdict(float)
    for term in row_terms:
        forward[term.term_id] += term.value
    for term in swapped_terms:
        reverse[term.term_id] += term.value
    assert set(forward) == set(reverse)
    assert all(abs(forward[term_id] + reverse[term_id]) < 1e-12 for term_id in forward)
    assert all("|side=" not in term_id for term_id in forward)


def test_blockwise_projection_and_termwise_orthogonality() -> None:
    rows = tuple(_base_rows())
    model = DraftInteractionModel(draw_count=12, draw_seed=11)
    family = model._family_by_id["residual-full"]
    design, term_meta, term_ids = model._build_design_matrix(rows, family)

    assert design.shape[1] == len(term_ids)
    assert design.shape[1] > 0

    block_to_cols: defaultdict[str, list[int]] = defaultdict(list)
    for idx, metadata in enumerate(term_meta):
        block_to_cols[str(metadata["block"])].append(idx)

    fit = model.fit(rows, family_id="pair-baseline")
    reference_scores = {
        key: value
        for key, value in fit.diagnostics.orthogonality.items()
        if key.startswith("reference_")
    }
    assert reference_scores
    assert max(reference_scores.values()) < 1e-8
    assert fit.transform_sha256
    assert fit.reference_sha256
    distribution = model.legal_reference_distribution()
    assert distribution["row_count"] == 800
    assert len({row["row_id"] for row in distribution["rows"]}) == 800
    assert all(row["ordinal"] == index for index, row in enumerate(distribution["rows"]))
    assert all(row["weight"] == pytest.approx(1.0 / 800.0) for row in distribution["rows"])
    assert sum(row["weight"] for row in distribution["rows"]) == pytest.approx(
        1.0,
        abs=distribution["normalization_tolerance"],
    )
    assert (
        fit.reference_sha256
        == distribution["sha256"]
        == model.validate_legal_reference_distribution(distribution)
    )
    stale_weight = json.loads(json.dumps(distribution))
    stale_weight["rows"][0]["weight"] = 0.025
    with pytest.raises(DraftInteractionError, match="uniformly normalized"):
        model.validate_legal_reference_distribution(stale_weight)
    stale_order = json.loads(json.dumps(distribution))
    stale_order["rows"][0], stale_order["rows"][1] = (
        stale_order["rows"][1],
        stale_order["rows"][0],
    )
    with pytest.raises(DraftInteractionError, match="row order"):
        model.validate_legal_reference_distribution(stale_order)

    raw_index = {
        term_id: index
        for index, term_id in enumerate(fit.raw_feature_terms)
    }
    raw_row = np.zeros(len(fit.raw_feature_terms), dtype=float)
    for term in model._row_terms(
        rows[0], model._family_by_id["pair-baseline"]
    ):
        if term.term_id in raw_index:
            raw_row[raw_index[term.term_id]] += term.value
    transformed_row = np.einsum(
        "i,ij->j", raw_row, np.asarray(fit.transform_matrix)
    )
    assert transformed_row == pytest.approx(
        model._build_design_matrix(
            rows, model._family_by_id["pair-baseline"]
        )[0][0]
    )
    fitted_row_logit = float(
        np.dot(
            transformed_row,
            np.asarray(
                [fit.coefficients[term] for term in fit.feature_terms]
            ),
        )
    )
    assert model.predict(fit, rows[0]).raw_logit == pytest.approx(fitted_row_logit)


def test_predict_reconciliation_is_exact_when_identified() -> None:
    rows = _base_rows()
    model = DraftInteractionModel(draw_count=16, draw_seed=99)
    fit = model.fit(rows, family_id="pair-baseline")
    row = rows[0]
    forged = model.predict(replace(fit, decomposition_mode="identified"), row)
    assert forged.decomposition_mode == "total_only"
    assert forged.ledger["status"] == "unavailable"
    assert "raw_probability" not in forged.as_payload
    assert forged.as_payload["component_payload_available"] is False
    assert "coefficients" not in replace(
        fit, decomposition_mode="identified"
    ).as_payload

    oracle_diagnostics = replace(
        fit.diagnostics,
        identification_status="identified",
        warnings=(),
        fallback_term_count=0,
        fallback_counts={},
    )
    proof = canonical_sha256(
        {
            "diagnostics": oracle_diagnostics.as_payload,
            "transform_sha256": fit.transform_sha256,
            "reference_sha256": fit.reference_sha256,
        }
    )
    oracle_fit = replace(
        fit,
        diagnostics=oracle_diagnostics,
        decomposition_mode="identified",
        identification_proof_sha256=proof,
    )
    pred = model.predict(oracle_fit, row)

    assert pred.ledger["status"] == "available"
    assert pred.ledger["reconciliation_error"] < 1e-8
    assert pred.ledger["reconciliation_total"]["prediction_logit"] == pytest.approx(pred.raw_logit)
    block_sum = sum(
        sum(parts.values()) for parts in pred.ledger["block_contributions"].values()
    )
    assert block_sum == pytest.approx(pred.raw_logit)
    assert pred.ledger["term_count_by_block"]["unclassified"] == 0
    assert all(
        item["block"] != "unclassified" for item in pred.ledger["used_terms"]
    )


def test_archetype_transfer_guard_for_sparse_sparse() -> None:
    rows = _base_rows()
    model = DraftInteractionModel()
    no_transfer = model.fit(rows, family_id="no-archetype-transfer")
    assert all(
        metadata["block"] not in {"ally_sparse", "enemy_sparse"}
        for _, metadata in no_transfer.term_metadata
    )
    residual = model.fit(rows, family_id="residual-full")
    assert set(no_transfer.raw_feature_terms) != set(
        residual.raw_feature_terms
    )
    assert len(no_transfer.raw_feature_terms) != len(
        residual.raw_feature_terms
    )

    bucket, confidence, reason = model._archetype_bucket("riot:champion:99911", "mid", "26.14", allow_unknown=True)
    assert bucket.startswith("archetype|")
    assert 0.0 <= confidence <= 1.0
    assert isinstance(reason, str) and reason

    fit = model.fit(rows, family_id="pair-baseline")
    unseen_payload = rows[0].to_payload()
    unseen_payload["row_id"] = "unseen-champion"
    unseen_payload["side_a"]["mid"] = "riot:champion:777777"
    unseen = DraftCompositionRow.from_payload(unseen_payload)
    unseen_exact_terms = {
        term.term_id
        for term in model._row_terms(unseen, model._family_by_id["residual-full"])
        if term.block in {"ally_exact", "enemy_exact"}
        and "riot:champion:777777" in term.metadata.get("champions", ())
    }
    assert unseen_exact_terms
    assert unseen_exact_terms.isdisjoint(fit.coefficients)
    unseen_sparse_terms = {
        term.term_id
        for term in model._row_terms(unseen, model._family_by_id["pair-baseline"])
        if term.block in {"ally_sparse", "enemy_sparse"}
    }
    assert unseen_sparse_terms & set(fit.coefficients)
    assert all(
        "riot:champion:777777"
        not in {champion for _, champion in row.side_a + row.side_b}
        for row in fit.raw_rows
    )
    held_champion = "riot:champion:115"
    loo_rows = tuple(
        row
        for row in rows
        if held_champion
        not in {champion for _, champion in row.side_a + row.side_b}
    )
    assert loo_rows
    loo_fit = model.fit(loo_rows, family_id="pair-baseline")
    held_row = next(
        row
        for row in rows
        if held_champion
        in {champion for _, champion in row.side_a + row.side_b}
    )
    held_sparse = {
        term.term_id
        for term in model._row_terms(
            held_row, model._family_by_id["pair-baseline"]
        )
        if term.block in {"ally_sparse", "enemy_sparse"}
        and term.metadata.get("scope") == "global"
    }
    assert held_sparse & set(loo_fit.raw_feature_terms)

    new_patch_payload = unseen.to_payload()
    new_patch_payload["row_id"] = "new-patch"
    new_patch_payload["patch_id"] = "99.99"
    new_patch = DraftCompositionRow.from_payload(new_patch_payload)
    base_sparse = model._row_terms(
        unseen, model._family_by_id["pair-baseline"]
    )
    new_patch_sparse = model._row_terms(
        new_patch, model._family_by_id["pair-baseline"]
    )
    for scope in ("global", "competition_scope", "league"):
        assert {
            term.term_id
            for term in base_sparse
            if term.block in {"ally_sparse", "enemy_sparse"}
            and term.metadata.get("scope") == scope
        } == {
            term.term_id
            for term in new_patch_sparse
            if term.block in {"ally_sparse", "enemy_sparse"}
            and term.metadata.get("scope") == scope
        }
    assert {
        term.term_id
        for term in base_sparse
        if term.metadata.get("scope") == "patch"
    }.isdisjoint(
        {
            term.term_id
            for term in new_patch_sparse
            if term.metadata.get("scope") == "patch"
        }
    )

    changed_scope = replace(unseen, competition_scope_id="other-scope")
    changed_terms = model._row_terms(
        changed_scope, model._family_by_id["pair-baseline"]
    )
    assert {
        term.term_id
        for term in base_sparse
        if term.metadata.get("scope") == "competition_scope"
    }.isdisjoint(
        {
            term.term_id
            for term in changed_terms
            if term.metadata.get("scope") == "competition_scope"
        }
    )
    assert all("tier_eligible" not in metadata for _, metadata in fit.term_metadata)


def test_identification_failures_and_posterior_oracle() -> None:
    rows = _base_rows()
    model = DraftInteractionModel(draw_count=16, draw_seed=71)
    deficient = model.fit(rows, family_id="main-only")
    assert deficient.decomposition_mode == "total_only"
    assert {"rank_deficiency", "ill_conditioned"} <= set(
        deficient.diagnostics.warnings
    )

    assert deficient.diagnostics.fallback_counts["duplicate_columns"] > 0
    assert deficient.diagnostics.collinearity_max_correlation == pytest.approx(
        1.0
    )
    assert deficient.diagnostics.condition_number == float("inf")

    pair_metadata = next(
        metadata
        for _, metadata in model.fit(
            rows, family_id="residual-full"
        ).term_metadata
        if metadata["block"] == "ally_exact"
        and metadata.get("scope") == "global"
    )
    expected_support = sum(
        1
        for row in rows
        if all(
            assignment in set(row.side_a)
            for assignment in zip(
                pair_metadata["roles"], pair_metadata["champions"]
            )
        )
        or all(
            assignment in set(row.side_b)
            for assignment in zip(
                pair_metadata["roles"], pair_metadata["champions"]
            )
        )
    )
    assert model._context_support_count(
        tuple(rows), pair_metadata
    ) == expected_support
    assert expected_support < len(rows)
    source_delta = model._source_removal_stability(
        tuple(rows), model._family_by_id["main-only"]
    )
    patch_delta = model._patch_removal_stability(
        tuple(rows), model._family_by_id["main-only"]
    )
    assert source_delta > 0.0
    assert patch_delta > 0.0

    x = np.array([[-1.0], [1.0]])
    beta = np.array([0.0])
    prior_var = np.array([2.0])
    covariance_diag, covariance_factor, posterior_corr = model._approximate_covariance(
        x, beta, prior_var
    )
    exact_hessian_inverse = 1.0 / (0.25 * 2.0 + 0.5)
    assert covariance_diag[0] == pytest.approx(2.0)
    assert covariance_factor.shape == (2, 1)
    assert model._predictive_variance(
        np.array([1.0]), covariance_diag, covariance_factor
    ) == pytest.approx(exact_hessian_inverse)
    oracle_draws = model._predictive_draws(
        np.array([1.0]),
        beta,
        tuple(covariance_diag),
        covariance_factor=tuple(tuple(row) for row in covariance_factor),
        draw_count=100_000,
        covariance_seed=314159,
    )
    oracle_logits = np.log(oracle_draws / (1.0 - oracle_draws))
    assert float(np.var(oracle_logits)) == pytest.approx(
        exact_hessian_inverse, abs=0.02
    )
    assert posterior_corr == 0.0
    row_vector = np.array([1.0])
    first = model._predictive_draws(row_vector, beta, tuple(covariance_diag))
    second = model._predictive_draws(row_vector, beta, tuple(covariance_diag))
    assert np.array_equal(first, second)

    pair_fit = model.fit(rows, family_id="pair-baseline")
    interval_diagnostics = replace(
        pair_fit.diagnostics,
        identification_status="identified",
        warnings=(),
        fallback_term_count=0,
        fallback_counts={},
    )
    interval_proof = canonical_sha256(
        {
            "diagnostics": interval_diagnostics.as_payload,
            "transform_sha256": pair_fit.transform_sha256,
            "reference_sha256": pair_fit.reference_sha256,
        }
    )
    interval_fit = replace(
        pair_fit,
        diagnostics=interval_diagnostics,
        decomposition_mode="identified",
        identification_proof_sha256=interval_proof,
    )
    first_model = DraftInteractionModel(draw_count=8, draw_seed=1)
    second_model = DraftInteractionModel(draw_count=256, draw_seed=999)
    first_prediction = first_model.predict(
        interval_fit, rows[0]
    )
    second_prediction = second_model.predict(
        interval_fit, rows[0]
    )
    assert first_prediction.lower_95 == second_prediction.lower_95
    assert first_prediction.upper_95 == second_prediction.upper_95

    splits = model._development_splits(tuple(rows))
    assert splits
    for train_rows, eval_rows in splits:
        assert {row.source_id for row in train_rows}.isdisjoint(
            {row.source_id for row in eval_rows}
        )
        assert {row.label for row in train_rows} == {0, 1}
        assert {row.label for row in eval_rows} == {0, 1}


def test_candidate_selection_fail_closed_and_reproducible() -> None:
    single_row = _base_rows()[:1]
    report = run_candidate_selection(single_row, draw_count=8, draw_seed=20260728)
    assert report.selection_status == "blocked_no_identified_candidate"
    assert report.selected_family is None
    assert set(candidate.family_id for candidate in report.candidates) == {family.family_id for family in model_family_registry()}
    assert report.selected_sha256 is None

    multi_report = run_candidate_selection(_base_rows(), draw_count=16, draw_seed=20260728)
    assert multi_report.candidate_count == len(model_family_registry())
    assert multi_report.selection_status in {"selected", "blocked_no_identified_candidate"}


def test_artifacts_are_replayable_and_synthetics_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = build_interactions_config()
    report = build_development_report()
    fixture = build_fixture_payload()
    manifest = render_draft_interactions_manifest()
    authority_rebuild = build_authority()

    assert config["principal_estimand"] == "neutral_five_versus_five_composition_value"
    assert config["production_eligible"] is False
    assert config["claim_ceiling"]["synthetic_only"] is True
    assert "rows" in config and config["rows"]
    assert isinstance(config["rows_checksum"], str) and len(config["rows_checksum"]) > 10

    assert report["candidate_count"] == len(config["families"])
    assert report["production_eligible"] is False

    assert fixture["kind"] == "draft-interactions-l6-synthetic-rows"
    assert all(isinstance(item["row_id"], str) for item in fixture["rows"])
    root = Path(__file__).resolve().parents[4]
    authority_path = root / "data/lol/v2/models/draft-interactions/draft-interactions-authority.json"
    authority = json.loads(authority_path.read_text(encoding="utf-8"))
    generated = {
        "draft-interactions-config.json": config,
        "draft-interactions-fixtures.json": fixture,
        "draft-interactions-development-report.json": report,
        "draft-interactions-authority.json": authority_rebuild,
        "draft-interactions-manifest.json": manifest,
    }
    artifact_root = root / "data/lol/v2/models/draft-interactions"
    for filename, rebuilt in generated.items():
        assert (artifact_root / filename).read_bytes() == canonical_artifact_bytes(
            rebuilt
        )

    for item in (
        authority["artifacts"]
        + authority["implementation_inputs"]
        + authority["foundation_inputs"]
    ):
        path = root / item["locator"]
        assert path.is_file() and not path.is_symlink()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item["raw_sha256"]

    assert manifest["hash_semantics"] == "sha256_of_exact_file_bytes"
    assert "manifest" not in {
        item["role"] for item in authority["artifacts"]
    }
    assert authority["legal_reference_transform"]["learned_from_evaluation_rows"] is False
    assert (
        config["legal_reference_distribution"]["sha256"]
        == authority["legal_reference_transform"]["sha256"]
        == DraftInteractionModel().legal_reference_distribution()["sha256"]
    )
    assert authority["identity_kind"] == "non_authorizing_candidate_identity"
    assert authority["independent_l6_authority_present"] is False
    assert authority["external_authority_digest"] is None
    with pytest.raises(ValueError):
        load_authorized_l6_model()
    forged = dict(config)
    forged["production_eligible"] = True
    with pytest.raises(ValueError):
        _validate_artifact_payload(
            forged,
            role="config",
            artifact_id=ARTIFACT_CONFIG_ID,
        )
    for forbidden_claim in ("publication_authorized", "pass_b2", "c2"):
        forged_claim = dict(config)
        forged_claim["claim_ceiling"] = {
            **config["claim_ceiling"],
            forbidden_claim: True,
        }
        with pytest.raises(ValueError):
            _validate_artifact_payload(
                forged_claim,
                role="config",
                artifact_id=ARTIFACT_CONFIG_ID,
            )
    for alias in (
        "/tmp/absolute.json",
        "./data/file.json",
        "../data/file.json",
        "data//file.json",
        "data/./file.json",
    ):
        with pytest.raises(ValueError):
            _owned_regular_path(alias, must_exist=False)
    for role, digest in manifest["artifact_hashes"].items():
        locator = (
            "data/lol/v2/models/draft-interactions/draft-interactions-authority.json"
            if role == "candidate_identity"
            else next(item["locator"] for item in authority["artifacts"] if item["role"] == role)
        )
        assert hashlib.sha256((root / locator).read_bytes()).hexdigest() == digest

    isolated_root = tmp_path / "root"
    isolated_root.mkdir()
    real_parent = isolated_root / "real"
    real_parent.mkdir()
    (real_parent / "payload.json").write_text("{}", encoding="utf-8")
    (isolated_root / "alias").symlink_to(real_parent, target_is_directory=True)
    monkeypatch.setattr(artifacts_module, "_REPO_ROOT", isolated_root)
    with pytest.raises(ValueError):
        artifacts_module._owned_regular_path("alias/payload.json")
    hardlink = isolated_root / "hardlink.json"
    os.link(real_parent / "payload.json", hardlink)
    with pytest.raises(ValueError):
        artifacts_module._owned_regular_path("hardlink.json")


def model_family_registry() -> set[object]:
    model = DraftInteractionModel()
    return set(model.FAMILY_REGISTRY)
