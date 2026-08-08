from __future__ import annotations

import numpy as np
import pytest

from lol_kills.v2.draft.interactions.structured_prior import (
    PRINCIPAL_PRIOR_SPEC,
    OntologyEvidenceIdentity,
    StructuredPriorError,
    StructuredPriorSpec,
    center_ontology_embeddings,
    effect_space_prior_covariance,
    make_ally_relations,
    orient_enemy_relations,
    prior_scale,
    score_ally_side,
    score_enemy_ordered,
    score_side_swap_logit,
    weighted_contrast_basis,
)

EVIDENCE = OntologyEvidenceIdentity(
    snapshot_sha256="a" * 64,
    as_of="2026-07-01T00:00:00Z",
    reliability=0.8,
)


def test_weighted_contrast_is_centered_and_whitened_for_nonuniform_weights() -> None:
    result = weighted_contrast_basis(
        [0.1, 0.2, 0.3, 0.4], labels=["a", "b", "c", "d"]
    )
    assert result.basis.shape == (4, 3)
    assert result.diagnostics.rank == 3
    np.testing.assert_allclose(
        np.einsum("i,ij->j", result.weights, result.basis), 0.0, atol=1e-12
    )
    np.testing.assert_allclose(
        np.einsum(
            "ji,jk->ik",
            result.basis,
            result.weights[:, None] * result.basis,
        ),
        np.eye(3),
        atol=1e-12,
    )


def test_weighted_contrast_and_ontology_moments_are_permutation_deterministic() -> None:
    labels = ["gamma", "alpha", "beta"]
    weights = [0.5, 0.2, 0.3]
    embeddings = [[3.0, 1.0], [1.0, -1.0], [2.0, 0.0]]
    first = weighted_contrast_basis(weights, labels=labels)
    second = weighted_contrast_basis(
        [weights[1], weights[2], weights[0]],
        labels=[labels[1], labels[2], labels[0]],
    )
    assert first.labels == second.labels
    np.testing.assert_array_equal(first.weights, second.weights)
    np.testing.assert_array_equal(first.basis, second.basis)

    moments_a = center_ontology_embeddings(embeddings, weights, labels=labels)
    moments_b = center_ontology_embeddings(
        [embeddings[1], embeddings[2], embeddings[0]],
        [weights[1], weights[2], weights[0]],
        labels=[labels[1], labels[2], labels[0]],
    )
    assert moments_a.payload_sha256 == moments_b.payload_sha256
    np.testing.assert_allclose(
        np.einsum(
            "i,ij->j", moments_a.weights, moments_a.centered_embeddings
        ),
        [0.0, 0.0],
        atol=1e-12,
    )


def test_public_array_results_cannot_be_mutated_or_drift_their_hashes() -> None:
    contrast = weighted_contrast_basis([0.4, 0.6], labels=["a", "b"])
    moments = center_ontology_embeddings(
        [[1.0, 2.0], [3.0, 4.0]], [0.4, 0.6], labels=["a", "b"]
    )
    ally, enemy = _relations()
    effect_prior = effect_space_prior_covariance(contrast, 0.2)
    objects_and_arrays = [
        (contrast, contrast.weights),
        (contrast, contrast.basis),
        (moments, moments.weights),
        (moments, moments.mean),
        (moments, moments.centered_embeddings),
        (ally, ally.block("top", "mid")),
        (enemy, enemy.block("top", "mid")),
        (effect_prior, effect_prior.coordinate_covariance),
    ]
    for owner, array in objects_and_arrays:
        before = owner.payload_sha256
        with pytest.raises(ValueError):
            array.setflags(write=True)
        with pytest.raises(ValueError):
            array.flat[0] = 999.0
        assert owner.payload_sha256 == before


def _relations():
    roles = ("top", "mid")
    ally = make_ally_relations(
        roles,
        {
            ("top", "top"): [[1.0, 0.2], [0.2, -0.1]],
            ("top", "mid"): [[0.3, -0.4], [0.1, 0.2]],
            ("mid", "top"): [[0.3, 0.1], [-0.4, 0.2]],
            ("mid", "mid"): [[0.5, 0.0], [0.0, 0.1]],
        },
    )
    enemy = orient_enemy_relations(
        roles,
        {
            ("top", "top"): [[1.0, 0.3], [-0.1, 2.0]],
            ("top", "mid"): [[0.4, -0.2], [0.1, 0.5]],
            ("mid", "top"): [[-0.3, 0.6], [0.2, -0.4]],
            ("mid", "mid"): [[0.2, -0.7], [0.1, 0.3]],
        },
    )
    return ally, enemy


def test_ally_relation_is_symmetric_under_role_pair_exchange() -> None:
    ally, _ = _relations()
    top = np.array([0.7, -0.3])
    mid = np.array([-0.2, 0.9])
    forward = top @ ally.block("top", "mid") @ mid
    reverse = mid @ ally.block("mid", "top") @ top
    assert forward == pytest.approx(reverse)
    assert score_ally_side({"top": top, "mid": mid}, ally) == pytest.approx(forward)


def test_enemy_relation_and_complete_logit_are_side_swap_antisymmetric() -> None:
    ally, enemy = _relations()
    side_a = {"top": [0.7, -0.3], "mid": [-0.2, 0.9]}
    side_b = {"top": [-0.4, 0.8], "mid": [0.5, 0.1]}
    np.testing.assert_allclose(
        enemy.block("top", "top"), -enemy.block("top", "top").T, atol=1e-12
    )
    assert score_enemy_ordered(side_a, side_b, enemy) == pytest.approx(
        -score_enemy_ordered(side_b, side_a, enemy)
    )
    ab = score_side_swap_logit(
        side_a, side_b, ally=ally, enemy=enemy, main_a=-0.2, main_b=0.35
    )
    ba = score_side_swap_logit(
        side_b, side_a, ally=ally, enemy=enemy, main_a=0.35, main_b=-0.2
    )
    assert ab == pytest.approx(-ba)


def test_rank_two_shapes_are_enforced() -> None:
    moments = center_ontology_embeddings(
        [[1.0, 2.0], [3.0, 4.0]], [0.25, 0.75], labels=["a", "b"]
    )
    assert moments.centered_embeddings.shape == (2, 2)
    with pytest.raises(StructuredPriorError, match="shape"):
        center_ontology_embeddings([[1.0], [2.0]], [0.5, 0.5], labels=["a", "b"])


def test_structural_variance_is_independent_of_same_fit_support() -> None:
    sparse = prior_scale("ally", same_fit_support=1, evidence_identity=EVIDENCE)
    many = prior_scale("ally", same_fit_support=10_000, evidence_identity=EVIDENCE)
    assert (
        sparse.structural_effect_variance_upper_bound
        == many.structural_effect_variance_upper_bound
    )
    assert sparse.structural_prior_identity_sha256 == many.structural_prior_identity_sha256
    assert sparse.predictive_epistemic_variance > many.predictive_epistemic_variance
    unknown = prior_scale("ally", same_fit_support=None, evidence_identity=EVIDENCE)
    zero = prior_scale("ally", same_fit_support=0, evidence_identity=EVIDENCE)
    assert unknown.evidence_exposure == "unknown"
    assert unknown.predictive_epistemic_variance > zero.predictive_epistemic_variance

    changed_evidence = OntologyEvidenceIdentity(
        snapshot_sha256="b" * 64,
        as_of="2026-07-02T00:00:00Z",
        reliability=0.6,
    )
    changed = prior_scale(
        "ally", same_fit_support=10_000, evidence_identity=changed_evidence
    )
    assert changed.structural_prior_identity_sha256 != many.structural_prior_identity_sha256
    assert (
        changed.structural_effect_variance_upper_bound
        != many.structural_effect_variance_upper_bound
    )


def test_zero_evidence_exact_residual_is_unavailable_and_more_uncertain() -> None:
    unseen = prior_scale(
        "champion_residual", same_fit_support=0, evidence_identity=EVIDENCE
    )
    unknown = prior_scale(
        "champion_residual", same_fit_support=None, evidence_identity=EVIDENCE
    )
    observed = prior_scale(
        "champion_residual", same_fit_support=20, evidence_identity=EVIDENCE
    )
    assert unseen.mean is None
    assert unseen.available is False
    assert unseen.structural_effect_variance_upper_bound is None
    assert unseen.predictive_epistemic_variance is None
    assert unknown.evidence_exposure == "unknown"
    assert unknown.available is False
    assert unseen.parent_block == "main/archetype"
    assert observed.available is True


@pytest.mark.parametrize("block", ["H", "K"])
def test_future_h_and_k_blocks_fail_closed_as_unavailable(block: str) -> None:
    decision = prior_scale(block, same_fit_support=100, evidence_identity=EVIDENCE)
    assert decision.available is False
    assert decision.reason == "future interaction block is not estimated"
    assert decision.mean is None
    assert decision.structural_effect_variance_upper_bound is None
    assert decision.predictive_epistemic_variance is None


def test_canonical_spec_hash_is_stable_and_declares_no_authority() -> None:
    assert PRINCIPAL_PRIOR_SPEC.payload_sha256 == StructuredPriorSpec().payload_sha256
    assert (
        PRINCIPAL_PRIOR_SPEC.payload_sha256
        == "96e6151b824cf819273e6980410139d1ba1c863b9f81082438f62a27a348255c"
    )
    payload = PRINCIPAL_PRIOR_SPEC.to_payload()
    assert payload["authority"]["authorizes_predictive_claims"] is False
    hierarchy = payload["principal_model"]["strong_hierarchy"]
    assert hierarchy["status"] == "declared_principal_model_constraint_not_yet_fit_enforced"
    assert hierarchy["blocks"]["champion_residual"]["semantics"] == "deviation_not_replacement"
    assert payload["principal_model"]["reference_conditioning"][
        "minimum_effective_weight"
    ] == 1e-8
    assert payload["principal_model"]["block_logit_sd_semantics"] == (
        "original_effect_space_marginal_standard_deviation_upper_bound"
    )


def test_effect_space_covariance_bounds_200_uniform_level_marginals() -> None:
    count = 200
    tau = 0.35
    contrast = weighted_contrast_basis(
        [1.0 / count] * count,
        labels=[f"champion-{index:03d}" for index in range(count)],
    )
    prior = effect_space_prior_covariance(contrast, tau)
    induced = np.einsum(
        "ik,kl,jl->ij",
        contrast.basis,
        prior.coordinate_covariance,
        contrast.basis,
    )
    assert prior.coordinate_covariance.shape == (count - 1, count - 1)
    assert prior.diagnostics.coordinate_rank == count - 1
    assert prior.diagnostics.maximum_induced_marginal_variance <= tau**2 + 1e-10
    assert float(np.max(np.diag(induced))) <= tau**2 + 1e-10
    assert float(np.min(np.linalg.eigvalsh(induced))) >= -1e-10


def test_effect_space_covariance_handles_permitted_nonuniform_boundary() -> None:
    tau = 0.2
    contrast = weighted_contrast_basis(
        [1e-8, 0.25, 0.74999999], labels=["rare", "common", "dominant"]
    )
    prior = effect_space_prior_covariance(contrast, tau)
    induced = np.einsum(
        "ik,kl,jl->ij",
        contrast.basis,
        prior.coordinate_covariance,
        contrast.basis,
    )
    assert contrast.diagnostics.minimum_effective_weight == 1e-8
    assert prior.diagnostics.gram_condition_number < 1e9
    assert float(np.max(np.diag(induced))) <= tau**2 + 1e-10


def test_effect_space_covariance_is_permutation_and_hash_deterministic() -> None:
    first = weighted_contrast_basis([0.2, 0.3, 0.5], labels=["b", "c", "a"])
    second = weighted_contrast_basis([0.5, 0.2, 0.3], labels=["a", "b", "c"])
    prior_a = effect_space_prior_covariance(first, 0.15)
    prior_b = effect_space_prior_covariance(second, 0.15)
    np.testing.assert_array_equal(
        prior_a.coordinate_covariance, prior_b.coordinate_covariance
    )
    assert prior_a.payload_sha256 == prior_b.payload_sha256


@pytest.mark.parametrize("tau", [0.0, -0.1, np.nan, np.inf, True])
def test_effect_space_covariance_rejects_invalid_tau(tau) -> None:
    contrast = weighted_contrast_basis([0.4, 0.6], labels=["a", "b"])
    with pytest.raises(StructuredPriorError):
        effect_space_prior_covariance(contrast, tau)


@pytest.mark.parametrize(
    ("weights", "labels"),
    [
        ([0.5, 0.4], ["a", "b"]),
        ([0.5, 0.5, 0.0], ["a", "b", "c"]),
        ([0.5, np.nan, 0.5], ["a", "b", "c"]),
        ([0.5, 0.5], ["a", "a"]),
        ([1.0 - 1e-14, 1e-14], ["a", "b"]),
    ],
)
def test_invalid_weight_inputs_fail_closed(weights, labels) -> None:
    with pytest.raises(StructuredPriorError):
        weighted_contrast_basis(weights, labels=labels)


def test_nonfinite_relations_and_invalid_config_fail_closed() -> None:
    with pytest.raises(StructuredPriorError, match="finite"):
        orient_enemy_relations(("top",), {("top", "top"): [[0.0, np.inf], [0.0, 0.0]]})
    with pytest.raises(StructuredPriorError, match="rank"):
        StructuredPriorSpec(ontology_rank=3)
    with pytest.raises(StructuredPriorError, match="reliability"):
        OntologyEvidenceIdentity(
            snapshot_sha256="a" * 64,
            as_of="2026-07-01T00:00:00Z",
            reliability=1.1,
        )
    with pytest.raises(StructuredPriorError, match="time-safe"):
        OntologyEvidenceIdentity(
            snapshot_sha256="a" * 64,
            as_of="2026-07-01T00:00:00Z",
            reliability=0.8,
            time_safe=False,
        )
    with pytest.raises(StructuredPriorError, match="required"):
        prior_scale("ally", evidence_identity=None)
    absent_identity = OntologyEvidenceIdentity(
        snapshot_sha256="c" * 64,
        as_of="2026-07-01T00:00:00Z",
        reliability=0.8,
        ontology_dimensions_present=False,
    )
    missing = prior_scale("ally", evidence_identity=absent_identity, same_fit_support=5)
    present = prior_scale("ally", evidence_identity=EVIDENCE, same_fit_support=5)
    assert missing.available is False
    assert missing.predictive_epistemic_variance is None
    assert present.predictive_epistemic_variance is not None
