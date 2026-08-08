"""Focused tests for the isolated sparse joint pooled estimator."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from lol_kills.v2.tierlists.joint_pooled_model import (
    AtomFeatureRegistry,
    AtomFeatureVector,
    JointMapObservation,
    JointPooledModelError,
    design_vector_for_observation,
    fit_joint_pooled_model,
    predict_linear_predictor,
    sample_posterior,
)


ROLES = ("top", "mid")
REGISTRY = AtomFeatureRegistry.from_names(
    ("family.control", "attribute.mobility", "ontology.range"),
    source="test_atom_bridge",
)


def _vector(values: tuple[float, float, float]) -> AtomFeatureVector:
    return AtomFeatureVector.from_values(values, available=(True, True, True))


def _map(
    map_id: str,
    outcome: int,
    *,
    scope: str = "LEC",
    patch: str = "16.14",
    top: tuple[str, str] = ("Aatrox", "Gnar"),
    mid: tuple[str, str] = ("Ahri", "Orianna"),
    atom_top: tuple[float, float, float] = (1.0, 0.5, -0.25),
    atom_mid: tuple[float, float, float] = (-0.5, 0.25, 0.75),
    offset: float = 0.0,
    weight: float = 1.0,
) -> JointMapObservation:
    return JointMapObservation(
        map_id=map_id,
        outcome=outcome,
        scope_id=scope,
        oe_patch_id=patch,
        picks={"top": top, "mid": mid},
        atom_pair_features={"top": _vector(atom_top), "mid": _vector(atom_mid)},
        offset=offset,
        weight=weight,
        synthetic=True,
    )


def _maps() -> list[JointMapObservation]:
    return [
        _map("m1", 1),
        _map("m2", 0, top=("Gnar", "Aatrox"), mid=("Orianna", "Ahri"), atom_top=(-1.0, -0.5, 0.25), atom_mid=(0.5, -0.25, -0.75)),
        _map("m3", 1, scope="LCS"),
        _map("m4", 0, scope="LCS", patch="16.15", top=("Aatrox", "Gnar"), mid=("Orianna", "Ahri"), atom_mid=(0.4, -0.3, -0.6)),
        _map("m5", 1, scope="LEC", patch="16.15", top=("Gnar", "Aatrox"), mid=("Ahri", "Orianna"), atom_top=(-1.0, -0.5, 0.25)),
        _map("m6", 0, scope="LCS", patch="16.15", top=("Gnar", "Aatrox"), mid=("Ahri", "Orianna"), atom_top=(-1.0, -0.5, 0.25)),
    ]


def _fit() :
    return fit_joint_pooled_model(
        _maps(),
        feature_registry=REGISTRY,
        roles=ROLES,
        atom_deviation_dim=2,
    )


def test_each_map_contributes_one_outcome_row() -> None:
    fit = _fit()

    assert fit.design_matrix.shape[0] == len(_maps())
    assert fit.metadata["n_maps"] == len(_maps())
    assert fit.metadata["outcome_count"] == len(_maps())
    assert fit.metadata["outcomes_used_once"] is True
    assert fit.metadata["offset_in_likelihood"] is True
    assert fit.metadata["temporal_weight_in_likelihood"] is True
    assert fit.design_metadata["map_contribution_rule"] == {
        "rows_per_map": 1,
        "outcomes_per_map": 1,
        "role_outcomes": 0,
        "likelihood": "one Bernoulli outcome per completed map",
    }
    assert [row["map_id"] for row in fit.design_metadata["rows"]] == [
        "m1", "m2", "m3", "m4", "m5", "m6"
    ]
    assert len(fit.map_predictions) == len(_maps())
    for prediction in fit.map_predictions:
        assert sum(prediction["contributions"].values()) == pytest.approx(
            prediction["linear_predictor"], abs=1e-12
        )


def test_side_reversal_negates_the_complete_sparse_row() -> None:
    original_map = _map("original", 1)
    reversed_map = JointMapObservation(
        map_id="reversed",
        outcome=0,
        scope_id=original_map.scope_id,
        oe_patch_id=original_map.oe_patch_id,
        picks={role: (pair[1], pair[0]) for role, pair in original_map.picks.items()},
        atom_pair_features={
            role: vector.reversed()
            for role, vector in original_map.atom_pair_features.items()
        },
        offset=original_map.offset,
        weight=original_map.weight,
        synthetic=True,
    )
    original = fit_joint_pooled_model(
        [original_map], feature_registry=REGISTRY, roles=ROLES, atom_deviation_dim=2
    )
    reversed_fit = fit_joint_pooled_model(
        [reversed_map], feature_registry=REGISTRY, roles=ROLES, atom_deviation_dim=2
    )

    assert original.parameter_names == reversed_fit.parameter_names
    np.testing.assert_allclose(
        reversed_fit.design_matrix.toarray(),
        -original.design_matrix.toarray(),
        rtol=0.0,
        atol=0.0,
    )
    original_eta_under_reversed_row = (reversed_fit.design_matrix @ original.coefficients).item()
    original_eta = (original.design_matrix @ original.coefficients).item()
    assert original_eta_under_reversed_row == pytest.approx(-original_eta, abs=1e-12)


def test_league_and_patch_parameters_are_partially_pooled() -> None:
    fit = _fit()
    columns = fit.design_metadata["columns"]

    global_atom = [column for column in columns if column["family"] == "global_atom_pair"]
    scope_atom = [column for column in columns if column["family"] == "scope_atom_deviation"]
    patch_atom = [column for column in columns if column["family"] == "oe_patch_atom_deviation"]
    scope_strength = [column for column in columns if column["family"] == "scope_strength_deviation"]

    assert len(global_atom) == len(ROLES) * len(REGISTRY.names)
    assert {column["scope_id"] for column in scope_atom} == {"LEC", "LCS"}
    assert {column["oe_patch_id"] for column in patch_atom} == {"16.14", "16.15"}
    assert {column["scope_id"] for column in scope_strength} == {"LEC", "LCS"}
    assert all(column["prior_sd"] > 0 for column in columns)
    assert "shared across scopes and OE patches" in fit.metadata["partial_pooling"]["global_atom_pair"]


def test_fit_and_sampler_are_deterministic() -> None:
    first = _fit()
    second = _fit()

    assert first.parameter_names == second.parameter_names
    np.testing.assert_array_equal(first.coefficients, second.coefficients)
    np.testing.assert_array_equal(first.covariance_diagonal, second.covariance_diagonal)
    np.testing.assert_array_equal(first.design_matrix.data, second.design_matrix.data)
    assert first.design_metadata == second.design_metadata
    assert first.map_predictions == second.map_predictions
    np.testing.assert_array_equal(
        first.sample_posterior(posterior_draws=17, seed=91),
        second.sample_posterior(posterior_draws=17, seed=91),
    )


def test_two_thousand_complete_posterior_draws_are_finite() -> None:
    fit = _fit()

    draws = sample_posterior(fit, posterior_draws=2000, seed=123, chunk_size=37)

    assert draws.shape == (2000, len(fit.parameter_names))
    assert np.all(np.isfinite(draws))
    assert fit.metadata["laplace"]["covariance_type"] == "diagonal"
    assert fit.metadata["laplace"]["covariance_is_full"] is False
    assert fit.pair_posteriors
    assert all(
        np.isfinite(float(value["mean"]))
        and np.isfinite(float(value["width"]))
        and float(value["width"]) > 0.0
        for value in fit.pair_posteriors.values()
    )


def test_unavailable_features_are_explicit_and_never_zero_imputed() -> None:
    vector = AtomFeatureVector.from_values((1.0, None, -0.25), available=(True, False, True))
    row = _map("missing", 1)
    row = JointMapObservation(
        map_id=row.map_id,
        outcome=row.outcome,
        scope_id=row.scope_id,
        oe_patch_id=row.oe_patch_id,
        picks=row.picks,
        atom_pair_features={"top": vector, "mid": vector},
        offset=row.offset,
        weight=row.weight,
        synthetic=True,
    )
    fit = fit_joint_pooled_model(
        [row], feature_registry=REGISTRY, roles=ROLES, atom_deviation_dim=2
    )

    assert fit.design_metadata["rows"][0]["available_atom_features"]["top"] == [
        "family.control", "ontology.range"
    ]
    global_mobility = [
        column["index"]
        for column in fit.design_metadata["columns"]
        if column["family"] == "global_atom_pair" and column["feature"] == "attribute.mobility"
    ]
    assert len(global_mobility) == len(ROLES)
    assert all(index not in fit.design_metadata["rows"][0]["csr_indices"] for index in global_mobility)


def test_duplicate_map_ids_fail_closed() -> None:
    with pytest.raises(JointPooledModelError, match="duplicated"):
        fit_joint_pooled_model(
            [_map("same", 1), _map("same", 0)],
            feature_registry=REGISTRY,
            roles=ROLES,
        )


def test_offset_changes_the_map_prediction() -> None:
    baseline_maps = _maps()
    shifted_maps = list(baseline_maps)
    shifted_maps[0] = replace(baseline_maps[0], offset=4.0)

    baseline = fit_joint_pooled_model(
        baseline_maps, feature_registry=REGISTRY, roles=ROLES, atom_deviation_dim=2
    )
    shifted = fit_joint_pooled_model(
        shifted_maps, feature_registry=REGISTRY, roles=ROLES, atom_deviation_dim=2
    )

    assert shifted.map_predictions[0]["offset"] == 4.0
    assert shifted.map_predictions[0]["probability"] != pytest.approx(
        baseline.map_predictions[0]["probability"], abs=1e-8
    )


def test_nonpositive_temporal_weight_is_rejected() -> None:
    with pytest.raises(JointPooledModelError, match="weight must be positive"):
        _map("zero-weight", 1, weight=0.0)


def test_known_observation_adapter_matches_fitted_sparse_row() -> None:
    maps = _maps()
    fit = fit_joint_pooled_model(
        maps, feature_registry=REGISTRY, roles=ROLES, atom_deviation_dim=2
    )
    observation = maps[0]
    row_index = next(
        row["row_index"]
        for row in fit.design_metadata["rows"]
        if row["map_id"] == observation.map_id
    )

    adapted_row = design_vector_for_observation(fit, observation)

    np.testing.assert_array_equal(
        adapted_row.toarray(), fit.design_matrix.getrow(row_index).toarray()
    )
    assert predict_linear_predictor(fit, observation) == pytest.approx(
        fit.map_predictions[row_index]["linear_predictor"], abs=1e-12
    )


def test_hypothetical_adapter_rejects_unknown_keys() -> None:
    fit = _fit()
    known = _maps()[0]

    with pytest.raises(JointPooledModelError, match="unknown scope key"):
        design_vector_for_observation(fit, replace(known, scope_id="UNKNOWN"))
    with pytest.raises(JointPooledModelError, match="unknown OE patch key"):
        design_vector_for_observation(fit, replace(known, oe_patch_id="99.99"))
    with pytest.raises(JointPooledModelError, match="unknown champion key"):
        design_vector_for_observation(
            fit,
            replace(known, picks={"top": ("Unseen", "Gnar"), "mid": known.picks["mid"]}),
        )

    pair_fit = fit_joint_pooled_model(
        [* _maps(), _map("m7", 1, top=("Camille", "Gnar"))],
        feature_registry=REGISTRY,
        roles=ROLES,
        atom_deviation_dim=2,
    )
    with pytest.raises(JointPooledModelError, match="unknown canonical pair key"):
        design_vector_for_observation(
            pair_fit,
            _map("hypothetical", 1, top=("Aatrox", "Camille")),
        )
