# Champion tier mathematics

## Scope

The model uses completed professional maps from 2025-01-01 through the source watermark. It fits each league or event and role separately. The output is descriptive. Oracle's Elixir draft data is observational, so the model does not claim a causal counter-pick effect.

## Patch ingestion

Every refresh rebuilds and validates the champion atom bridge before tier calculation. The candidate records the bridge hash, canonical Wiki-backed data patch, LCC commit, generation time, per-champion atom profile status, and each champion's `patchLastChanged` value. A failed atomization step blocks promotion.

The atom bridge owns the patch update agenda and current champion-state provenance. Oracle's Elixir patch tokens stay in their source namespace. The model does not infer an official-patch mapping from the OE token. Atom features remain structured priors. They do not become empirical matchup effects.

## Paired-comparison model

The strength stage fits all five roles in one map model. For map `g`, let `b(g,r)` and `q(g,r)` be the blue-side and red-side champions in role `r`:

```text
logit P(Y_g = 1) = o_g + blue_side + sum_r(alpha_b(g,r) - alpha_q(g,r)).
```

`o_g` is the pre-map team Elo log-odds. `alpha` is the marginal champion-role strength after the other four role pairs enter the same equation. The model uses one map outcome once.

The matchup stage fits residual effects after the joint strength model. For each observed same-role pair:

```text
logit P(Y_g = 1) = fitted_joint_logit_g + s_ij gamma_ij.
```

`gamma` is one parameter for each unordered matchup. `s_ij` changes its sign when the order changes. This makes the residual matchup matrix antisymmetric. Strength and matchup residuals are separate registered estimands, which removes the free strength-versus-interaction decomposition in a one-stage model.

Each role comparison graph is split into connected components. The model builds an orthonormal contrast basis with one sum-to-zero constraint for each component. It fits and samples in those reduced coordinates. The proper Gaussian prior keeps the posterior defined when the local likelihood has weak directions. The artifact records likelihood rank and condition diagnostics for every scope. A published Blind or Counter row also requires every focal-to-opponent strength contrast to have posterior standard deviation no greater than 0.50 logit.

The likelihood weight is

```text
w_g = 2^(-age_days / 120).
```

The priors are `alpha_i ~ Normal(0, 0.75^2)` and `gamma_ij ~ Normal(0, 0.35^2)`. They partially pool sparse champions and matchups toward neutral. The implementation finds the penalized maximum-a-posteriori estimate. The joint strength stage inverts the full observed Hessian. The matchup stage uses each pair's exact scalar curvature. An implicit-function sensitivity term propagates the full joint-strength covariance into every matchup residual draw.

This structure follows hierarchical Bradley-Terry work for sparse sports comparisons and dynamic paired-comparison work. Relevant references include Phelan and Whelan, [Hierarchical Bayesian Bradley-Terry for Applications in Major League Baseball](https://doi.org/10.13164/MA.2018.07), Cattelan, Varin, and Firth, [Dynamic Bradley-Terry modelling of sports tournaments](https://doi.org/10.1111/J.1467-9876.2012.01046.X), and Coulom, [Whole-History Rating](https://doi.org/10.1007/978-3-540-87608-3_11).

## Strength

For champion `i`, the model evaluates the five highest-pick legal same-role opponents:

```text
Strength_i = sum_j pi_j logistic(alpha_i - alpha_j).
```

The registered pool contains the six highest-pick champions in the exact role and scope. The focal champion is removed, then the five highest remaining picks are renormalized. This replacement rule prevents self-matchups. Each row records its exact opponents, weights, and distribution hash. Tier Value is `100 * (Strength_i - 0.5)` percentage points. Rank movement compares this strength rank with the previous weekly snapshot.

## Blind score

The model draws joint champion strengths from the full Laplace covariance. It also draws every observed matchup residual from its fitted posterior. An unobserved residual draws from the neutral prior. Each draw gives a matchup probability:

```text
p_ij^(d) = logistic(alpha_i^(d) - alpha_j^(d) + gamma_ij^(d)).
```

For each posterior draw, the model calculates the pick-frequency-weighted mean of the lowest 20% of opponent outcomes. This is lower-tail CVaR. The published Blind score is the posterior 10th percentile of that CVaR. It rewards champions whose weak and uncertain matchups remain strong. Ogryczak's [Robust Decisions under Risk for Imprecise Probabilities](https://doi.org/10.1007/978-3-642-22884-1_3) gives the robust-decision basis for a tail-mean criterion.

## Counter breadth

An observed opponent counts as countered when

```text
P(gamma_ij > 0.05 logit | data) >= 0.80.
```

Counter ranking uses posterior expected weighted breadth, `5 * sum_j pi_j P(gamma_ij > 0.05 | data)`, across the five registered legal opponents. The artifact also reports the stricter count and weighted share that pass the 0.80 probability rule. Each pair needs a Kish effective sample size of at least three maps, at least three effective series, both canonical outcomes, and posterior standard deviation no greater than 0.34 logit. A champion needs complete support against all five opponents before Blind or Counter is available. Zero-evidence matchups never enter the breadth sum.

## Tier assignment

The board uses this order:

```text
Z Blind
Z Counter
S Blind
S Counter
A
B
C
D
```

Point labels use uncertainty-qualified Counter breadth and the conservative Blind tail score. Posterior draws repeat the complete exclusive assignment. A Z or S label needs at least 0.65 posterior membership probability. A through D divide the remaining strength-ranked champions. Sparse cells can have empty Blind or Counter rows.

## Required checks

- Fit all five roles and every published league or event scope.
- Preserve antisymmetry when blue and red champion order is reversed.
- Keep unobserved matchup effects at the neutral prior.
- Withhold Blind and Counter labels below the support gate.
- Refit the previous weekly snapshot with the same method before calculating movement.
- Report source hashes, source watermark, fit constants, and claim limits in the artifact.
- Keep predictive, causal, recommendation, and betting authority closed.
