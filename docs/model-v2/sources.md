# Sources and design basis

This bibliography distinguishes established methods from the proposed
Scryglass synthesis. A citation supports a method or constraint; it does not
prove the final Scryglass candidate is best.

## Established rating and paired-comparison methods

1. **Bradley, R. A. and Terry, M. E. (1952), “Rank Analysis of Incomplete
   Block Designs: I. The Method of Paired Comparisons.”**
   <https://doi.org/10.2307/2334029>
   Basis for logistic paired-comparison likelihoods. Established method.

2. **Glickman, M. E. (1999), “Parameter Estimation in Large Dynamic Paired
   Comparison Experiments.”**
   <https://www.glicko.net/research/glicko.pdf>
   Supports time-varying ability, uncertainty, and state-space paired
   comparisons instead of fixed-K Elo updates. Established method.

3. **Herbrich, R., Minka, T., and Graepel, T. (2006), “TrueSkill: A Bayesian
   Skill Rating System.”**
   <https://www.microsoft.com/en-us/research/wp-content/uploads/2007/01/NIPS2006_0688.pdf>
   Supports Bayesian uncertainty and inference of individual skills from team
   results. Established method; Scryglass role/policy synthesis is proposed.

4. **Dangauthier, P. et al. (2007), “TrueSkill Through Time.”**
   <https://papers.nips.cc/paper_files/paper/2007/file/9f53d83ec0691550f7d2507d57f4f5a2-Paper.pdf>
   Supports smoothing/filtering time-varying ratings and motivates historical
   as-of replay. Established method.

5. **Duffield, S., Power, S., and Rimella, L. (2024), “A State-Space
   Perspective on Modelling and Inference for Online Skill Rating.”**
   <https://doi.org/10.1093/jrsssc/qlae035>
   Reviews online skill rating as state-space modelling, including filtering,
   smoothing, parameter estimation, scalable approximations, and reproducible
   comparisons. It motivates candidate dynamics and evaluation baselines; it
   does not establish Scryglass's final model as state of the art.

6. **Minka, T., Cleven, R., and Zaykov, Y. (2018), “TrueSkill 2: An Improved
   Bayesian Skill Rating System.”**
   <https://www.microsoft.com/en-us/research/publication/trueskill-2-improved-bayesian-skill-rating-system/>
   Demonstrates a Bayesian rating extension that uses auxiliary performance
   information in addition to outcomes. It is an auxiliary-channel baseline
   only when Scryglass inputs pass availability, leakage, role-fairness, and
   missingness audits.

7. **FIDE Rating Regulations, effective 1 March 2024 (current page checked
   27 July 2026), and Elo, A. E. (1978), *The Rating of Chessplayers, Past and
   Present*.**
   <https://handbook.fide.com/chapter/B022024>
   The official rules document the public convention that rating differences
   map to expected scores through a conversion table; the current page also
   contains later effective amendments to rating calculations. Scryglass
   explicitly chooses the uncapped continuous base-10, 400-point Elo display
   curve. It does not claim exact FIDE tables, caps, K factors, update rules, or
   pool semantics. Established Elo interpretation; Scryglass application to
   esports is proposed.

## Shrinkage, set structure, and explanation

8. **Piironen, J. and Vehtari, A. (2017), “Sparsity Information and
   Regularization in the Horseshoe and Other Shrinkage Priors.”**
   <https://doi.org/10.1214/17-EJS1337SI>
   Supports regularized shrinkage for many weakly identified interactions.
   Established method; prior scales remain an empirical choice.

9. **Zaheer, M. et al. (2017), “Deep Sets.”**
   <https://papers.nips.cc/paper/2017/hash/f22e4747da1aa27e363d86d40ff442fe-Abstract.html>
   Supports permutation-invariant functions over set inputs. Established
   architecture; use for five-champion compositions is a candidate only.

10. **Lee, J. et al. (2019), “Set Transformer.”**
   <https://proceedings.mlr.press/v97/lee19d.html>
   Supports attention-based interactions within permutation-invariant sets.
   Established architecture; use is conditional on Scryglass ablations,
   uncertainty, and reconciliation.

11. **Rendle, S. (2010), “Factorization Machines.”**
   <https://doi.org/10.1109/ICDM.2010.127>
   Supports a compact low-rank representation of pairwise interactions in
   sparse high-dimensional feature spaces. It is a reproducible draft-model
   baseline/candidate, not proof that a factorization machine is sufficient for
   five-versus-five composition effects.

12. **Shapley, L. S. (1953), “A Value for n-Person Games.”**
   <https://www.rand.org/pubs/papers/P295.html>
   Basis for exact additive allocation of whole-set residuals. Established
   cooperative-game result; the Scryglass neutral baseline and cross-team
   allocation are proposed design choices.

## Probabilistic evaluation, calibration, and inference checks

13. **Gneiting, T. and Raftery, A. E. (2007), “Strictly Proper Scoring Rules,
    Prediction, and Estimation.”**
    <https://doi.org/10.1198/016214506000001437>
    Supports log loss and Brier score for honest probabilistic evaluation.
    Established method.

14. **Gneiting, T., Balabdaoui, F., and Raftery, A. E. (2007),
    “Probabilistic Forecasts, Calibration and Sharpness.”**
    <https://doi.org/10.1111/j.1467-9868.2007.00587.x>
    Supports treating calibration and sharpness together. Established method.

15. **Guo, C. et al. (2017), “On Calibration of Modern Neural Networks.”**
    <https://proceedings.mlr.press/v70/guo17a.html>
    Supports separately fitted calibration and evaluation of the resulting
    transform. It does not justify temperature scaling without comparison.

16. **Varma, S. and Simon, R. (2006), “Bias in Error Estimation When Using
    Cross-Validation for Model Selection.”**
    <https://doi.org/10.1186/1471-2105-7-91>
    Supports nested evaluation when selecting models/hyperparameters.
    Established method.

17. **Cameron, A. C., Gelbach, J. B., and Miller, D. L. (2008),
    “Bootstrap-Based Improvements for Inference with Clustered Errors.”**
    <https://www.nber.org/papers/t0344.pdf>
    Supports cluster-aware bootstrap inference. Scryglass treats series as
    inner blocks and preregisters higher-level participant/time dependence.

18. **Talts, S. et al. (2018), “Validating Bayesian Inference Algorithms with
    Simulation-Based Calibration.”**
    <https://arxiv.org/abs/1804.06788>
    Supports testing posterior computation on simulated known parameters.
    Established method.

## Partial-draft game policy

19. **McKelvey, R. D. and Palfrey, T. R. (1995), “Quantal Response Equilibria
    for Normal Form Games.”**
    <https://doi.org/10.1006/game.1995.1023>
    Supports probabilistic better-response behavior instead of assuming
    perfectly optimal choices. Established game-theory method; the sequential
    draft recursion is a proposed adaptation.

20. **Ziebart, B. D. et al. (2008), “Maximum Entropy Inverse Reinforcement
    Learning.”**
    <https://www.cs.cmu.edu/~bziebart/publications/maximum-entropy-inverse-reinforcement-learning.html>
    Supports maximum-entropy distributions over decision sequences. Established
    method; it motivates rather than mandates the Scryglass soft policy.

21. **Chen, Z. et al. (2018), “The Art of Drafting: A Team-Oriented Hero
    Recommendation System for Multiplayer Online Battle Arena Games.”**
    <https://doi.org/10.1145/3240323.3240345>
    Frames MOBA drafting as a combinatorial game and applies Monte Carlo tree
    search to hero recommendation. It is a field-adjacent baseline and
    motivation for explicit legal-state search, not evidence that its Dota 2
    estimand or evaluation transfers unchanged to professional League of
    Legends.

22. **Chen, S. et al. (2021), “Which Heroes to Pick? Learning to Draft in MOBA
    Games with Neural Networks and Tree Search” (JueWuDraft).**
    <https://arxiv.org/abs/2012.10171>
    Combines neural value estimation with Monte Carlo tree search for a
    multi-round Honor of Kings drafting game. It motivates scalable
    value/search baselines; protocol, title, outcome, and series rules differ,
    so it does not validate Scryglass semantics or establish SOTA here.

## Data, patch, and publication constraints

23. **Oracle's Elixir, Match Data Downloads.**
    <https://lol.timsevenhuysen.com/matchdata/>
    Identifies the primary public esports match-data source currently used by
    the repository. Redistribution rights are not inferred from availability;
    L1 must resolve them in the publication matrix.

24. **Riot Developer Portal, League of Legends/Data Dragon documentation.**
    <https://developer.riotgames.com/docs/lol>
    Supports official champion IDs/assets and patch-versioned static data.
    Source freshness and coverage still require audit.

25. **Riot Games Developer General Policies.**
    <https://developer.riotgames.com/policies/general>
    Supports registration, key security, legal attribution, monetization, and
    prohibition of betting/gambling functionality. Authoritative product
    constraint; policies are time-sensitive and require periodic review.

26. **Riot Games API Terms and Conditions.**
    <https://support-developer.riotgames.com/hc/en-us/articles/22698917218323-API-Terms-and-Conditions>
    Supports keeping API keys confidential and reviewing storage/use rights.
    Authoritative legal constraint, not a blanket publication license.

27. **JSON Schema Core and Validation, Draft 2020-12.**
    <https://json-schema.org/draft/2020-12/json-schema-core> and
    <https://json-schema.org/draft/2020-12/json-schema-validation>
    Defines the machine-contract dialect used in `contracts/`.

28. **Jiang, N. and Li, L. (2016), “Doubly Robust Off-policy Value
    Evaluation for Reinforcement Learning.”**
    <https://proceedings.mlr.press/v48/jiang16.html>
    Supports treating evaluation of a policy from data generated by another
    policy as a distinct sequential off-policy problem and motivates doubly
    robust sensitivity. It does not make positivity or exchangeability true for
    professional drafts.

29. **Cameron, A. C., Gelbach, J. B., and Miller, D. L. (2011), “Robust
    Inference With Multiway Clustering.”**
    <https://doi.org/10.1198/jbes.2010.07136>
    Supports inference when dependence is non-nested across more than one
    cluster dimension. Scryglass's participant/tournament/time design remains a
    preregistered application choice.

## Proposed Scryglass synthesis

The following combination is new to this product and must earn promotion:

- dynamic role-adjusted player states plus exact-roster team policy and lineup
  synergy on a common Elo display scale;
- a separately identifiable League Rating for globally eligible tier-1 teams;
- archetype-transfer priors plus champion-specific residuals;
- a neutral all-five/all-five composition model plus contextual
  player-champion/team-policy fit with baseline strength equalized;
- exact reconciled contribution ledgers for set and interaction residuals;
- a soft-minimax partial-draft graph with explicit flex role sets;
- model-standardized role×league×patch Tier Value with validated
  counterability; and
- evidence coverage, predictive reliability, stability, and 95% interval
  behavior as separate objects.

These are hypotheses encoded in a disciplined architecture, not SOTA findings.
Only the registered benchmark can promote them.
