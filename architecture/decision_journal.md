Decision Journal on Self learning and Meta Evaluation & Compliance
The first part is the self-learning improvement loop, including what I measured, how I evaluated, and why I made those choices. The second part is the metaevaluation loop, including how the system checks itself, what it found, and what changed.
We also add a section on how we maintained compliance at all times.

1) Self-Learning Approach

What to measure:
The self-learning pipeline looks at transcript level outcomes and quality signals. There were certain parameters used to do that. resolution, debt_collected, compliance_violation, tone_score, next_step_clarity. These were the metrics I used to eval the agents.

The main goal is that the resolution should not regress, and compliance should not decrease. Tone and clarity are supporting parameters.

How to eval:
The improvement ran in loops, we used batched transcripts for better eval. The loop ran in stages.
We had kept the transcript size 60 for auto eval loop run, which can be tweaked later for better results.

Stages of the run


Score baseline transcripts.
Analyze failure patterns.
Generate prompt improvments.
Run compliance checks.
Execute v2 on replayed or simulated conversations in stage 6a.
Compare v1 and v2 statistically in stage 6b.
Generate patch hypotheses and decide to adopt or reject.

Promotion decision is based on this logic: promote = confidence in {high, medium, directional} AND compliance_non_regression

For step 6, I also ran v2 on simulated conversations from the user of all the transcripts, because v2 was never used to talk to the customer, it was a llm playing as a borrower with the same tone as that of those transcripts.

We confidence tiers as of now with a score for promotion:
if significant and enough_effect and enough_delta:
confidence = "high"
elif p_value < AB_MEDIUM_MAX_P_VALUE and cohen_d >= AB_MEDIUM_MIN_COHEN_D and delta > AB_MEDIUM_MIN_DELTA:
confidence = "medium"
elif cohen_d >= AB_DIRECTIONAL_MIN_COHEN_D and delta >= AB_DIRECTIONAL_MIN_DELTA and n_v1 >= AB_DIRECTIONAL_MIN_N_PER_ARM and n_v2 >= AB_DIRECTIONAL_MIN_N_PER_ARM:
confidence = "directional"
else:
confidence = "low"



Why this methodology?
This was done to keep decisions stable and explainable.
The model summary can explain the result, but it did not decide the result.
Promotion should happen only when data showed a clear improvement as per our logic.
Compliance regression is hard gated and if the resolution is better, compliance can’t decrease at any cost. It either has to stay the same or increase as compared to previous.

Flow


Trade-off for this methodology:

Strong gating improves consistency, but can reject borderline real improvements.
Hard focus on resolution and compliance keeps decisions clear, but softer qualities are not taken much weighted into account.
Confidence tiers increase threshold tuning which can lead to overfitting!
I am doing fast small iterations which can bias towards short term gains but regress in long term behavioural quality.



2) Meta-Evaluation

What it does?
The meta evaluation checks L2 label consistency using an L3 shadow judge. It rescored the same transcripts with a stricter shadow prompt, compared L2 and L3 labels. Then it computed observed agreement, expected agreement, Cohen’s kappa, and disagreement counts.



Flow:


What it caught?
I was able to see these observations:

Some runs had low kappa even when overall positive rates looked similar.
Early recommendation text was too generic and not specific enough by agent.

The key point here was that evaluator reliability can drift even when high level numbers look good.


What changed?
Output was more objective and actionable

Recommendation text now includes concrete evidence such as kappa, disagreement ratio, L2 positive rate, L3 positive rate, and dominant mismatch direction.
Structured tuning proposals now include metric, action (add, remove, tighten), rationale, evidence count, and a 0 to 100 priority score.
Recommendations are now agent specific.

Trade offs?

L2 vs L3 agreement checks evaluator reliability, but still depends on prompt-based judges (not on ground-truth of humans).
Agent specific recos are actionable, but still heuristic in nature.
Reliability auditing catches inconsistency, but doesn’t yet automatically calibrate evaluator thresholds end-to-end

3) Compliance

Compliance was paramount in the workflow and has never been allowed to regress using stricter conditions.

How?
Compliance was kept as rules and was the north star metric to follow before making any changes and assessing or evaluating the prompts.

Checks:
The loop checks generated prompts and compares compliance posture with the current prompt.
Promotion is not based on outcome lift alone. Compliance non-regression is required.

Why?
Without hard compliance controls, the loop could improve conversion while quietly increasing policy or legal risk. So the rule is simple. A change is an improvement only if outcome quality improves and compliance is within certain threshold
version_comparison.improvement.compliance_non_regression == true






What could be better? What could be changed?

Build a full fusion meta layer (consistency + A/B testing + drift correlation) with one final confidence decision and explicit weights.
Add human labeled calibration set per agent, then periodically benchmark L2/L3 against that anchor.
Replace static thresholds with adaptive ones (Bayesian or rolling uncertainty-aware thresholds by agent/cohort).
Add counterfactual replay diversity packs (more borrower styles, edge case intents, adversarial phrasing).
Improve explainability UX: show exactly which gate failed and which metric change would have flipped decision. (for L3 evaluation)
Run offline replay tournament before adoption (candidate prompt vs current prompt across larger test banks) (Need Huge API credits! 🙂)
Do time-based prompt rotation for near scored prompts for each agent
Use RLHF like approach (use reward modelling and iterative tuning) but full online RL can be riskier too due to compliance.

